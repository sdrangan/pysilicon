"""Exhaustive tests for the integer-backed fixed-point core.

Every vectorized path (quantization AND arithmetic) is proven bit-equal to an
independent exact oracle (``Fraction`` / arbitrary-precision Python ``int``), on
**array** inputs. The width guard is proven to fire (raise) on any declared or
derived format > 64 bits — including a sum-of-products whose accumulator exceeds
int64 — so a silent numpy wrap can never masquerade as a pass.
"""
import math
from fractions import Fraction

import numpy as np
import pytest

from waveflow.utils import fixputils as fp
from waveflow.utils.fixputils import (
    Format, OMode, QMode, add, add_format, fixed_sum, mult, mult_format, quantize,
    quantize_real, shift, sub, sub_format, sum_format, to_bits, to_float,
)

TRN, RND = QMode.AP_TRN, QMode.AP_RND
WRAP, SAT = OMode.AP_WRAP, OMode.AP_SAT


# --- exact oracles ------------------------------------------------------------
def _ovf(q, fmt):
    if fmt.o_mode == WRAP:
        y = q & ((1 << fmt.W) - 1)
        if fmt.signed and (y >> (fmt.W - 1)) & 1:
            y -= 1 << fmt.W
        return y
    if fmt.signed:
        lo, hi = -(1 << (fmt.W - 1)), (1 << (fmt.W - 1)) - 1
    else:
        lo, hi = 0, (1 << fmt.W) - 1
    return max(lo, min(hi, q))


def oracle_quantize_real(value, fmt):
    scaled = Fraction(value) * (Fraction(2) ** fmt.frac_bits)
    q = math.floor(scaled) if fmt.q_mode == TRN else math.floor(scaled + Fraction(1, 2))
    return _ovf(q, fmt)


def oracle_requantize(s_src, src, target):
    scaled = Fraction(int(s_src)) * (Fraction(2) ** (target.frac_bits - src.frac_bits))
    q = math.floor(scaled) if target.q_mode == TRN else math.floor(scaled + Fraction(1, 2))
    return _ovf(q, target)


# --- curated configs + edge values (carried over from the quantization ref) ---
CONFIGS = [
    Format(4, 2, True), Format(8, 4, True), Format(8, 4, False),
    Format(16, 8, True), Format(8, 8, True), Format(8, 0, True), Format(8, 8, False),
]


def edge_values(fmt):
    lsb = 2.0 ** (-fmt.frac_bits)
    if fmt.signed:
        hi = ((1 << (fmt.W - 1)) - 1) * lsb
        lo = -(1 << (fmt.W - 1)) * lsb
    else:
        hi = ((1 << fmt.W) - 1) * lsb
        lo = 0.0
    vals = [0.0, lsb, -lsb, 2 * lsb, -2 * lsb, 0.25 * lsb, -0.25 * lsb, 0.5 * lsb,
            -0.5 * lsb, 0.75 * lsb, -0.75 * lsb, 1.5 * lsb, -1.5 * lsb,
            hi, hi + 0.5 * lsb, hi + lsb, 8 * hi, lo, lo - 0.5 * lsb, lo - lsb, -8 * hi]
    if not fmt.signed:
        vals += [-0.5 * lsb, -lsb, -2.0]
    return vals


# --- Format + the width guard -------------------------------------------------
def test_format_basics():
    f = Format(8, 4, True)
    assert (f.frac_bits, f.dtype) == (4, np.int64)
    assert Format(8, 4, False).dtype == np.uint64


def test_width_guard_fires_on_declared_over_64():
    Format(64, 32)                               # 64 is allowed
    with pytest.raises(NotImplementedError):
        Format(65, 32)
    with pytest.raises(ValueError):
        Format(0, 0)


@pytest.mark.parametrize("q", [TRN, RND])
@pytest.mark.parametrize("o", [WRAP, SAT])
@pytest.mark.parametrize("base", CONFIGS)
def test_quantize_real_matches_oracle(base, q, o):
    fmt = Format(base.W, base.int_bits, base.signed, q, o)
    values = edge_values(fmt)
    got = quantize_real(np.array(values, dtype=np.float64), fmt)
    assert got.dtype == fmt.dtype
    for v, g in zip(values, got):
        assert int(g) == oracle_quantize_real(v, fmt), f"{fmt} value={v!r}"


def test_quantize_real_round_trip_on_representable():
    for base in CONFIGS:
        fmt = Format(base.W, base.int_bits, base.signed)
        codes = list(range(-(1 << (fmt.W - 1)), 1 << (fmt.W - 1)) if fmt.signed
                     else range(0, 1 << fmt.W))
        vals = np.array([c * 2.0 ** (-fmt.frac_bits) for c in codes], dtype=np.float64)
        stored = quantize_real(vals, fmt)
        np.testing.assert_array_equal(np.asarray(stored).astype(np.int64), np.array(codes))
        np.testing.assert_array_equal(np.asarray(to_float(stored, fmt)), vals)


# --- requantize (the integer arithmetic quantize) -----------------------------
@pytest.mark.parametrize("q", [TRN, RND])
@pytest.mark.parametrize("o", [WRAP, SAT])
def test_requantize_matches_oracle(q, o):
    src = Format(16, 8, True)               # a "product-like" wider source
    target = Format(8, 4, True, q, o)
    src_codes = np.arange(-(1 << 15), 1 << 15, 137, dtype=np.int64)
    got = quantize(src_codes, src, target)
    for s, g in zip(src_codes, got):
        assert int(g) == oracle_requantize(s, src, target), f"requant s={s} {q}/{o}"


def test_requantize_up_scale_exact():
    src = Format(8, 4, True)
    target = Format(12, 4, True)            # more fraction bits, no rounding
    codes = np.arange(-128, 128, dtype=np.int64)
    got = quantize(codes, src, target)
    for s, g in zip(codes, got):
        assert int(g) == oracle_requantize(s, src, target)


# --- arithmetic format derivation ---------------------------------------------
def test_format_derivation_rules():
    a, b = Format(8, 4, True), Format(8, 2, True)
    assert mult_format(a, b) == Format(16, 6, True)               # W,I add
    assert add_format(a, b) == Format(4 + 1 + max(4, 6), max(4, 2) + 1, True)
    assert sub_format(Format(8, 4, False), Format(8, 4, False)).signed is True
    assert fp.shift_format(a, 3) == Format(8, 7, True)            # bits unchanged, I+3
    assert sum_format(Format(8, 4, True), 16) == Format(12, 8, True)   # +ceil(log2 16)=4
    assert sum_format(Format(8, 4, True), 5) == Format(11, 7, True)    # ceil(log2 5)=3


def test_mixed_sign_raises():
    with pytest.raises(NotImplementedError):
        mult_format(Format(8, 4, True), Format(8, 4, False))
    with pytest.raises(NotImplementedError):
        add_format(Format(8, 4, True), Format(8, 4, False))


def test_width_guard_fires_on_derived_over_64():
    with pytest.raises(NotImplementedError):
        mult_format(Format(40, 20, True), Format(40, 20, True))   # -> W=80
    with pytest.raises(NotImplementedError):
        sum_format(Format(64, 32, True), 4)                       # -> W=66


# --- arithmetic values vs oracle (array inputs) -------------------------------
def test_mult_matches_oracle():
    a, b = Format(8, 4, True), Format(8, 4, True)
    sa = np.arange(-128, 128, 9, dtype=np.int64)
    sb = np.arange(127, -129, -9, dtype=np.int64)[: len(sa)]
    out, r = mult(sa, a, sb, b)
    assert r == Format(16, 8, True) and out.dtype == np.int64
    for x, y, o in zip(sa, sb, out):
        assert int(o) == int(x) * int(y)


def test_mult_w64_no_wrap():
    # s32_16 x s32_16 -> s64_32: product up to 2^62, must stay exact in int64.
    a = b = Format(32, 16, True)
    sa = np.array([2**31 - 1, -(2**31), 12345678, -98765432], dtype=np.int64)
    sb = np.array([2**31 - 1, 2**31 - 1, -76543210, 55555555], dtype=np.int64)
    out, r = mult(sa, a, sb, b)
    assert r.W == 64
    for x, y, o in zip(sa, sb, out):
        assert int(o) == int(x) * int(y)        # exact (Python-int oracle)


def test_add_sub_match_oracle():
    a, b = Format(8, 4, True), Format(8, 2, True)     # different F -> alignment
    sa = np.arange(-128, 128, 11, dtype=np.int64)
    sb = np.arange(-128, 128, 11, dtype=np.int64)
    for op, fmt_fn, sign in [(add, add_format, +1), (sub, sub_format, -1)]:
        out, r = op(sa, a, sb, b)
        assert r == fmt_fn(a, b)
        fr = r.frac_bits
        for x, y, o in zip(sa, sb, out):
            expect = (int(x) << (fr - a.frac_bits)) + sign * (int(y) << (fr - b.frac_bits))
            assert int(o) == expect


def test_shift_is_lossless_point_move():
    a = Format(8, 4, True)
    sa = np.arange(-128, 128, dtype=np.int64)
    out, r = shift(sa, a, 3)
    assert r == Format(8, 7, True)
    np.testing.assert_array_equal(np.asarray(out), sa)            # bits unchanged
    np.testing.assert_array_equal(np.asarray(to_float(out, r)), np.asarray(to_float(sa, a)) * 8)


# --- sum-of-products: <=64 computes exactly; >64 accumulation raises ----------
def test_sum_of_products_within_64_is_exact():
    # s24_12 . s24_12 -> products s48_24; dot N=16 -> acc s52_24 (<= 64), int64.
    fmt = Format(24, 12, True)
    rng = np.random.default_rng(0)
    n = 16
    sa = rng.integers(-(1 << 23), 1 << 23, size=n, dtype=np.int64)
    sb = rng.integers(-(1 << 23), 1 << 23, size=n, dtype=np.int64)
    prod, pf = mult(sa, fmt, sb, fmt)
    assert pf == Format(48, 24, True)
    acc, af = fixed_sum(prod, pf)
    assert af == Format(52, 28, True) and acc.dtype == np.int64   # I: 24 + ceil(log2 16)=4
    # exact arbitrary-precision oracle
    assert int(acc) == sum(int(x) * int(y) for x, y in zip(sa, sb))


def test_sum_of_products_beyond_64_raises():
    # s32_16 products are s64_32 (W=64 OK); summing 16 -> acc W=68 must RAISE.
    fmt = Format(32, 16, True)
    sa = np.arange(16, dtype=np.int64)
    sb = np.arange(16, dtype=np.int64)
    prod, pf = mult(sa, fmt, sb, fmt)
    assert pf.W == 64                          # the product itself is fine
    with pytest.raises(NotImplementedError):
        fixed_sum(prod, pf)                    # the accumulator would be 68 bits


# --- bit view -----------------------------------------------------------------
def test_to_bits_twos_complement():
    assert to_bits(-1, 4) == 15
    assert to_bits(7, 4) == 7
    np.testing.assert_array_equal(
        np.asarray(to_bits(np.array([-1, 7, -8], dtype=np.int64), 4)),
        np.array([15, 7, 8], dtype=np.uint64))
