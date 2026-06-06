"""Exhaustive Python tests for the ap_fixed quantization engine.

The float64-vectorized :func:`quantize` is proven bit-equal to an independent
**exact** scalar oracle (`Fraction` + integer floor), over the curated configs and
the edge-value vectors (exact, rounding midpoints, min/max overflow, negatives,
unsigned-negative inputs).  The specific-edge tests document the exact semantics
the Phase-4 Vitis conformance run must confirm.
"""
import math
from fractions import Fraction

import numpy as np
import pytest

from pysilicon.utils.fixputils import (
    AP_RND, AP_SAT, AP_TRN, AP_WRAP, OVERFLOW_MODES, QUANT_MODES,
    quantize, to_bits, to_float,
)


# --- exact scalar oracle: the spec, via rationals (no float rounding) ---------
def oracle_quantize(value, W, I, signed, q_mode, o_mode):  # noqa: E741
    F = W - I
    scaled = Fraction(value) * (Fraction(2) ** F)
    q = math.floor(scaled) if q_mode == AP_TRN else math.floor(scaled + Fraction(1, 2))
    if o_mode == AP_WRAP:
        y = q & ((1 << W) - 1)
        if signed and (y >> (W - 1)) & 1:
            y -= 1 << W
        return y
    if signed:
        lo, hi = -(1 << (W - 1)), (1 << (W - 1)) - 1
    else:
        lo, hi = 0, (1 << W) - 1
    return max(lo, min(hi, q))


# --- curated configs (W, I, signed) -------------------------------------------
CONFIGS = [
    (4, 2, True),    # tiny, near-exhaustive
    (8, 4, True),    # mid-width
    (8, 4, False),   # unsigned fraction
    (16, 8, True),   # wider
    (8, 8, True),    # F=0 integer (overflow only)
    (8, 0, True),    # pure fractional [-0.5, 0.5)
    (8, 8, False),   # F=0 unsigned integer
]


def edge_values(W, I, signed):  # noqa: E741
    F = W - I
    lsb = 2.0 ** (-F)
    if signed:
        max_repr = ((1 << (W - 1)) - 1) * lsb
        min_repr = -(1 << (W - 1)) * lsb
    else:
        max_repr = ((1 << W) - 1) * lsb
        min_repr = 0.0
    vals = [
        0.0, lsb, -lsb, 2 * lsb, -2 * lsb,                       # exact
        0.25 * lsb, -0.25 * lsb, 0.5 * lsb, -0.5 * lsb,          # rounding
        0.75 * lsb, -0.75 * lsb, 1.5 * lsb, -1.5 * lsb,
        max_repr, max_repr + 0.5 * lsb, max_repr + lsb, 8 * max_repr,   # +overflow
        min_repr, min_repr - 0.5 * lsb, min_repr - lsb, -8 * max_repr,  # -overflow
    ]
    if not signed:
        vals += [-0.5 * lsb, -lsb, -2.0]   # negative inputs to an unsigned format
    return vals


@pytest.mark.parametrize("o_mode", OVERFLOW_MODES)
@pytest.mark.parametrize("q_mode", QUANT_MODES)
@pytest.mark.parametrize("W,I,signed", CONFIGS)
def test_quantize_matches_fraction_oracle(W, I, signed, q_mode, o_mode):  # noqa: E741
    """The float64 path equals the exact Fraction oracle, bit-for-bit."""
    for v in edge_values(W, I, signed):
        got = quantize(v, W, I, signed, q_mode, o_mode)
        exp = oracle_quantize(v, W, I, signed, q_mode, o_mode)
        assert got == exp, (
            f"<W={W},I={I},signed={signed}> {q_mode}/{o_mode} value={v!r}: "
            f"got {got}, oracle {exp}")


@pytest.mark.parametrize("o_mode", OVERFLOW_MODES)
@pytest.mark.parametrize("q_mode", QUANT_MODES)
@pytest.mark.parametrize("W,I,signed", CONFIGS)
def test_vectorized_equals_scalar(W, I, signed, q_mode, o_mode):  # noqa: E741
    """Array-in/array-out matches the per-element scalar results (numpy vectorized)."""
    vals = edge_values(W, I, signed)
    arr = quantize(np.array(vals, dtype=np.float64), W, I, signed, q_mode, o_mode)
    assert isinstance(arr, np.ndarray) and arr.dtype == np.int64
    for v, a in zip(vals, arr):
        assert int(a) == quantize(v, W, I, signed, q_mode, o_mode)


@pytest.mark.parametrize("W,I,signed", CONFIGS)
def test_round_trip_identity_on_representable(W, I, signed):  # noqa: E741
    """Every exactly-representable value survives quantize -> to_float unchanged."""
    F = W - I
    lsb = 2.0 ** (-F)
    codes = list(range(-(1 << (W - 1)), 1 << (W - 1)) if signed else range(0, 1 << W))
    if len(codes) > 64:
        codes = codes[:: len(codes) // 64]
    for c in codes:
        v = c * lsb
        st = quantize(v, W, I, signed, AP_TRN, AP_WRAP)
        assert st == c, f"representable {v!r} (code {c}) quantized to {st}"
        assert to_float(st, W, I) == v


# --- specific edge semantics (the spec the Vitis run must confirm) ------------
def test_ap_trn_floors_negatives():
    lsb = 1 / 16   # ap_fixed<8,4>
    assert quantize(-0.5 * lsb, 8, 4, True, AP_TRN, AP_WRAP) == -1   # floor(-0.5) toward -inf


def test_ap_rnd_ties_round_half_up():
    lsb = 1 / 16
    assert quantize(0.5 * lsb, 8, 4, True, AP_RND, AP_WRAP) == 1     # +0.5 -> +1
    assert quantize(-0.5 * lsb, 8, 4, True, AP_RND, AP_WRAP) == 0    # -0.5 -> 0 (toward +inf)
    assert quantize(-1.5 * lsb, 8, 4, True, AP_RND, AP_WRAP) == -1   # -1.5 -> -1


def test_ap_sat_asymmetric_signed():
    assert quantize(100.0, 4, 4, True, AP_TRN, AP_SAT) == 7     # range [-8, 7]
    assert quantize(-100.0, 4, 4, True, AP_TRN, AP_SAT) == -8   # asymmetric min


def test_rounding_induced_overflow():
    # 7.5 in ap_fixed<4,4>: AP_RND floors(7.5+0.5)=8, out of [-8,7].
    assert quantize(7.5, 4, 4, True, AP_RND, AP_SAT) == 7       # saturates to max
    assert quantize(7.5, 4, 4, True, AP_RND, AP_WRAP) == -8     # wraps


def test_unsigned_negative_input():
    # ap_ufixed<8,8>: -1.0 -> floor(-1) = -1, out of [0, 255].
    assert quantize(-1.0, 8, 8, False, AP_TRN, AP_WRAP) == 255   # wraps
    assert quantize(-1.0, 8, 8, False, AP_TRN, AP_SAT) == 0      # saturates to 0


def test_to_bits_twos_complement():
    assert to_bits(-1, 4) == 15
    assert to_bits(7, 4) == 7
    np.testing.assert_array_equal(
        np.asarray(to_bits(np.array([-1, 7, -8]), 4)), np.array([15, 7, 8], dtype=np.uint64))


def test_invalid_modes_and_width_raise():
    with pytest.raises(ValueError):
        quantize(1.0, 8, 4, True, "AP_RND_CONV", AP_WRAP)   # not in v1
    with pytest.raises(ValueError):
        quantize(1.0, 8, 4, True, AP_TRN, "AP_SAT_SYM")     # not in v1
    with pytest.raises(ValueError):
        quantize(1.0, 0, 0)                                 # W must be positive
