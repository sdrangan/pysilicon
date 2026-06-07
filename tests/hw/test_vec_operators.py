"""Phase 1: type-preserving arithmetic operators on DataArray.

Operators (`+`/`-`/`*`) are full-precision sugar over the per-type math: fixed reuses
the FixedField functions (operator == function), int grows (a*b -> Wa+Wb, a+b ->
max+1) under the shared single-64-bit fail-fast guard, float is numpy passthrough.
Rounding stays an explicit quantize(); `.val` stays the numpy escape.
"""
import numpy as np
import pytest

from waveflow.hw.dataschema import DataArray, FloatField, IntField
from waveflow.hw.fixpoint import (
    FixedField, add, from_real, mult, quantize, sub, to_real,
)


def iarr(W, signed, vals):
    return DataArray.specialize(IntField.specialize(W, signed), max_shape=(len(vals),))(vals)


def farr(bw, vals):
    return DataArray.specialize(FloatField.specialize(bw), max_shape=(len(vals),))(vals)


# --- fixed: operators are exactly the free functions -------------------------
@pytest.mark.parametrize("opsym,fn", [("mul", mult), ("add", add), ("sub", sub)])
def test_fixed_operator_equals_function(opsym, fn):
    Q = FixedField.specialize(8, 4)
    a, b = from_real([1.5, -2.0, 0.5, 7.0], Q), from_real([2.0, 1.5, -1.0, 1.0], Q)
    op = {"mul": a * b, "add": a + b, "sub": a - b}[opsym]
    ref = fn(a, b)
    assert op.element_type.cpp_type == ref.element_type.cpp_type
    np.testing.assert_array_equal(np.asarray(op), np.asarray(ref))


def test_fixed_full_precision_no_loss_then_explicit_quantize():
    Q = FixedField.specialize(8, 4)
    a, b = from_real([1.5, -2.0, 0.5], Q), from_real([2.0, 1.5, -1.0], Q)
    c = from_real([0.5, 0.5, 0.5], Q)
    y = a * b + c                                  # full precision, no rounding
    assert y.element_type.bitwidth > Q.bitwidth    # grew
    np.testing.assert_array_equal(to_real(y), to_real(a) * to_real(b) + to_real(c))
    q = quantize(y, Q)                             # rounding only here
    assert q.element_type.cpp_type == Q.cpp_type


# --- int: growth-aware, exact ------------------------------------------------
def test_int_growth_and_values():
    a, b = iarr(8, True, [3, -4, 5]), iarr(8, True, [6, 7, -8])
    p, s, d = a * b, a + b, a - b
    assert p.element_type.get_bitwidth() == 16            # Wa+Wb
    assert s.element_type.get_bitwidth() == 9             # max+1
    assert d.element_type.get_bitwidth() == 9 and d.element_type.signed
    np.testing.assert_array_equal(np.asarray(p), [18, -28, -40])
    np.testing.assert_array_equal(np.asarray(s), [9, 3, -3])
    np.testing.assert_array_equal(np.asarray(d), [-3, -11, 13])


def test_int_unsigned_and_a_mul_b_plus_c():
    a, b = iarr(8, False, [200, 100]), iarr(8, False, [3, 2])
    p = a * b
    assert p.element_type.get_bitwidth() == 16 and not p.element_type.signed
    np.testing.assert_array_equal(np.asarray(p), [600, 200])      # exact, no int8 wrap
    # a*b + c full precision matches plain numpy int
    c = iarr(8, True, [1, -1, 2])
    x, y = iarr(8, True, [3, -4, 5]), iarr(8, True, [6, 7, -8])
    np.testing.assert_array_equal(np.asarray(x * y + c),
                                  np.asarray(x).astype(np.int64) * np.asarray(y) + np.asarray(c))


# --- float: numpy passthrough, no growth -------------------------------------
def test_float_passthrough():
    fa, fb = farr(32, [1.5, 2.5, 3.5]), farr(32, [2.0, 2.0, 2.0])
    p = fa * fb
    assert p.element_type.get_bitwidth() == 32           # no growth
    np.testing.assert_array_equal(np.asarray(p), np.float32([3.0, 5.0, 7.0]))
    np.testing.assert_array_equal(np.asarray(fa * fb + fa), np.asarray(fa) * np.asarray(fb) + np.asarray(fa))


# --- guards ------------------------------------------------------------------
def test_over_64_raises_int_and_fixed():
    big = iarr(40, True, [1, 2])
    with pytest.raises(NotImplementedError):
        big * big                                         # 80 bits
    A = FixedField.specialize(40, 20)
    fa = from_real([1.0], A)
    with pytest.raises(NotImplementedError):
        fa * fa


def test_mixed_sign_raises_int_and_fixed():
    with pytest.raises(NotImplementedError):
        iarr(8, True, [1]) * iarr(8, False, [1])
    with pytest.raises(NotImplementedError):
        from_real([1.0], FixedField.specialize(8, 4, signed=True)) * \
            from_real([1.0], FixedField.specialize(8, 4, signed=False))


def test_cross_kind_and_scalar_operands_raise():
    with pytest.raises(TypeError):
        iarr(8, True, [1]) * farr(32, [1.0])              # int x float
    with pytest.raises(TypeError):
        iarr(8, True, [1, 2]) * 2                          # scalar operand (v1)


def test_existing_fixedfield_math_unchanged():
    # operators are pure sugar; the underlying functions/bits are untouched
    Q = FixedField.specialize(8, 4, q_mode=FixedField.specialize(8, 4).q_mode)
    a, b = from_real([3.3, -3.9], Q), from_real([2.5, 1.1], Q)
    np.testing.assert_array_equal(np.asarray(a * b), np.asarray(mult(a, b)))
