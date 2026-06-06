from numpy.typing import NDArray
import numpy as np

# Vitis ap_fixed quantization (Q) and overflow (O) modes — the v1 subset.
# Quantization decides how the real value's sub-LSB bits become an integer;
# overflow decides what happens when that integer leaves the W-bit range.
AP_TRN = "AP_TRN"    # quantize: round toward minus infinity (floor) — Vitis default
AP_RND = "AP_RND"    # quantize: round half up (toward plus infinity on ties)
AP_WRAP = "AP_WRAP"  # overflow: two's-complement wrap (mask) — Vitis default
AP_SAT = "AP_SAT"    # overflow: saturate (clip to [min, max])

QUANT_MODES = (AP_TRN, AP_RND)
OVERFLOW_MODES = (AP_WRAP, AP_SAT)


def truncate(
    x: NDArray[np.int64] | int,
    wid: int = 32,
    signed: bool = True,
) -> NDArray[np.int64] | int:
    """
    Truncate integers to a signed or unsigned `wid`-bit two's-complement value.

    Parameters
    ----------
    x : int or NDArray[np.int64]
        Input integer(s) to be truncated.
    wid : int
        Target bit-width (including sign bit if signed=True).
    signed : bool
        If True, interpret the result as signed two's-complement.

    Returns
    -------
    y : int or NDArray[np.int64]
        The truncated value(s).
    """
    mask = (1 << wid) - 1
    y = x & mask

    if signed:
        signbit = 1 << (wid - 1)
        y = np.where(y >= signbit, y - (1 << wid), y)

    return y

def saturate(
    x: NDArray[np.int64] | int,
    wid: int = 32,
    signed: bool = True,
) -> NDArray[np.int64] | int:
    """
    Saturate integers to a signed or unsigned `wid`-bit two's-complement value.

    Parameters
    ----------
    x : int or NDArray[np.int64]
        Input integer(s) to be saturated.
    wid : int
        Target bit-width (including sign bit if signed=True).
    signed : bool
        If True, interpret the result as signed two's-complement.

    Returns
    -------
    y : int or NDArray[np.int64]
        The saturated value(s).
    """
    if signed:
        min_val = -(1 << (wid - 1))
        max_val = (1 << (wid - 1)) - 1
    else:
        min_val = 0
        max_val = (1 << wid) - 1
    y = np.clip(x, min_val, max_val)

    return y


def _validate_format(W: int, q_mode: str, o_mode: str) -> None:
    if W <= 0:
        raise ValueError(f"W (total bits) must be positive, got {W}.")
    if q_mode not in QUANT_MODES:
        raise ValueError(f"Unsupported quantization mode {q_mode!r}; v1 supports {QUANT_MODES}.")
    if o_mode not in OVERFLOW_MODES:
        raise ValueError(f"Unsupported overflow mode {o_mode!r}; v1 supports {OVERFLOW_MODES}.")


def quantize(
    value: NDArray[np.float64] | float,
    W: int,
    I: int,  # noqa: E741 — ap_fixed integer-bit count
    signed: bool = True,
    q_mode: str = AP_TRN,
    o_mode: str = AP_WRAP,
) -> NDArray[np.int64] | int:
    """Quantize real value(s) to the stored integer of ``ap_fixed<W, I, q, o>``.

    An ``ap_fixed<W, I>`` holds ``stored_int * 2**-(W-I)``: a ``W``-bit two's-
    complement (signed) or unsigned integer with an implied binary point ``I``
    bits from the MSB.  This returns that stored integer — bit-exact with what
    Vitis stores when the same value is assigned to the type.

    The pipeline is **quantize then overflow** (Vitis's order):

    1. Scale to LSB units: ``scaled = value * 2**(W-I)``.  Multiplying a float64
       by an exact power of two is an exact exponent shift, so for the v1 range
       (``W <= 53``, magnitudes well under ``2**53``) this and the floor below are
       exact — no rounding is introduced before the quantization being modeled.
    2. Quantize to an integer: ``AP_TRN`` → ``floor(scaled)`` (toward -inf, for
       positives and negatives alike); ``AP_RND`` → ``floor(scaled + 0.5)``
       (round half up, ties toward +inf).
    3. Overflow the W-bit range: ``AP_WRAP`` → two's-complement wrap
       (:func:`truncate`); ``AP_SAT`` → clip to ``[min, max]`` (:func:`saturate`).

    Fully vectorized: scalar in → ``int`` out, array in → ``int64`` array out.

    Notes
    -----
    For ``W > 53`` (beyond float64's exact-integer range) an exact int/``Fraction``
    backend would be required; v1 formats are ``W <= 53`` (compile-time template
    params). See ``tests/utils/test_fixputils.py`` for the exact ``Fraction``
    oracle this float64 path is proven bit-equal to.
    """
    _validate_format(W, q_mode, o_mode)
    F = W - I
    arr = np.asarray(value, dtype=np.float64)
    scaled = arr * (2.0 ** F)               # exact exponent shift for v1 range

    if q_mode == AP_TRN:
        q = np.floor(scaled)
    else:                                   # AP_RND — round half up (NOT np.round)
        q = np.floor(scaled + 0.5)
    q = q.astype(np.int64)

    out = truncate(q, W, signed) if o_mode == AP_WRAP else saturate(q, W, signed)
    out = np.asarray(out, dtype=np.int64)
    return int(out) if arr.ndim == 0 else out


def to_float(
    stored_int: NDArray[np.int64] | int,
    W: int,
    I: int,  # noqa: E741 — ap_fixed integer-bit count
) -> NDArray[np.float64] | float:
    """Inverse of :func:`quantize`: the real value ``stored_int * 2**-(W-I)``.

    Exact for the v1 range (``W <= 53``).  Vectorized (scalar → ``float``, array
    → ``float64`` array)."""
    F = W - I
    arr = np.asarray(stored_int, dtype=np.int64)
    out = arr * (2.0 ** (-F))
    return float(out) if arr.ndim == 0 else out


def to_bits(
    stored_int: NDArray[np.int64] | int,
    W: int,
) -> NDArray[np.uint64] | int:
    """The raw ``W``-bit unsigned bit pattern of the stored integer.

    This is what Vitis's ``.range()`` returns — the two's-complement bits for a
    signed value.  Vectorized."""
    if W <= 0:
        raise ValueError(f"W (total bits) must be positive, got {W}.")
    arr = np.asarray(stored_int, dtype=np.int64)
    out = arr & ((1 << W) - 1)
    return int(out) if arr.ndim == 0 else np.asarray(out, dtype=np.uint64)