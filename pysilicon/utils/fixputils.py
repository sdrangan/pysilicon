from numpy.typing import NDArray
import numpy as np

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