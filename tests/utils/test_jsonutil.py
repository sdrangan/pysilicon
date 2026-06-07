"""Unit tests for waveflow.utils.jsonutil (json_scalar, hex_word)."""
from __future__ import annotations

import numpy as np
import pytest

from waveflow.utils.jsonutil import hex_word, json_scalar


def test_json_scalar_coerces_numpy_scalars():
    out_i = json_scalar(np.int64(7))
    assert out_i == 7 and isinstance(out_i, int)
    out_f = json_scalar(np.float32(1.5))
    assert out_f == 1.5 and isinstance(out_f, float)


def test_json_scalar_passes_through_native_values():
    # Non-numpy values are returned unchanged (the same object).
    for v in (3, 2.5, "x", None, [1, 2]):
        assert json_scalar(v) is v


def test_hex_word_fixed_width_masked():
    assert hex_word(255, 8) == "0xff"
    assert hex_word(0, 16) == "0x0000"
    assert hex_word(np.uint32(0xABCD), 16) == "0xabcd"
    assert hex_word(0x1FF, 8) == "0xff"          # masks to width
    assert hex_word(1, 12) == "0x001"            # ceil(bitwidth/4) digits


def test_hex_word_rejects_nonpositive_bitwidth():
    with pytest.raises(ValueError):
        hex_word(1, 0)
    with pytest.raises(ValueError):
        hex_word(1, -4)
