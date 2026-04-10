"""
tests/utils/test_vcd.py – unit tests for pysilicon.utils.vcd.SigInfo numeric_values
storage convention.

The convention under test (for numeric_type == 'uint'):
  * wid <= 32  → np.ndarray, dtype np.uint32, shape (n,)
  * wid <= 64  → np.ndarray, dtype np.uint64, shape (n,)
  * wid > 64   → np.ndarray, dtype np.uint64, shape (n, k), k = ceil(wid/64),
                  word 0 = least-significant 64 bits (LSW-first)
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest

from vcdvcd import VCDVCD

from pysilicon.utils.vcd import SigInfo, VcdParser


# ---------------------------------------------------------------------------
# Minimal hand-written VCD fixture
# ---------------------------------------------------------------------------

# Signal definitions
#   a  – 8-bit  wire, identifier '!'
#   b  – 40-bit wire, identifier '"'
#   c  – 65-bit wire, identifier '#'
#   d  – 130-bit wire, identifier '$'
#
# Values chosen so that we can verify chunk packing for wide signals.

_VCD_TEXT = """\
$date today $end
$version test $end
$timescale 1ns $end
$scope module top $end
$var wire 8   ! a $end
$var wire 40  " b $end
$var wire 65  # c $end
$var wire 130 $ d $end
$upscope $end
$enddefinitions $end
#0
b00000001 !
b0000000000000000000000000000000000000001 "
b00000000000000000000000000000000000000000000000000000000000000001 #
b0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001 $
#10
b11111111 !
b1111111111111111111111111111111111111111 "
b10000000000000000000000000000000000000000000000000000000000000001 #
b1111111111111111111111111111111111111111111111111111111111111111110000000000000000000000000000000000000000000000000000000000000001 $
"""


@pytest.fixture
def vcd_path(tmp_path):
    """Write the VCD fixture to a temp file and return its path."""
    p = tmp_path / "test.vcd"
    p.write_text(_VCD_TEXT)
    return str(p)


@pytest.fixture
def parsed_signals(vcd_path):
    """
    Parse the VCD fixture and return a dict of {short_name: SigInfo} with
    numeric_values already computed.
    """
    vcd = VCDVCD(vcd_path)
    # Signal full names inside a 'top' scope use 'top.' prefix
    signals = {}
    for full_name in vcd.signals:
        short = full_name.split('.')[-1]
        tv = vcd[full_name].tv
        # Determine width from VCD variable declaration
        wid_map = {'a': 8, 'b': 40, 'c': 65, 'd': 130}
        wid = wid_map.get(short)
        si = SigInfo(full_name, tv, numeric_type='uint', wid=wid)
        si.get_values()
        signals[short] = si
    return signals


# ---------------------------------------------------------------------------
# Tests: dtype and shape
# ---------------------------------------------------------------------------

class TestUintStorageConvention:
    def test_wid_le32_dtype(self, parsed_signals):
        """8-bit signal must produce np.uint32 array."""
        si = parsed_signals['a']
        assert si.numeric_values.dtype == np.uint32, (
            f"Expected np.uint32, got {si.numeric_values.dtype}"
        )
        assert si.numeric_values.ndim == 1

    def test_wid_le64_dtype(self, parsed_signals):
        """40-bit signal must produce np.uint64 array."""
        si = parsed_signals['b']
        assert si.numeric_values.dtype == np.uint64, (
            f"Expected np.uint64, got {si.numeric_values.dtype}"
        )
        assert si.numeric_values.ndim == 1

    def test_wid_gt64_dtype_and_shape(self, parsed_signals):
        """65-bit signal must produce (n, 2) np.uint64 array."""
        si = parsed_signals['c']
        assert si.numeric_values.dtype == np.uint64
        assert si.numeric_values.ndim == 2
        k = math.ceil(65 / 64)  # == 2
        assert si.numeric_values.shape == (2, k), (
            f"Expected shape (2, {k}), got {si.numeric_values.shape}"
        )

    def test_wid_130_dtype_and_shape(self, parsed_signals):
        """130-bit signal must produce (n, 3) np.uint64 array."""
        si = parsed_signals['d']
        assert si.numeric_values.dtype == np.uint64
        assert si.numeric_values.ndim == 2
        k = math.ceil(130 / 64)  # == 3
        assert si.numeric_values.shape == (2, k), (
            f"Expected shape (2, {k}), got {si.numeric_values.shape}"
        )


# ---------------------------------------------------------------------------
# Tests: correct values
# ---------------------------------------------------------------------------

class TestUintValues:
    def test_8bit_values(self, parsed_signals):
        si = parsed_signals['a']
        assert si.numeric_values[0] == 1
        assert si.numeric_values[1] == 0xFF

    def test_40bit_values(self, parsed_signals):
        si = parsed_signals['b']
        assert si.numeric_values[0] == 1
        assert si.numeric_values[1] == (1 << 40) - 1

    def test_65bit_lsw_first(self, parsed_signals):
        """
        65-bit sample #0 = 1  → word[0]=1, word[1]=0
        65-bit sample #1 = (1 << 64) + 1 → word[0]=1, word[1]=1
        """
        si = parsed_signals['c']
        # sample 0
        assert si.numeric_values[0, 0] == np.uint64(1)
        assert si.numeric_values[0, 1] == np.uint64(0)
        # sample 1: value = 2^64 + 1
        assert si.numeric_values[1, 0] == np.uint64(1)
        assert si.numeric_values[1, 1] == np.uint64(1)

    def test_130bit_lsw_first(self, parsed_signals):
        """
        130-bit sample #0 = 1  → word[0]=1, word[1]=0, word[2]=0
        130-bit sample #1 = (2^129 - 2^64 + 1)
          binary = 1 followed by 65 ones, followed by 64 zeros, followed by 1
          Wait, let's recompute: the VCD value for sample #1 is:
            1111...1 (65 ones) 0000...0 (64 zeros) 0000...0001 (actually 1)

          Let's read the exact VCD string:
            b1111111111111111111111111111111111111111111111111111111111111111110000000000000000000000000000000000000000000000000000000000000001

          That is:
            65 ones followed by 64 zeros = 129 bits total, but wid=130, so
            zero-padded on the left to 130 bits:
            0_1111...1(65)_0000...0(64) = 0 followed by 65 ones followed by 64 zeros

          Actually let me count carefully:
            The binary string has 129 characters.
            With zero padding to 130: 0 + 129 chars.

          Numerical value:
            bits [129:65] = 65 ones   → contributes 2^129 - 2^65
            bits [64:1]   = 0 (64 zeros)
            bit  [0]      = 1

          So value = (2^65 - 1) * 2^65 + 1
                   = 2^130 - 2^65 + 1

          In 64-bit words (LSW first):
            word[0] = value & mask64 = 1
            word[1] = (value >> 64) & mask64 = (2^66 - 2) >> 0... 

          Let me just compute directly.
        """
        si = parsed_signals['d']

        # sample 0: value = 1
        assert si.numeric_values[0, 0] == np.uint64(1)
        assert si.numeric_values[0, 1] == np.uint64(0)
        assert si.numeric_values[0, 2] == np.uint64(0)

        # sample 1: compute expected words from the Python int representation
        # The VCD binary string (129 chars + 1 padding = 130-bit value):
        bin_str = "1111111111111111111111111111111111111111111111111111111111111111110000000000000000000000000000000000000000000000000000000000000001"
        value = int(bin_str, 2)
        mask = (1 << 64) - 1
        expected_w0 = np.uint64(value & mask)
        expected_w1 = np.uint64((value >> 64) & mask)
        expected_w2 = np.uint64((value >> 128) & mask)

        assert si.numeric_values[1, 0] == expected_w0
        assert si.numeric_values[1, 1] == expected_w1
        assert si.numeric_values[1, 2] == expected_w2
