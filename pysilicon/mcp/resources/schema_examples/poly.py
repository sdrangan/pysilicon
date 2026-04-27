"""
Curated schema definitions from the polynomial accelerator example.

This file contains the data schemas for the polynomial accelerator,
extracted from examples/poly/poly_demo.py for use as teaching examples.
"""
from __future__ import annotations

from enum import IntEnum

from pysilicon.hw.dataschema import DataArray, DataList, EnumField, FloatField, IntField


INCLUDE_DIR = "include"

# ---------------------------------------------------------------------------
# Reusable field specializations
# ---------------------------------------------------------------------------

TxIdField = IntField.specialize(bitwidth=16, signed=False)
NsampField = IntField.specialize(bitwidth=16, signed=False)
Float32 = FloatField.specialize(bitwidth=32, include_dir=INCLUDE_DIR)

# ---------------------------------------------------------------------------
# Enum and enum field for accelerator error codes
# ---------------------------------------------------------------------------


class PolyError(IntEnum):
    NO_ERROR = 0
    TLAST_EARLY_CMD_HDR = 1  # TLAST was asserted before the full command header was received
    NO_TLAST_CMD_HDR = 2  # The full command header was received but TLAST was never asserted
    TLAST_EARLY_SAMP_IN = 3  # TLAST was asserted before all input samples were received
    NO_TLAST_SAMP_IN = 4  # All input samples were received but TLAST was never asserted
    WRONG_NSAMP = 5  # The number of samples received does not match the expected number


PolyErrorField = EnumField.specialize(enum_type=PolyError, include_dir=INCLUDE_DIR)

# ---------------------------------------------------------------------------
# DataArray: polynomial coefficients
# ---------------------------------------------------------------------------


class CoeffArray(DataArray):
    """
    Array of polynomial coefficients, stored in ascending order (constant term first).
    For example, for a cubic polynomial c0 + c1*x + c2*x^2 + c3*x^3, the array would be
    [c0, c1, c2, c3].
    """

    ncoeff: int = 4
    element_type = Float32
    static = True
    max_shape = (ncoeff,)
    include_dir = INCLUDE_DIR


# ---------------------------------------------------------------------------
# Command and response schemas
# ---------------------------------------------------------------------------


class PolyCmdHdr(DataList):
    """
    Command header sent to the accelerator, containing the transaction ID,
    polynomial coefficients, and number of samples.
    The accelerator expects to receive exactly ``nsamp`` input samples after
    the command header, and will return ``nsamp`` output samples.
    """

    elements = {
        "tx_id": {
            "schema": TxIdField,
            "description": "Transaction ID",
        },
        "coeffs": {
            "schema": CoeffArray,
            "description": "Polynomial coefficients",
        },
        "nsamp": {
            "schema": NsampField,
            "description": "Number of samples",
        },
    }
    include_dir = INCLUDE_DIR


class PolyRespHdr(DataList):
    """
    Response header sent back from the accelerator, containing an echo of the
    transaction ID from the command header.  This allows the host to correlate
    responses with the commands that generated them, which is especially
    important if the accelerator is processing multiple commands concurrently.
    """

    elements = {
        "tx_id": {
            "schema": TxIdField,
            "description": "Echo of the transaction ID sent in the command",
        },
    }
    include_dir = INCLUDE_DIR


class PolyRespFtr(DataList):
    """
    Response footer sent back from the accelerator, containing the number of
    samples read and any error codes.
    """

    elements = {
        "nsamp_read": {
            "schema": NsampField,
            "description": "Number of samples returned in the response",
        },
        "error": {
            "schema": PolyErrorField,
            "description": "Error code indicating success or type of failure",
        },
    }
    include_dir = INCLUDE_DIR
