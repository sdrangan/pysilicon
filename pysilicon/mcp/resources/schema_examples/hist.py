"""
Curated schema definitions from the histogram accelerator example.

This file contains the data schemas for the histogram accelerator,
extracted from examples/histogram/hist_demo.py for use as teaching examples.
"""
from __future__ import annotations

from enum import IntEnum

from pysilicon.hw.dataschema import DataList, EnumField, FloatField, IntField, MemAddr


INCLUDE_DIR = "include"

# ---------------------------------------------------------------------------
# Reusable field specializations
# ---------------------------------------------------------------------------

TxIdField = IntField.specialize(bitwidth=16, signed=False)
NdataField = IntField.specialize(bitwidth=32, signed=False)
NbinField = IntField.specialize(bitwidth=32, signed=False)
Float32 = FloatField.specialize(bitwidth=32, include_dir=INCLUDE_DIR)
AddrField = MemAddr.specialize(bitwidth=64, include_dir=INCLUDE_DIR)

# ---------------------------------------------------------------------------
# Enum and enum field for histogram error codes
# ---------------------------------------------------------------------------


class HistError(IntEnum):
    NO_ERROR = 0
    INVALID_NDATA = 1
    INVALID_NBINS = 2
    ADDRESS_ERROR = 3


HistErrorField = EnumField.specialize(enum_type=HistError, include_dir=INCLUDE_DIR)

# ---------------------------------------------------------------------------
# Command and response schemas
# ---------------------------------------------------------------------------


class HistCmd(DataList):
    """Command descriptor for the histogram accelerator."""

    elements = {
        "tx_id": {
            "schema": TxIdField,
            "description": "Transaction ID for correlating command and response",
        },
        "data_addr": {
            "schema": AddrField,
            "description": "Base address of the input data buffer",
        },
        "bin_edges_addr": {
            "schema": AddrField,
            "description": (
                "Base address of the output histogram bin edges.  "
                "There should be nbins-1 edge values.  "
                "bin 0 will have values x < bin_edges[0], bin i will have values "
                "bin_edges[i-1] <= x < bin_edges[i], and the last bin will have values "
                "x >= bin_edges[nbins-2]"
            ),
        },
        "ndata": {
            "schema": NdataField,
            "description": "Number of input data elements to histogram",
        },
        "nbins": {
            "schema": NbinField,
            "description": "Number of histogram bins to produce",
        },
        "cnt_addr": {
            "schema": AddrField,
            "description": "Base address of the output histogram counts buffer",
        },
    }
    include_dir = INCLUDE_DIR


class HistResp(DataList):
    """Response descriptor returned by the histogram accelerator."""

    elements = {
        "tx_id": {
            "schema": TxIdField,
            "description": "Echo of the transaction ID from the command",
        },
        "status": {
            "schema": HistErrorField,
            "description": "Histogram execution status code",
        },
    }
    include_dir = INCLUDE_DIR
