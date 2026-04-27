"""
Curated schema definitions from the 2-D convolution accelerator example.

This file contains the data schemas for the conv2d accelerator,
extracted from examples/conv2d/conv2d_demo.py for use as teaching examples.
"""
from __future__ import annotations

from enum import IntEnum

from pysilicon.hw.dataschema import DataList, EnumField, IntField, MemAddr


INCLUDE_DIR = "include"

# Hardware constants referenced by the schemas
PIXEL_BITWIDTH = 8   # Bit width of each input pixel
KERNEL_BITWIDTH = 8  # Total number of bits in the kernel including sign and fractional bits
KERNEL_FBITS = 7     # Number of fractional bits in the fixed-point kernel representation
MEM_AWIDTH = 64      # Memory address width in bits

# ---------------------------------------------------------------------------
# Reusable field specializations
# ---------------------------------------------------------------------------

AddrField = MemAddr.specialize(bitwidth=MEM_AWIDTH, include_dir=INCLUDE_DIR)
NrowField = IntField.specialize(bitwidth=16, signed=False, include_dir=INCLUDE_DIR)
NcolField = IntField.specialize(bitwidth=16, signed=False, include_dir=INCLUDE_DIR)
KernelSizeField = IntField.specialize(bitwidth=8, signed=False, include_dir=INCLUDE_DIR)

# ---------------------------------------------------------------------------
# Enums and enum fields
# ---------------------------------------------------------------------------


class Conv2DError(IntEnum):
    NO_ERROR = 0
    INVALID_NROWS = 1
    INVALID_NCOLS = 2
    INVALID_KSIZE = 3
    ADDRESS_ERROR = 4


class Conv2DEvent(IntEnum):
    MAIN_START = 0
    MAIN_END = 1
    LOAD_START = 2
    LOAD_END = 3
    COMPUTE_START = 4
    COMPUTE_END = 5
    STORE_START = 6
    STORE_END = 7


Conv2DErrorField = EnumField.specialize(
    enum_type=Conv2DError,
    include_dir=INCLUDE_DIR,
    include_filename="conv2d_error.h",
)
Conv2DEventField = EnumField.specialize(
    enum_type=Conv2DEvent,
    bitwidth=3,
    include_dir=INCLUDE_DIR,
    include_filename="conv2d_event.h",
)

# ---------------------------------------------------------------------------
# Command and response schemas
# ---------------------------------------------------------------------------


class Conv2DCmd(DataList):
    elements = {
        "nrows": {
            "schema": NrowField,
            "description": "Number of rows in the input image",
        },
        "ncols": {
            "schema": NcolField,
            "description": "Number of columns in the input image",
        },
        "kernel_size": {
            "schema": KernelSizeField,
            "description": "Size of the convolution kernel (assumed to be square)",
        },
        "input_addr": {
            "schema": AddrField,
            "description": (
                "Base address of the input image in external memory.  "
                "The image is stored in row-major order, meaning the first ncols elements "
                "correspond to the first row of the image, the next ncols elements correspond "
                "to the second row, and so on, with each pixel represented as an unsigned "
                "integer of bit width PIXEL_BITWIDTH."
            ),
        },
        "output_addr": {
            "schema": AddrField,
            "description": (
                "Base address of the output image in external memory.  "
                "The image is stored in the same row-major order as the input image, "
                "with each pixel represented as an unsigned integer of bit width PIXEL_BITWIDTH."
            ),
        },
        "kernel_addr": {
            "schema": AddrField,
            "description": (
                "Base address of the convolution kernel in external memory.  "
                "The kernel is stored in row-major order with each element represented as a "
                "fixed-point integer of bit width KERNEL_BITWIDTH with KERNEL_FBITS "
                "fractional bits."
            ),
        },
    }
    include_dir = INCLUDE_DIR
    include_filename = "conv2d_cmd.h"


class Conv2DResp(DataList):
    elements = {
        "error_code": {
            "schema": Conv2DErrorField,
            "description": "Error code indicating the status of the convolution operation",
        }
    }
    include_dir = INCLUDE_DIR
    include_filename = "conv2d_resp.h"


class Conv2DDebug(DataList):
    """
    A schema for sending internal debug information from the accelerator back
    to the testbench during co-simulation.
    """

    elements = {
        "row_ind": {
            "schema": NrowField,
            "description": "Row index associated with the debug event",
        },
        "event": {
            "schema": Conv2DEventField,
            "description": "Debug event emitted by the convolution accelerator",
        },
    }
    include_dir = INCLUDE_DIR
    include_filename = "conv2d_debug.h"
