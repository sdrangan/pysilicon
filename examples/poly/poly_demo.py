from __future__ import annotations

import argparse
import subprocess
from enum import IntEnum
from pathlib import Path

import numpy as np

from pysilicon.codegen.build import CodeGenConfig
from pysilicon.codegen.streamutils import copy_streamutils
from pysilicon.hw.dataschema import DataArray, DataList, EnumField, FloatField, IntField
from pysilicon.xilinxutils import toolchain


EXAMPLE_DIR = Path(__file__).resolve().parent
INCLUDE_DIR = "include"
WORD_BW_SUPPORTED = [32, 64]
U16 = IntField.specialize(bitwidth=16, signed=False)
F32 = FloatField.specialize(bitwidth=32)

"""
Define the data schemas for the polynomial accelerator
"""

class PolyError(IntEnum):
    NO_ERROR = 0
    WRONG_NSAMP = 1


PolyErrorField = EnumField.specialize(enum_type=PolyError, include_dir=INCLUDE_DIR)


class CoeffArray(DataArray):
    element_type = F32
    static = True
    max_shape = (4,)
    include_dir = INCLUDE_DIR


class PolyCmdHdr(DataList):
    elements = {
        "tx_id": {
            "schema": U16,
            "description": "Transaction ID",
        },
        "coeffs": {
            "schema": CoeffArray,
            "description": "Polynomial coefficients",
        },
        "nsamp": {
            "schema": U16,
            "description": "Number of samples",
        },
    }
    include_dir = INCLUDE_DIR


class PolyRespHdr(DataList):
    elements = {
        "tx_id": {
            "schema": U16,
            "description": "Echo of the transaction ID sent in the command",
        },
    }
    include_dir = INCLUDE_DIR


class PolyRespFtr(DataList):
    elements = {
        "nsamp_read": {
            "schema": U16,
            "description": "Number of samples returned in the response",
        },
        "error": {
            "schema": PolyErrorField,
            "description": "Error code indicating success or type of failure",
        },
    }
    include_dir = INCLUDE_DIR


class SampDataIn(DataArray):
    element_type = F32
    static = False
    max_shape = (128,)
    include_dir = INCLUDE_DIR


class SampDataOut(DataArray):
    element_type = F32
    static = False
    max_shape = (128,)
    include_dir = INCLUDE_DIR


SCHEMA_CLASSES = [
    PolyErrorField,
    CoeffArray,
    PolyCmdHdr,
    PolyRespHdr,
    PolyRespFtr,
    SampDataIn,
    SampDataOut,
]

"""
Python Golden model for polynomial evaluation
"""
def polynomial_eval(
    cmd_hdr: PolyCmdHdr,
    samp_in: SampDataIn,
) -> tuple[PolyRespHdr, SampDataOut, PolyRespFtr]:
    resp_hdr = PolyRespHdr()
    resp_hdr.tx_id = cmd_hdr.tx_id

    coeffs = np.asarray(cmd_hdr.coeffs, dtype=np.float32)
    x = np.asarray(samp_in.val, dtype=np.float32)
    y = np.zeros_like(x)
    power = np.ones_like(x)
    for coeff in coeffs:
        y += coeff * power
        power *= x

    samp_out = SampDataOut()
    samp_out.val = y

    resp_ftr = PolyRespFtr()
    resp_ftr.nsamp_read = len(x)
    resp_ftr.error = PolyError.NO_ERROR if len(x) == int(cmd_hdr.nsamp) else PolyError.WRONG_NSAMP

    return resp_hdr, samp_out, resp_ftr


"""
Building inputs, generating headers, writing vectors, running Vitis, and validating outputs
"""
def build_demo_inputs(nsamp: int = 100) -> tuple[PolyCmdHdr, SampDataIn]:
    coeffs = CoeffArray()
    coeffs.val = np.array([1.0, -2.0, -3.0, 4.0], dtype=np.float32)

    cmd_hdr = PolyCmdHdr()
    cmd_hdr.tx_id = 42
    cmd_hdr.coeffs = coeffs.val
    cmd_hdr.nsamp = nsamp

    samp_in = SampDataIn()
    samp_in.val = np.linspace(0.0, 1.0, nsamp, dtype=np.float32)
    return cmd_hdr, samp_in


def generate_headers(example_dir: Path) -> None:
    cfg = CodeGenConfig(root_dir=example_dir, util_dir=INCLUDE_DIR)
    for schema_class in SCHEMA_CLASSES:
        out_path = schema_class.gen_include(cfg=cfg, word_bw_supported=WORD_BW_SUPPORTED)
        print(f"generated {out_path}")
    copy_streamutils(cfg)


def write_vectors(example_dir: Path, cmd_hdr: PolyCmdHdr, samp_in: SampDataIn) -> Path:
    data_dir = example_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cmd_hdr.write_uint32_file(data_dir / "cmd_hdr_data.bin")
    samp_in.write_uint32_file(data_dir / "samp_in_data.bin", nwrite=cmd_hdr.nsamp)
    return data_dir


def validate_python_model(data_dir: Path, resp_hdr: PolyRespHdr, samp_out: SampDataOut, resp_ftr: PolyRespFtr) -> None:
    resp_hdr.write_uint32_file(data_dir / "resp_hdr_data.bin")
    samp_out.write_uint32_file(data_dir / "samp_out_data.bin", nwrite=resp_ftr.nsamp_read)
    resp_ftr.write_uint32_file(data_dir / "resp_ftr_data.bin")


def run_vitis(example_dir: Path) -> subprocess.CompletedProcess[str]:
    return toolchain.run_vitis_hls(example_dir / "run.tcl", work_dir=example_dir)


def check_vitis_outputs(
    data_dir: Path,
    expected_resp_hdr: PolyRespHdr,
    expected_samp_out: SampDataOut,
    expected_resp_ftr: PolyRespFtr,
) -> None:
    got_resp_hdr = PolyRespHdr().read_uint32_file(data_dir / "resp_hdr_data.bin")
    got_resp_ftr = PolyRespFtr().read_uint32_file(data_dir / "resp_ftr_data.bin")
    got_samp_out_words = np.fromfile(data_dir / "samp_out_data.bin", dtype="<u4")
    got_samp_out = got_samp_out_words.view("<f4")

    if not got_resp_hdr.is_close(expected_resp_hdr):
        raise RuntimeError("Response header mismatch after Vitis C-simulation.")
    if not got_resp_ftr.is_close(expected_resp_ftr):
        raise RuntimeError("Response footer mismatch after Vitis C-simulation.")
    if not np.allclose(got_samp_out, expected_samp_out.val[: got_samp_out.size], rtol=1e-6, atol=1e-6):
        raise RuntimeError("Sample output mismatch after Vitis C-simulation.")


def maybe_plot(cmd_hdr: PolyCmdHdr, samp_in: SampDataIn, samp_out: SampDataOut) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot")
        return

    plt.plot(samp_in.val, samp_out.val, label=f"tx_id={cmd_hdr.tx_id}")
    plt.xlabel("Input sample")
    plt.ylabel("Polynomial output")
    plt.title("Polynomial evaluation")
    plt.grid(True)
    plt.legend()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the polynomial accelerator example.")
    parser.add_argument("--nsamp", type=int, default=100, help="Number of input samples to generate.")
    parser.add_argument("--skip-vitis", action="store_true", help="Only run the Python side of the example.")
    parser.add_argument("--plot", action="store_true", help="Plot the Python golden-model output.")
    args = parser.parse_args()

    generate_headers(EXAMPLE_DIR)

    cmd_hdr, samp_in = build_demo_inputs(nsamp=args.nsamp)
    resp_hdr, samp_out, resp_ftr = polynomial_eval(cmd_hdr, samp_in)
    data_dir = write_vectors(EXAMPLE_DIR, cmd_hdr, samp_in)
    validate_python_model(data_dir, resp_hdr, samp_out, resp_ftr)

    print(f"wrote example data under {data_dir}")
    print(f"tx_id={resp_hdr.tx_id}, nsamp={resp_ftr.nsamp_read}, error={resp_ftr.error.name}")

    if args.plot:
        maybe_plot(cmd_hdr, samp_in, samp_out)

    if args.skip_vitis:
        return

    try:
        result = run_vitis(EXAMPLE_DIR)
    except RuntimeError as exc:
        print(f"skipping Vitis run: {exc}")
        return

    check_vitis_outputs(data_dir, resp_hdr, samp_out, resp_ftr)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    print("Vitis outputs matched the Python golden model.")


if __name__ == "__main__":
    main()
