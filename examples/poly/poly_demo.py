from __future__ import annotations

import argparse
import json
import subprocess
from enum import IntEnum
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pysilicon.build.build import CodeGenConfig
from pysilicon.build.streamutils import copy_streamutils
from pysilicon.hw.arrayutils import gen_array_utils, read_uint32_file, write_uint32_file
from pysilicon.hw.dataschema import DataArray, DataList, EnumField, FloatField, IntField
from pysilicon.toolchain import toolchain


EXAMPLE_DIR = Path(__file__).resolve().parent
INCLUDE_DIR = "include"
WORD_BW_SUPPORTED = [32, 64]
TxIdField = IntField.specialize(bitwidth=16, signed=False)
NsampField = IntField.specialize(bitwidth=16, signed=False)
Float32 = FloatField.specialize(bitwidth=32, include_dir=INCLUDE_DIR)

"""
Define the data schemas for the polynomial accelerator
"""

# Error codes for polynomial evaluation results
class PolyError(IntEnum):
    NO_ERROR = 0
    TLAST_EARLY_CMD_HDR = 1  # TLAST was asserted before the full command header was received
    NO_TLAST_CMD_HDR = 2  # The full command header was received but TLAST was never asserted
    TLAST_EARLY_SAMP_IN = 3  # TLAST was asserted before all input samples were received
    NO_TLAST_SAMP_IN = 4  # All input samples were received but TLAST was never asserted
    WRONG_NSAMP = 5  # The number of samples received does not match the expected number
PolyErrorField = EnumField.specialize(enum_type=PolyError, include_dir=INCLUDE_DIR)



class CoeffArray(DataArray):
    """
    Array of polynomial coefficients, stored in ascending order (constant term first).
    For example, for a cubic polynomial c0 + c1*x + c2*x^2 + c3*x^3, the array would be [c0, c1, c2, c3].
    """
    ncoeff: int = 4
    element_type = Float32
    static = True
    max_shape = (ncoeff,)
    include_dir = INCLUDE_DIR


class PolyCmdHdr(DataList):
    """
    Command header sent to the accelerator, containing the transaction ID, polynomial coefficients, and number of samples.
    The accelerator expects to receive exactly `nsamp` input samples after the command header, and will return `nsamp` output samples.
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
    Response header sent back from the accelerator, containing an echo of the transaction ID from the command header.
    This allows the host to correlate responses with the commands that generated them, which is especially important if
    the accelerator is processing multiple commands concurrently.
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
    Response footer sent back from the accelerator, containing the number of samples read and any error codes.
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


SCHEMA_CLASSES = [
    PolyErrorField,
    CoeffArray,
    PolyCmdHdr,
    PolyRespHdr,
    PolyRespFtr,
]

"""
Python Golden model for polynomial evaluation
"""
def polynomial_eval(
    cmd_hdr: PolyCmdHdr,
    samp_in: npt.NDArray[np.float32],
) -> tuple[PolyRespHdr, npt.NDArray[np.float32], PolyRespFtr]:
    resp_hdr = PolyRespHdr()
    resp_hdr.tx_id = cmd_hdr.tx_id

    coeffs = np.asarray(cmd_hdr.coeffs, dtype=np.float32)
    x = np.asarray(samp_in, dtype=np.float32)
    y = np.zeros_like(x)
    power = np.ones_like(x)
    for coeff in coeffs:
        y += coeff * power
        power *= x

    resp_ftr = PolyRespFtr()
    resp_ftr.nsamp_read = len(x)
    resp_ftr.error = PolyError.NO_ERROR if len(x) == int(cmd_hdr.nsamp) else PolyError.WRONG_NSAMP

    return resp_hdr, y, resp_ftr


"""
Building inputs, generating headers, writing vectors, running Vitis, and validating outputs
"""
def build_demo_inputs(nsamp: int = 100) -> tuple[PolyCmdHdr, npt.NDArray[np.float32]]:
    coeffs = CoeffArray()
    coeffs.val = np.array([1.0, -2.0, -3.0, 4.0], dtype=np.float32)

    cmd_hdr = PolyCmdHdr()
    cmd_hdr.tx_id = 42
    cmd_hdr.coeffs = coeffs.val
    cmd_hdr.nsamp = nsamp

    samp_in = np.linspace(0.0, 1.0, nsamp, dtype=np.float32)
    return cmd_hdr, samp_in


def generate_headers(example_dir: Path) -> None:
    cfg = CodeGenConfig(root_dir=example_dir, util_dir=INCLUDE_DIR)
    for schema_class in SCHEMA_CLASSES:
        out_path = schema_class.gen_include(cfg=cfg, word_bw_supported=WORD_BW_SUPPORTED)
        print(f"generated {out_path}")
    out_path = gen_array_utils(Float32, WORD_BW_SUPPORTED, cfg=cfg)
    print(f"generated {out_path}")
    copy_streamutils(cfg)


def write_vectors(example_dir: Path, cmd_hdr: PolyCmdHdr, samp_in: npt.NDArray[np.float32]) -> Path:
    data_dir = example_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cmd_hdr.write_uint32_file(data_dir / "cmd_hdr_data.bin")
    write_uint32_file(samp_in, elem_type=Float32, file_path=data_dir / "samp_in_data.bin", nwrite=cmd_hdr.nsamp)
    return data_dir


def validate_python_model(data_dir: Path, resp_hdr: PolyRespHdr, samp_out: npt.NDArray[np.float32], resp_ftr: PolyRespFtr) -> None:
    resp_hdr.write_uint32_file(data_dir / "resp_hdr_data.bin")
    write_uint32_file(samp_out, elem_type=Float32, file_path=data_dir / "samp_out_data.bin", nwrite=resp_ftr.nsamp_read)
    resp_ftr.write_uint32_file(data_dir / "resp_ftr_data.bin")


def run_vitis(example_dir: Path) -> subprocess.CompletedProcess[str]:
    return toolchain.run_vitis_hls(example_dir / "run.tcl", work_dir=example_dir)


def report_vitis_synthesis(example_dir: Path) -> None:
    try:
        from pysilicon.utils.csynthparse import CsynthParser
    except ModuleNotFoundError as exc:
        print(f"Skipping csynth report parsing: {exc}")
        return

    sol_path = example_dir / "pysilicon_poly_proj" / "solution1"
    if not sol_path.exists():
        print(f"Skipping csynth report parsing: solution directory not found at {sol_path}")
        return

    try:
        parser = CsynthParser(sol_path=str(sol_path))
        parser.get_loop_pipeline_info()
        parser.get_resources()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Skipping csynth report parsing: {exc}")
        return

    print("\nLatency and Initiation Interval:")
    if parser.loop_df.empty:
        print("No loop pipeline information found in csynth.xml.")
    else:
        print(parser.loop_df.to_string())
        non_unit_ii = parser.loop_df[
            parser.loop_df["PipelineII"].apply(lambda value: isinstance(value, (int, np.integer)) and value > 1)
        ]
        if non_unit_ii.empty:
            print("All reported loops have PipelineII <= 1.")
        else:
            print("Loops with PipelineII > 1:")
            print(non_unit_ii.to_string())
            raise RuntimeError(
                "Vitis synthesis produced loops with PipelineII > 1. "
                "See the loop pipeline report above."
            )

    print("\nResource Usage:")
    if parser.res_df.empty:
        print("No resource information found in csynth.xml.")
    else:
        print(parser.res_df.to_string())


def check_vitis_outputs(
    data_dir: Path,
    expected_resp_hdr: PolyRespHdr,
    expected_samp_out: npt.NDArray[np.float32],
    expected_resp_ftr: PolyRespFtr,
) -> None:
    got_resp_hdr = PolyRespHdr().read_uint32_file(data_dir / "resp_hdr_data.bin")
    got_resp_ftr = PolyRespFtr().read_uint32_file(data_dir / "resp_ftr_data.bin")
    got_samp_out = np.asarray(
        read_uint32_file(data_dir / "samp_out_data.bin", elem_type=Float32, shape=int(expected_resp_ftr.nsamp_read)),
        dtype=np.float32,
    )

    if not got_resp_hdr.is_close(expected_resp_hdr):
        raise RuntimeError("Response header mismatch after Vitis C-simulation.")
    if not got_resp_ftr.is_close(expected_resp_ftr):
        raise RuntimeError("Response footer mismatch after Vitis C-simulation.")
    if not np.allclose(got_samp_out, expected_samp_out[: got_samp_out.size], rtol=1e-6, atol=1e-6):
        raise RuntimeError("Sample output mismatch after Vitis C-simulation.")

    sync_status_path = data_dir / "sync_status.json"
    if sync_status_path.exists():
        sync_status = json.loads(sync_status_path.read_text(encoding="utf-8"))
        expected_sync = {
            "resp_hdr_tlast": "tlast_at_end",
            "samp_out_tlast": "tlast_at_end",
            "resp_ftr_tlast": "tlast_at_end",
        }
        if sync_status != expected_sync:
            raise RuntimeError(
                "Output-stream TLAST synchronization mismatch after Vitis C-simulation. "
                f"Expected {expected_sync}, got {sync_status}."
            )


def maybe_plot(cmd_hdr: PolyCmdHdr, samp_in: npt.NDArray[np.float32], samp_out: npt.NDArray[np.float32]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot")
        return

    plt.plot(samp_in, samp_out, label=f"tx_id={cmd_hdr.tx_id}")
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
    report_vitis_synthesis(EXAMPLE_DIR)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    print("Vitis outputs matched the Python golden model.")


if __name__ == "__main__":
    main()
