from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
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
TRACE_LEVELS = ("none", "port", "all")
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


@dataclass(slots=True)
class PolySimResult:
    """Result bundle returned by PolyTest.simulate."""

    cmd_hdr: PolyCmdHdr
    samp_in: npt.NDArray[np.float32]
    resp_hdr: PolyRespHdr
    samp_out: npt.NDArray[np.float32]
    resp_ftr: PolyRespFtr

    @property
    def passed(self) -> bool:
        return self.resp_ftr.error == PolyError.NO_ERROR


class PolyAccel(object):
    """
    Python model for the polynomial accelerator.

    This class implements the core streaming polynomial evaluation logic.
    In a real implementation this would be synthesized into hardware; here we
    use NumPy to evaluate a degree-3 polynomial on a vector of input samples.
    """

    def evaluate(
        self,
        cmd_hdr: PolyCmdHdr,
        samp_in: npt.NDArray[np.float32],
    ) -> tuple[PolyRespHdr, npt.NDArray[np.float32], PolyRespFtr]:
        """
        Evaluate the polynomial described by *cmd_hdr* on every element of *samp_in*.

        Parameters
        ----------
        cmd_hdr : PolyCmdHdr
            Command header containing the transaction ID, polynomial coefficients
            (constant term first), and expected number of samples.
        samp_in : npt.NDArray[np.float32]
            Input samples to evaluate.

        Returns
        -------
        tuple[PolyRespHdr, npt.NDArray[np.float32], PolyRespFtr]
            Response header, output samples, and response footer.
        """
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


class PolyTest(object):
    """Stateful test/demo harness for the polynomial accelerator flow."""

    def __init__(
        self,
        nsamp: int = 100,
        example_dir: Path = EXAMPLE_DIR,
        include_dir: str = INCLUDE_DIR,
    ):
        self.nsamp = max(1, int(nsamp))
        self.example_dir = Path(example_dir)
        self.include_dir = include_dir

        self.poly_accel: PolyAccel | None = None
        self.cmd_hdr: PolyCmdHdr | None = None
        self.samp_in: npt.NDArray[np.float32] | None = None
        self.resp_hdr: PolyRespHdr | None = None
        self.samp_out: npt.NDArray[np.float32] | None = None
        self.resp_ftr: PolyRespFtr | None = None

    def _build_inputs(self) -> None:
        """Construct the command header and input sample vector."""
        coeffs = CoeffArray()
        coeffs.val = np.array([1.0, -2.0, -3.0, 4.0], dtype=np.float32)

        self.cmd_hdr = PolyCmdHdr()
        self.cmd_hdr.tx_id = 42
        self.cmd_hdr.coeffs = coeffs.val
        self.cmd_hdr.nsamp = self.nsamp

        self.samp_in = np.linspace(0.0, 1.0, self.nsamp, dtype=np.float32)

    def simulate(self) -> PolySimResult:
        """
        Run the Python model and store both inputs and outputs.

        Returns
        -------
        PolySimResult
            Result bundle with the command header, input samples, response
            header, output samples, and response footer.
        """
        self._build_inputs()

        self.poly_accel = PolyAccel()
        self.resp_hdr, self.samp_out, self.resp_ftr = self.poly_accel.evaluate(
            self.cmd_hdr, self.samp_in
        )

        return PolySimResult(
            cmd_hdr=self.cmd_hdr,
            samp_in=self.samp_in,
            resp_hdr=self.resp_hdr,
            samp_out=self.samp_out,
            resp_ftr=self.resp_ftr,
        )

    def gen_vitis_code(self) -> list[Path]:
        """Generate schema and utility headers needed for the Vitis flow."""
        cfg = CodeGenConfig(root_dir=self.example_dir, util_dir=self.include_dir)
        generated_paths: list[Path] = []
        for schema_class in SCHEMA_CLASSES:
            generated_paths.append(schema_class.gen_include(cfg=cfg, word_bw_supported=WORD_BW_SUPPORTED))
        generated_paths.append(gen_array_utils(Float32, WORD_BW_SUPPORTED, cfg=cfg))
        copy_streamutils(cfg)
        return generated_paths

    def write_input_files(self, data_dir: Path | None = None) -> Path:
        """Write binary test-vector files for the Vitis testbench.

        The Python model (``simulate()``) must have been run first.

        Parameters
        ----------
        data_dir : Path | None
            Directory in which to write the files.  Defaults to
            ``<example_dir>/data``.

        Returns
        -------
        Path
            The directory containing the written files.
        """
        if self.cmd_hdr is None or self.samp_in is None or self.resp_hdr is None:
            self.simulate()

        if data_dir is None:
            data_dir = self.example_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        self.cmd_hdr.write_uint32_file(data_dir / "cmd_hdr_data.bin")
        write_uint32_file(
            self.samp_in,
            elem_type=Float32,
            file_path=data_dir / "samp_in_data.bin",
            nwrite=self.cmd_hdr.nsamp,
        )

        return data_dir

    def read_vitis_outputs(self, data_dir: Path) -> PolySimResult:
        """Read Vitis testbench output files and compare against the Python model.

        Parameters
        ----------
        data_dir : Path
            Directory containing the output files written by the Vitis testbench.

        Returns
        -------
        PolySimResult
            Result bundle populated from the Vitis output files.

        Raises
        ------
        RuntimeError
            If any output does not match the Python golden model.
        """
        got_resp_hdr = PolyRespHdr().read_uint32_file(data_dir / "resp_hdr_data.bin")
        got_resp_ftr = PolyRespFtr().read_uint32_file(data_dir / "resp_ftr_data.bin")
        got_samp_out = np.asarray(
            read_uint32_file(
                data_dir / "samp_out_data.bin",
                elem_type=Float32,
                shape=int(self.resp_ftr.nsamp_read),
            ),
            dtype=np.float32,
        )

        if not got_resp_hdr.is_close(self.resp_hdr):
            raise RuntimeError("Response header mismatch after Vitis C-simulation.")
        if not got_resp_ftr.is_close(self.resp_ftr):
            raise RuntimeError("Response footer mismatch after Vitis C-simulation.")
        if not np.allclose(got_samp_out, self.samp_out[: got_samp_out.size], rtol=1e-6, atol=1e-6):
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

        return PolySimResult(
            cmd_hdr=self.cmd_hdr,
            samp_in=self.samp_in,
            resp_hdr=got_resp_hdr,
            samp_out=got_samp_out,
            resp_ftr=got_resp_ftr,
        )

    def report_synthesis(self) -> None:
        """Parse and print the Vitis HLS C-synthesis report.

        Raises
        ------
        RuntimeError
            If any synthesized loop has a pipeline initiation interval greater
            than 1.
        """
        try:
            from pysilicon.utils.csynthparse import CsynthParser
        except ModuleNotFoundError as exc:
            print(f"Skipping csynth report parsing: {exc}")
            return

        sol_path = self.example_dir / "pysilicon_poly_proj" / "solution1"
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
                parser.loop_df["PipelineII"].apply(
                    lambda value: isinstance(value, (int, np.integer)) and value > 1
                )
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

    def test_vitis(
        self,
        cosim: bool = False,
        trace_level: str = "none",
        live_output: bool = False,
    ) -> PolySimResult:
        """Run the Vitis kernel and testbench, then compare against the Python model.

        This method orchestrates the full file-based Vitis simulation workflow:

        1. Run the Python model (``simulate()``) if not already done.
        2. Generate Vitis HLS headers via ``gen_vitis_code()``.
        3. Write binary input files for the C++ testbench via ``write_input_files()``.
        4. Invoke Vitis HLS C-simulation (and, optionally, RTL co-simulation).
        5. Read the testbench output files and compare against the Python reference.
        6. Parse and print the C-synthesis report via ``report_synthesis()``.

        Parameters
        ----------
        cosim : bool
            If ``True``, also run C-synthesis and RTL co-simulation after the
            C-simulation step.  Default is ``False``.
        trace_level : str
            RTL co-simulation trace level passed through to ``cosim_design``.
            Supported values are ``none``, ``port``, and ``all``. Ignored when
            ``cosim`` is ``False``.
        live_output : bool
            If ``True``, stream Vitis stdout/stderr directly to the terminal
            while the subprocess runs instead of buffering it for later.

        Returns
        -------
        PolySimResult
            Result bundle with outputs read from the Vitis testbench.

        Raises
        ------
        RuntimeError
            If the Vitis simulation outputs do not match the Python model.
        """
        if self.cmd_hdr is None or self.resp_hdr is None:
            self.simulate()

        if trace_level not in TRACE_LEVELS:
            raise ValueError(
                f"Unsupported trace level '{trace_level}'. Expected one of {TRACE_LEVELS}."
            )

        self.gen_vitis_code()
        data_dir = self.write_input_files()

        vitis_env = {
            "PYSILICON_POLY_COSIM": "1" if cosim else "0",
            "PYSILICON_POLY_TRACE_LEVEL": trace_level,
        }

        result = toolchain.run_vitis_hls(
            self.example_dir / "run.tcl",
            work_dir=self.example_dir,
            capture_output=not live_output,
            env=vitis_env,
        )

        vitis_result = self.read_vitis_outputs(data_dir)
        self.report_synthesis()

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)

        return vitis_result

    def maybe_plot(self) -> None:
        """Plot the polynomial input/output using matplotlib, if available."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not installed; skipping plot")
            return

        plt.plot(self.samp_in, self.samp_out, label=f"tx_id={self.cmd_hdr.tx_id}")
        plt.xlabel("Input sample")
        plt.ylabel("Polynomial output")
        plt.title("Polynomial evaluation")
        plt.grid(True)
        plt.legend()
        plt.show()

    def analyze_timing(self, vcd_path: str | Path) -> "PolyTimingResult":
        """
        Analyze timing from an existing VCD file captured from the poly kernel.

        Delegates to :func:`timing_analysis.analyze_poly_vcd`.  The VCD can
        be generated by :meth:`generate_vcd` or by running ``xsim_vcd``
        separately; it does **not** require rerunning RTL co-simulation.

        Parameters
        ----------
        vcd_path : str | Path
            Path to the VCD file to analyze.

        Returns
        -------
        PolyTimingResult
            Structured result with decoded headers, samples, and timing info.
        """
        from timing_analysis import analyze_poly_vcd
        return analyze_poly_vcd(vcd_path)

    def generate_vcd(
        self,
        output_vcd: str = "dump.vcd",
        soln: str | None = "solution1",
        trace_level: str = "*",
    ) -> Path:
        """
        Generate a VCD file by re-running the Vivado RTL simulation.

        Delegates to :func:`pysilicon.scripts.xsim_vcd.run_xsim_vcd`.
        Requires Vivado/xsim installed on Windows.

        Parameters
        ----------
        output_vcd : str
            Output VCD filename written inside a ``vcd/`` subdirectory.
        soln : str | None
            Solution name inside the component directory.
        trace_level : str
            VCD trace level (``'*'`` for all signals, ``'port'`` for ports only).

        Returns
        -------
        Path
            Absolute path to the written VCD file.
        """
        from pysilicon.scripts.xsim_vcd import run_xsim_vcd
        return run_xsim_vcd(
            top="poly",
            comp="pysilicon_poly_proj",
            out=output_vcd,
            soln=soln,
            trace_level=trace_level,
            workdir=self.example_dir,
        )


def main() -> None:
    """Command-line entry point for the polynomial accelerator example.

    Usage examples::

        # Python simulation only
        python poly_demo.py --skip-vitis

        # Python + Vitis C-simulation
        python poly_demo.py

        # Python + Vitis C-simulation + RTL co-simulation
        python poly_demo.py --cosim

        # Python + Vitis RTL co-simulation with full waveform tracing
        python poly_demo.py --cosim --trace-level all

        # Python + Vitis C-simulation with live Vitis output in the terminal
        python poly_demo.py --live-output
    """
    parser = argparse.ArgumentParser(description="Run the polynomial accelerator example.")
    parser.add_argument("--nsamp", type=int, default=100, help="Number of input samples to generate.")
    parser.add_argument("--skip-vitis", action="store_true", help="Only run the Python side of the example.")
    parser.add_argument("--cosim", action="store_true", help="Also run RTL co-simulation after C-synthesis.")
    parser.add_argument(
        "--trace-level",
        choices=TRACE_LEVELS,
        default="none",
        help="RTL co-simulation trace level passed to Vitis. Only used with --cosim.",
    )
    parser.add_argument(
        "--live-output",
        action="store_true",
        help="Stream Vitis stdout/stderr directly to the terminal while it runs.",
    )
    parser.add_argument("--plot", action="store_true", help="Plot the Python golden-model output.")
    args = parser.parse_args()

    test = PolyTest(nsamp=args.nsamp)
    test.gen_vitis_code()

    result = test.simulate()
    print(
        f"Python simulation: tx_id={result.resp_hdr.tx_id}, "
        f"nsamp={result.resp_ftr.nsamp_read}, "
        f"error={result.resp_ftr.error.name}, "
        f"passed={result.passed}"
    )

    if args.plot:
        test.maybe_plot()

    if args.skip_vitis:
        return

    try:
        vitis_result = test.test_vitis(
            cosim=args.cosim,
            trace_level=args.trace_level,
            live_output=args.live_output,
        )
    except RuntimeError as exc:
        print(f"Vitis run failed: {exc}")
        return

    print(f"Vitis simulation matched Python model. nsamp={vitis_result.resp_ftr.nsamp_read}")


if __name__ == "__main__":
    main()
