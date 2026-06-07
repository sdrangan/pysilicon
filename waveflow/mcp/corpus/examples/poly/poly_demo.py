from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np
import numpy.typing as npt

from waveflow.build.build import BuildConfig, BuildDag, BuildResult, BuildStep, FileArtifact, ObjectArtifact
from waveflow.build.streamutils import StreamUtilsStep
from waveflow.hw.arrayutils import ArrayUtilsStep, array, read_array, read_uint32_file, write_array, write_uint32_file
from waveflow.hw.clock import Clock
from waveflow.hw.dataschema import DataArray, DataList, DataSchemaStep, EnumField, FloatField, IntField
from waveflow.hw.hw_component import HwComponent, HwParam
from waveflow.hw.interface import StreamIF, StreamIFMaster, StreamIFSlave
from waveflow.hw.synth import sim_only, synthesizable
from waveflow.simulation.logger import Logger, NullLogger
from waveflow.simulation.simulation import Simulation
from waveflow.simulation.simobj import ProcessGen, SimObj
from waveflow.toolchain import toolchain


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
PolyErrorField = EnumField.specialize(enum_type=PolyError)



class CoeffArray(DataArray):
    """
    Array of polynomial coefficients, stored in ascending order (constant term first).
    For example, for a cubic polynomial c0 + c1*x + c2*x^2 + c3*x^3, the array would be [c0, c1, c2, c3].
    """
    ncoeff: int = 4
    element_type = Float32
    static = True
    max_shape = (ncoeff,)


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


# ---------------------------------------------------------------------------
# SimPy accelerator model (synthesizable HwComponent)
# ---------------------------------------------------------------------------

@dataclass
class PolyAccelComponent(HwComponent):
    """SimPy model of the polynomial accelerator kernel."""

    in_bw:        HwParam[int] = 32
    out_bw:       HwParam[int] = 32
    clk:          Clock = field(default_factory=lambda: Clock(freq=1e9))
    proc_ii:      int = 1
    proc_latency: int = 10

    logger:       Logger | NullLogger = field(default_factory=NullLogger)
    """Logger for debugging timing"""

    unroll_factor: int = 1
    """Unroll factor for the polynomial evaluation loop. Must be a positive integer that divides the number of coefficients (4 in this example)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.s_in  = StreamIFSlave( name=f'{self.name}_s_in',  sim=self.sim, bitwidth=self.in_bw)
        self.m_out = StreamIFMaster(name=f'{self.name}_m_out', sim=self.sim, bitwidth=self.out_bw)
        self.add_endpoint(self.s_in)
        self.add_endpoint(self.m_out)
        self._job: int = 0

    @sim_only
    def _inc_job(self) -> None:
        self._job += 1

    def run_proc(self) -> ProcessGen[None]:
        while True:
            self.logger.log(event='proc_begin', job=self._job)
            cmd_hdr: PolyCmdHdr = yield from self.s_in.get(PolyCmdHdr)
            yield from self.evaluate(cmd_hdr, self.s_in, self.m_out)
            self._inc_job()

    @synthesizable
    def evaluate(
        self,
        cmd_hdr: PolyCmdHdr,
        s_in: StreamIFSlave,
        m_out: StreamIFMaster,
    ) -> ProcessGen[None]:

        # Write the response header with the echo of the transaction ID
        resp_hdr = PolyRespHdr()
        resp_hdr.tx_id = cmd_hdr.tx_id
        self.logger.log(event='resp_hdr_write_begin', job=self._job)
        yield from m_out.write(resp_hdr)


        # Start the sample reading.  Since the processing is pipelined,
        # we can log the start of the sample read
        self.logger.log(event='samp_read_begin', job=self._job)
        samp_in, tstart = yield from s_in.get_pipelined(
            Float32, count=cmd_hdr.nsamp)

        # Perform the polynomial evaluation.
        y = np.zeros_like(samp_in, dtype=np.float32)
        power = np.ones_like(samp_in, dtype=np.float32)
        for coeff in cmd_hdr.coeffs:
            y += coeff * power
            power *= samp_in

        t_out_start = tstart + self.proc_latency * self.clk.period

        # Per unit processing time
        proc_time = cmd_hdr.nsamp / self.unroll_factor * self.proc_ii * self.clk.period
        proc_time = max(0.0, proc_time + (t_out_start - self.env.now))
        yield self.timeout(proc_time)

        yield from m_out.write_pipelined(
            array(Float32, y), t_out_start
        )
        self.logger.log(event='samp_out_write_end', job=self._job)

        resp_ftr = PolyRespFtr()
        resp_ftr.nsamp_read = len(samp_in)
        resp_ftr.error = (PolyError.NO_ERROR if len(samp_in) == cmd_hdr.nsamp
                          else PolyError.WRONG_NSAMP)
        yield from m_out.write(resp_ftr)
        self.logger.log(event='proc_end', job=self._job)


# ---------------------------------------------------------------------------
# SimPy testbench
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class PolyTB(SimObj):
    """Drives one polynomial transaction and captures the response."""

    cmd_hdr: PolyCmdHdr
    samp_in: npt.NDArray[np.float32]
    word_bw: int = 32

    def __post_init__(self) -> None:
        super().__post_init__()
        self.m_in  = StreamIFMaster(name=f'{self.name}_m_in',  sim=self.sim, bitwidth=self.word_bw)
        self.s_out = StreamIFSlave( name=f'{self.name}_s_out', sim=self.sim, bitwidth=self.word_bw)
        self.resp_hdr: PolyRespHdr | None = None
        self.samp_out: npt.NDArray[np.float32] | None = None
        self.resp_ftr: PolyRespFtr | None = None

    def run_proc(self) -> ProcessGen[None]:
        bw = self.word_bw
        yield from self.m_in.write(self.cmd_hdr.serialize(word_bw=bw))
        yield from self.m_in.write(write_array(self.samp_in, elem_type=Float32, word_bw=bw))

        resp_words = yield from self.s_out.get()
        samp_words = yield from self.s_out.get()
        ftr_words  = yield from self.s_out.get()

        self.resp_hdr = PolyRespHdr().deserialize(resp_words, word_bw=bw)
        self.samp_out = read_array(samp_words, elem_type=Float32, word_bw=bw, shape=int(self.cmd_hdr.nsamp))
        self.resp_ftr = PolyRespFtr().deserialize(ftr_words, word_bw=bw)


# ---------------------------------------------------------------------------
# Stream wiring
# ---------------------------------------------------------------------------

def connect(sim: Simulation, tb: PolyTB, accel: PolyAccelComponent, clk: Clock) -> None:
    """Wire in_stream and out_stream between the testbench and the accelerator."""
    in_stream  = StreamIF(sim=sim, clk=clk)
    out_stream = StreamIF(sim=sim, clk=clk)

    in_stream.bind( "master", tb.m_in)
    in_stream.bind( "slave",  accel.s_in)
    out_stream.bind("master", accel.m_out)
    out_stream.bind("slave",  tb.s_out)


# ---------------------------------------------------------------------------
# BuildDag steps for the poly simulation pipeline
# ---------------------------------------------------------------------------

class BuildInputsStep(BuildStep):
    """Create the command header and input sample vector."""

    def __init__(self, nsamp: int = 100) -> None:
        super().__init__()
        self._nsamp = nsamp

    def run(self, config: BuildConfig, results: dict = {}) -> BuildResult:
        coeffs = CoeffArray()
        coeffs.val = np.array([1.0, -2.0, -3.0, 4.0], dtype=np.float32)

        cmd_hdr = PolyCmdHdr()
        cmd_hdr.tx_id = 42
        cmd_hdr.coeffs = coeffs.val
        cmd_hdr.nsamp = self._nsamp

        samp_in = np.linspace(0.0, 1.0, self._nsamp, dtype=np.float32)

        return BuildResult(
            success=True,
            artifacts={
                "cmd_hdr": ObjectArtifact(value=cmd_hdr),
                "samp_in": ObjectArtifact(value=samp_in),
            },
        )


class GenCppStep(BuildStep):
    """Generate schema and utility headers needed for the Vitis flow."""

    def __init__(self, example_dir: Path, include_dir: str = "include") -> None:
        super().__init__()
        self._example_dir = Path(example_dir)
        self._include_dir = include_dir

    def run(self, config: BuildConfig, results: dict = {}) -> BuildResult:
        cfg = BuildConfig(root_dir=self._example_dir)
        dag = BuildDag()
        dag.add(StreamUtilsStep(output_dir=self._include_dir))
        for cls in SCHEMA_CLASSES:
            dag.add(DataSchemaStep(cls, word_bw_supported=WORD_BW_SUPPORTED, include_dir=self._include_dir))
        dag.add(ArrayUtilsStep(Float32, WORD_BW_SUPPORTED))
        inner_results = dag.run(cfg)

        failed = [name for name, r in inner_results.items() if not r.success]
        if failed:
            return BuildResult(success=False, message=f"Code generation failed: {failed}")

        include_path = self._example_dir / self._include_dir
        return BuildResult(
            success=True,
            artifacts={"include_dir": FileArtifact(path=include_path)},
        )


class PySimStep(BuildStep):
    """Run the Python SimPy simulation and capture the result."""

    def __init__(
        self,
        log_file: Path | str | None = None,
        in_bw: int = 32,
        out_bw: int = 32,
        unroll_factor: int = 1,
    ) -> None:
        super().__init__()
        self._log_file = log_file
        self._in_bw = in_bw
        self._out_bw = out_bw
        self._unroll_factor = unroll_factor

    def resolve_deps(self, other_steps: list) -> None:
        self._deps = [s for s in other_steps if isinstance(s, BuildInputsStep)]

    def run(self, config: BuildConfig, results: dict = {}) -> BuildResult:
        inputs_result = results.get("BuildInputsStep")
        if inputs_result is None:
            return BuildResult(success=False, message="BuildInputsStep result not found")

        cmd_hdr = inputs_result.object("cmd_hdr")
        samp_in = inputs_result.object("samp_in")

        sim = Simulation()
        clk = Clock(freq=1e9)
        log_file = self._log_file
        logger_kwargs: dict = {}
        if log_file is not None:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            logger_kwargs["logger"] = Logger(
                name="poly_log", sim=sim,
                file_path=log_file, fields=["event", "job"],
            )
        accel = PolyAccelComponent(
            name="poly_accel", sim=sim,
            in_bw=self._in_bw, out_bw=self._out_bw,
            unroll_factor=self._unroll_factor,
            **logger_kwargs,
        )
        tb = PolyTB(
            name="poly_tb", sim=sim,
            cmd_hdr=cmd_hdr, samp_in=samp_in,
            word_bw=self._in_bw,
        )
        connect(sim, tb, accel, clk)
        sim.run_sim()

        sim_result = PolySimResult(
            cmd_hdr=cmd_hdr,
            samp_in=samp_in,
            resp_hdr=tb.resp_hdr,
            samp_out=tb.samp_out,
            resp_ftr=tb.resp_ftr,
        )
        artifacts: dict = {"sim_result": ObjectArtifact(value=sim_result)}
        if log_file is not None:
            artifacts["log"] = FileArtifact(path=Path(log_file))
        return BuildResult(success=True, artifacts=artifacts)


class ValidateTimingStep(BuildStep):
    """Read the simulation log and verify timing expectations."""

    def __init__(self, proc_latency: int = 10, period: float = 1e-9) -> None:
        super().__init__()
        self._proc_latency = proc_latency
        self._period = period

    def resolve_deps(self, other_steps: list) -> None:
        self._deps = [s for s in other_steps if isinstance(s, PySimStep)]

    def run(self, config: BuildConfig, results: dict = {}) -> BuildResult:
        import csv

        py_sim_result = results.get("PySimStep")
        if py_sim_result is None:
            return BuildResult(success=False, message="PySimStep result not found")
        log_artifact = py_sim_result.artifacts.get("log")
        if log_artifact is None:
            return BuildResult(success=False, message="No log artifact from PySimStep")

        events: dict[str, float] = {}
        with open(log_artifact.path, newline="") as f:
            for row in csv.DictReader(f):
                ev = row["event"]
                if ev not in events:
                    events[ev] = float(row["time"])

        t_start = events.get("samp_read_begin")
        t_end = events.get("samp_out_write_end")
        if t_start is None or t_end is None:
            return BuildResult(
                success=False,
                message=f"Missing timing events in log: {list(events)}",
            )

        durations = {"samp_read_to_write_end": t_end - t_start}
        return BuildResult(
            success=True,
            artifacts={"durations": ObjectArtifact(value=durations)},
        )


class WriteInputsStep(BuildStep):
    """Write binary test-vector files for the Vitis testbench."""

    def __init__(self, example_dir: Path, data_dir: Path | None = None) -> None:
        super().__init__()
        self._example_dir = Path(example_dir)
        self._data_dir = data_dir

    def resolve_deps(self, other_steps: list) -> None:
        self._deps = [s for s in other_steps if isinstance(s, (BuildInputsStep, PySimStep))]

    def run(self, config: BuildConfig, results: dict = {}) -> BuildResult:
        inputs_result = results.get("BuildInputsStep")
        if inputs_result is None:
            return BuildResult(success=False, message="BuildInputsStep result not found")

        cmd_hdr = inputs_result.object("cmd_hdr")
        samp_in = inputs_result.object("samp_in")

        data_dir = Path(self._data_dir) if self._data_dir else self._example_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        cmd_hdr.write_uint32_file(data_dir / "cmd_hdr_data.bin")
        write_uint32_file(
            samp_in,
            elem_type=Float32,
            file_path=data_dir / "samp_in_data.bin",
            nwrite=cmd_hdr.nsamp,
        )

        return BuildResult(
            success=True,
            artifacts={"data_dir": FileArtifact(path=data_dir)},
        )


class CSimStep(BuildStep):
    """Invoke Vitis HLS C-simulation."""

    def __init__(self, example_dir: Path, live_output: bool = False) -> None:
        super().__init__()
        self._example_dir = Path(example_dir)
        self._live_output = live_output

    def resolve_deps(self, other_steps: list) -> None:
        self._deps = [s for s in other_steps if isinstance(s, (GenCppStep, WriteInputsStep))]

    def run(self, config: BuildConfig, results: dict = {}) -> BuildResult:
        write_result = results.get("WriteInputsStep")
        if write_result is None:
            return BuildResult(success=False, message="WriteInputsStep result not found")
        data_dir = write_result.path("data_dir")

        vitis_env = {
            "WAVEFLOW_POLY_COSIM": "0",
            "WAVEFLOW_POLY_TRACE_LEVEL": "none",
        }

        try:
            result = toolchain.run_vitis_hls(
                self._example_dir / "run.tcl",
                work_dir=self._example_dir,
                capture_output=not self._live_output,
                env=vitis_env,
            )
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
        except Exception as exc:
            return BuildResult(success=False, message=str(exc))

        return BuildResult(
            success=True,
            artifacts={"data_dir": FileArtifact(path=data_dir)},
        )


class ValidateCSimStep(BuildStep):
    """Read Vitis testbench output files and compare against the Python model."""

    def resolve_deps(self, other_steps: list) -> None:
        self._deps = [s for s in other_steps if isinstance(s, (CSimStep, PySimStep))]

    def run(self, config: BuildConfig, results: dict = {}) -> BuildResult:
        py_sim_result = results.get("PySimStep")
        csim_result = results.get("CSimStep")
        if py_sim_result is None or csim_result is None:
            return BuildResult(success=False, message="Missing dependency results")

        sim_result: PolySimResult = py_sim_result.object("sim_result")
        data_dir = csim_result.path("data_dir")

        try:
            got_resp_hdr = PolyRespHdr().read_uint32_file(data_dir / "resp_hdr_data.bin")
            got_resp_ftr = PolyRespFtr().read_uint32_file(data_dir / "resp_ftr_data.bin")
            got_samp_out = np.asarray(
                read_uint32_file(
                    data_dir / "samp_out_data.bin",
                    elem_type=Float32,
                    shape=int(sim_result.resp_ftr.nsamp_read),
                ),
                dtype=np.float32,
            )
        except Exception as exc:
            return BuildResult(success=False, message=f"Failed to read Vitis outputs: {exc}")

        if not got_resp_hdr.is_close(sim_result.resp_hdr):
            return BuildResult(success=False, message="Response header mismatch after Vitis C-simulation.")
        if not got_resp_ftr.is_close(sim_result.resp_ftr):
            return BuildResult(success=False, message="Response footer mismatch after Vitis C-simulation.")
        if not np.allclose(got_samp_out, sim_result.samp_out[:got_samp_out.size], rtol=1e-6, atol=1e-6):
            return BuildResult(success=False, message="Sample output mismatch after Vitis C-simulation.")

        sync_status_path = data_dir / "sync_status.json"
        if sync_status_path.exists():
            sync_status = json.loads(sync_status_path.read_text(encoding="utf-8"))
            expected_sync = {
                "resp_hdr_tlast": "tlast_at_end",
                "samp_out_tlast": "tlast_at_end",
                "resp_ftr_tlast": "tlast_at_end",
            }
            if sync_status != expected_sync:
                return BuildResult(
                    success=False,
                    message=f"TLAST sync mismatch. Expected {expected_sync}, got {sync_status}.",
                )

        vitis_result = PolySimResult(
            cmd_hdr=sim_result.cmd_hdr,
            samp_in=sim_result.samp_in,
            resp_hdr=got_resp_hdr,
            samp_out=got_samp_out,
            resp_ftr=got_resp_ftr,
        )
        return BuildResult(
            success=True,
            artifacts={"vitis_result": ObjectArtifact(value=vitis_result)},
        )


class CSynthStep(BuildStep):
    """Run Vitis HLS C-synthesis (and optionally RTL co-simulation)."""

    def __init__(self, example_dir: Path, live_output: bool = False) -> None:
        super().__init__()
        self._example_dir = Path(example_dir)
        self._live_output = live_output

    def resolve_deps(self, other_steps: list) -> None:
        # Depend on CSimStep (in addition to GenCppStep) so that synthesis always
        # runs after C-simulation.  Both steps call `open_project -reset` in
        # run.tcl, so they must be serialised; the one that runs last wins.
        self._deps = [s for s in other_steps if isinstance(s, (GenCppStep, CSimStep))]

    def run(self, config: BuildConfig, results: dict = {}) -> BuildResult:
        vitis_env = {
            "WAVEFLOW_POLY_COSIM": "1",
            "WAVEFLOW_POLY_TRACE_LEVEL": "none",
        }

        try:
            result = toolchain.run_vitis_hls(
                self._example_dir / "run.tcl",
                work_dir=self._example_dir,
                capture_output=not self._live_output,
                env=vitis_env,
            )
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
        except Exception as exc:
            return BuildResult(success=False, message=str(exc))

        report_dir = self._example_dir / "waveflow_poly_proj" / "solution1"
        return BuildResult(
            success=True,
            artifacts={"report_dir": FileArtifact(path=report_dir)},
        )


class InspectSynthStep(BuildStep):
    """Parse the Vitis HLS C-synthesis report and print resource/timing tables."""

    def resolve_deps(self, other_steps: list) -> None:
        self._deps = [s for s in other_steps if isinstance(s, CSynthStep)]

    def run(self, config: BuildConfig, results: dict = {}) -> BuildResult:
        try:
            from waveflow.utils.csynthparse import CsynthParser
        except ModuleNotFoundError as exc:
            return BuildResult(success=False, message=f"csynthparse not available: {exc}")

        csyn_result = results.get("CSynthStep")
        if csyn_result is None:
            return BuildResult(success=False, message="CSynthStep result not found")

        sol_path = csyn_result.path("report_dir")
        if not sol_path.exists():
            return BuildResult(success=False, message=f"Solution directory not found: {sol_path}")

        try:
            parser = CsynthParser(sol_path=str(sol_path))
            parser.get_loop_pipeline_info()
            parser.get_resources()
        except (FileNotFoundError, ValueError) as exc:
            return BuildResult(success=False, message=f"Synthesis report parsing failed: {exc}")

        print("\nLatency and Initiation Interval:")
        if parser.loop_df.empty:
            print("No loop pipeline information found in csynth.xml.")
        else:
            print(parser.loop_df.to_string())
            non_unit_ii = parser.loop_df[
                parser.loop_df["PipelineII"].apply(
                    lambda v: isinstance(v, (int, np.integer)) and v > 1
                )
            ]
            if not non_unit_ii.empty:
                print("Loops with PipelineII > 1:")
                print(non_unit_ii.to_string())
                return BuildResult(
                    success=False,
                    message="Vitis synthesis produced loops with PipelineII > 1.",
                    artifacts={"loop_df": ObjectArtifact(value=parser.loop_df)},
                )
            print("All reported loops have PipelineII <= 1.")

        print("\nResource Usage:")
        if parser.res_df.empty:
            print("No resource information found in csynth.xml.")
        else:
            print(parser.res_df.to_string())

        return BuildResult(
            success=True,
            artifacts={"loop_df": ObjectArtifact(value=parser.loop_df)},
        )


# ---------------------------------------------------------------------------
# Canonical DAG builder
# ---------------------------------------------------------------------------

def build_poly_dag(
    nsamp: int = 100,
    in_bw: int = 32,
    out_bw: int = 32,
    unroll_factor: int = 1,
    log_file: Path | str | None = None,
    example_dir: Path = EXAMPLE_DIR,
    live_output: bool = False,
) -> BuildDag:
    dag = BuildDag()
    dag.add(BuildInputsStep(nsamp=nsamp))
    dag.add(GenCppStep(example_dir=example_dir))
    dag.add(PySimStep(log_file=log_file, in_bw=in_bw, out_bw=out_bw, unroll_factor=unroll_factor))
    dag.add(ValidateTimingStep())
    dag.add(WriteInputsStep(example_dir=example_dir))
    dag.add(CSimStep(example_dir=example_dir, live_output=live_output))
    dag.add(ValidateCSimStep())
    dag.add(CSynthStep(example_dir=example_dir, live_output=live_output))
    dag.add(InspectSynthStep())
    return dag


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the polynomial accelerator example.")
    parser.add_argument(
        "--through", metavar="STEP", default="ValidateTimingStep",
        help="Run the DAG up to and including this step. Use --list-steps to see options.",
    )
    parser.add_argument(
        "--list-steps", action="store_true",
        help="Print all available step names in execution order and exit.",
    )
    parser.add_argument("--nsamp", type=int, default=100)
    parser.add_argument("--in-bw", type=int, default=32, choices=[32, 64])
    parser.add_argument("--out-bw", type=int, default=32, choices=[32, 64])
    parser.add_argument("--unroll-factor", type=int, default=1)
    parser.add_argument("--log", metavar="FILE", default="logs/poly_log.csv")
    parser.add_argument("--live-output", action="store_true")
    args = parser.parse_args()

    log_file = Path(args.log)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    dag = build_poly_dag(
        nsamp=args.nsamp,
        in_bw=args.in_bw,
        out_bw=args.out_bw,
        unroll_factor=args.unroll_factor,
        log_file=log_file,
        live_output=args.live_output,
    )

    if args.list_steps:
        for name in dag.step_names():
            print(name)
        return

    config = BuildConfig(root_dir=EXAMPLE_DIR)
    results = dag.run(config, through=args.through)

    for name, result in results.items():
        status = "OK" if result.success else f"FAILED: {result.message}"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
