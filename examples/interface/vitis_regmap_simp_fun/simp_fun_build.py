from __future__ import annotations

import argparse
import csv
import json
import time as _time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pysilicon.build.build import BuildConfig, BuildDag, BuildStep, SourceStep
from pysilicon.build.cosim_steps import ExtractCosimTimingStep, ValidateTimingStep
from pysilicon.build.hwcodegen_steps import HlsCodegenStep
from pysilicon.build.verify_steps import FunctionalVerifyStep
from pysilicon.toolchain import toolchain

try:
    from examples.interface.vitis_regmap_simp_fun.simp_fun import (
        DEFAULT_VECTOR,
        S32,
        SimpFunCase,
        SimpFunComponent,
        SimpFunTBHls,
        simulate_case,
        write_sim_summary,
    )
    from examples.interface.vitis_regmap_simp_fun.timing_diagram import write_timing_diagram
except ModuleNotFoundError:
    from simp_fun import (  # type: ignore[no-redef]
        DEFAULT_VECTOR,
        S32,
        SimpFunCase,
        SimpFunComponent,
        SimpFunTBHls,
        simulate_case,
        write_sim_summary,
    )
    from timing_diagram import write_timing_diagram  # type: ignore[no-redef]


_SOURCE_DIR = Path(__file__).resolve().parent


@dataclass(kw_only=True)
class BuildInputsStep(BuildStep):
    description = "Write scalar AXI-Lite inputs for the simp_fun example."
    consumes = ["simp_fun_source"]
    produces = {
        "x_in": Path("data/x.bin"),
        "a_in": Path("data/a.bin"),
        "b_in": Path("data/b.bin"),
        "data_dir": Path("data"),
    }
    params = {
        "x": DEFAULT_VECTOR["x"],
        "a": DEFAULT_VECTOR["a"],
        "b": DEFAULT_VECTOR["b"],
    }

    def run(self, config: BuildConfig, x, a, b, **_) -> dict:
        out_dir = config.root_dir / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        x_path = out_dir / "x.bin"
        a_path = out_dir / "a.bin"
        b_path = out_dir / "b.bin"
        S32(int(x)).write_uint32_file(x_path)
        S32(int(a)).write_uint32_file(a_path)
        S32(int(b)).write_uint32_file(b_path)
        return {"x_in": x_path, "a_in": a_path, "b_in": b_path, "data_dir": out_dir}


@dataclass(kw_only=True)
class PySimStep(BuildStep):
    description = "Run the Python regmap simulation and write results/sim/."
    consumes = ["simp_fun_source", "x_in", "a_in", "b_in"]
    produces = {
        "sim_dir": Path("results/sim"),
        "log": Path("results/sim_log.csv"),
        "sim_summary": Path("results/sim_summary.json"),
    }
    params = {"clk_freq": 100e6, "latency_cycles": 4, "log_file": "results/sim_log.csv"}

    def expected_paths(self, config: BuildConfig) -> dict[str, Path]:
        log_file = config.params.get("log_file", self.params["log_file"])
        return {"log": config.root_dir / log_file}

    def run(self, config: BuildConfig, x_in, a_in, b_in, clk_freq, latency_cycles, log_file, **_) -> dict:
        x = int(S32().read_uint32_file(x_in).val)
        a = int(S32().read_uint32_file(a_in).val)
        b = int(S32().read_uint32_file(b_in).val)
        case = SimpFunCase(x=x, a=a, b=b)
        log_path = config.root_dir / log_file
        result = simulate_case(case, clk_freq=clk_freq, latency_cycles=latency_cycles,
                               log_file=log_path)
        sim_dir = config.root_dir / "results" / "sim"
        sim_dir.mkdir(parents=True, exist_ok=True)
        S32(result.y).write_uint32_file(sim_dir / "y.bin")
        # See note in simp_fun.py SimpFunTBHls.main on why ap_done is omitted
        # from the regmap_status.json shape: the C++ TB cannot reach it as a
        # local variable. The two-flow comparison is on `y` only.
        (sim_dir / "regmap_status.json").write_text(
            json.dumps({"y": int(result.y)}, indent=2),
            encoding="utf-8",
        )
        summary_path = write_sim_summary(config.root_dir / "results" / "sim_summary.json", result)
        return {"sim_dir": sim_dir, "log": log_path, "sim_summary": summary_path}


@dataclass(kw_only=True)
class ExtractPyTimingStep(BuildStep):
    description = "Extract structured transaction timing from the Python sim log."
    consumes = ["log"]
    produces = {"py_timing": Path("results/py_timing.json")}
    params = {"clk_freq": 100e6}

    def run(self, config: BuildConfig, log, clk_freq, **_) -> dict:
        events: dict[str, float] = {}
        with open(log, newline="") as f:
            for row in csv.DictReader(f):
                event = row["event"]
                if event not in events:
                    events[event] = float(row["time"])
        t_start = events.get("ap_start_host")
        t_end = events.get("host_done", events.get("kernel_done"))
        if t_start is None or t_end is None:
            raise RuntimeError(f"Missing timing events in log: {list(events)}")
        transaction_seconds = t_end - t_start
        transaction_cycles = int(round(transaction_seconds * clk_freq))
        out_path = config.root_dir / "results" / "py_timing.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "transaction_cycles": transaction_cycles,
            "transaction_seconds": transaction_seconds,
            "clk_freq": float(clk_freq),
            "source": "py_sim",
            "events": {
                "ap_start_host": t_start,
                "host_done": t_end,
            },
        }, indent=2), encoding="utf-8")
        return {"py_timing": out_path}


@dataclass(kw_only=True)
class CSimStep(BuildStep):
    description = "Invoke Vitis HLS C-simulation."
    consumes = ["simp_fun_cpp", "simp_fun_compute_impl", "simp_fun_tb", "run_tcl",
                "data_dir"]
    produces = {"csim_data_dir": "data_dir"}
    params = {"live_output": False, "clk_freq": 100e6}

    def run(self, config: BuildConfig, data_dir, live_output, clk_freq, **_) -> dict:
        vitis_env = {
            "PYSILICON_SIMP_FUN_COSIM": "0",
            "PYSILICON_SIMP_FUN_TRACE_LEVEL": "none",
            "PYSILICON_SIMP_FUN_CLK_PERIOD_NS": f"{1e9 / clk_freq:g}",
        }
        try:
            result = toolchain.run_vitis_hls(
                config.root_dir / "run.tcl",
                work_dir=config.root_dir,
                capture_output=not live_output,
                env=vitis_env,
            )
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
        except Exception as exc:
            raise RuntimeError(str(exc))
        return {"csim_data_dir": data_dir}


@dataclass(kw_only=True)
class CSynthStep(BuildStep):
    description = "Run Vitis HLS C-synthesis and RTL co-simulation."
    consumes = ["simp_fun_cpp", "simp_fun_compute_impl", "simp_fun_tb", "run_tcl",
                "csim_data_dir"]
    produces = {"report_dir": Path("pysilicon_simp_fun_proj/solution1")}
    params = {"live_output": False, "clk_freq": 100e6}

    def run(self, config: BuildConfig, live_output, clk_freq, **_) -> dict:
        vitis_env = {
            "PYSILICON_SIMP_FUN_COSIM": "1",
            "PYSILICON_SIMP_FUN_TRACE_LEVEL": "none",
            "PYSILICON_SIMP_FUN_CLK_PERIOD_NS": f"{1e9 / clk_freq:g}",
        }
        try:
            result = toolchain.run_vitis_hls(
                config.root_dir / "run.tcl",
                work_dir=config.root_dir,
                capture_output=not live_output,
                env=vitis_env,
            )
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
        except Exception as exc:
            raise RuntimeError(str(exc))
        return {"report_dir": config.root_dir / "pysilicon_simp_fun_proj" / "solution1"}


@dataclass(kw_only=True)
class InspectSynthStep(BuildStep):
    description = "Parse the Vitis HLS synthesis report and write results/loop_df.csv."
    consumes = ["report_dir"]
    produces = {"loop_df": Path("results/loop_df.csv")}
    params = {}

    def run(self, config: BuildConfig, report_dir, **_) -> dict:
        from pysilicon.utils.csynthparse import CsynthParser

        if not report_dir.exists():
            raise RuntimeError(f"Solution directory not found: {report_dir}")
        parser = CsynthParser(sol_path=str(report_dir))
        parser.get_loop_pipeline_info()
        parser.get_resources()
        if not parser.loop_df.empty:
            non_unit_ii = parser.loop_df[
                parser.loop_df["PipelineII"].apply(
                    lambda v: isinstance(v, (int, np.integer)) and v > 1
                )
            ]
            if not non_unit_ii.empty:
                raise RuntimeError("Vitis synthesis produced loops with PipelineII > 1.")
        loop_df_path = config.root_dir / "results" / "loop_df.csv"
        loop_df_path.parent.mkdir(parents=True, exist_ok=True)
        parser.loop_df.to_csv(loop_df_path, index=False)
        return {"loop_df": loop_df_path}


@dataclass(kw_only=True)
class GenerateTimingDiagramStep(BuildStep):
    description = "Generate the committed timing-diagram artifacts from timing JSON."
    consumes = ["py_timing", "cosim_timing", "timing_verdict", "timing_diagram_source"]
    produces = {
        "timing_diagram_svg": Path("results/timing_diagram.svg"),
        "timing_diagram_json": Path("results/timing_diagram.json"),
    }
    params = {}

    def run(self, config: BuildConfig, py_timing, cosim_timing, timing_verdict, **_) -> dict:
        svg_path = config.root_dir / "results" / "timing_diagram.svg"
        json_path = config.root_dir / "results" / "timing_diagram.json"
        write_timing_diagram(py_timing, cosim_timing, timing_verdict, svg_path, json_path)
        return {"timing_diagram_svg": svg_path, "timing_diagram_json": json_path}


def build_simp_fun_dag() -> BuildDag:
    dag = BuildDag()
    dag.add(SourceStep(artifact="simp_fun_source", path=_SOURCE_DIR / "simp_fun.py"))
    dag.add(SourceStep(artifact="run_tcl", path=_SOURCE_DIR / "run.tcl"))
    dag.add(SourceStep(artifact="timing_diagram_source", path=_SOURCE_DIR / "timing_diagram.py"))

    dag.add(BuildInputsStep(name="build_inputs"))
    dag.add(PySimStep(name="py_sim"))
    dag.add(ExtractPyTimingStep(name="extract_py_timing"))

    dag.add(HlsCodegenStep(
        name="gen_kernel",
        comp_class=SimpFunComponent,
        source_artifact="simp_fun_source",
        output_dir="gen",
        impl_dir=".",
    ))
    dag.add(HlsCodegenStep(
        name="gen_tb",
        comp_class=SimpFunTBHls,
        source_artifact="simp_fun_source",
        output_dir="gen",
        is_testbench=True,
    ))

    dag.add(CSimStep(name="csim"))
    dag.add(FunctionalVerifyStep(
        name="validate_csim",
        golden_dir_artifact="sim_dir",
        actual_dir_artifact="csim_data_dir",
        schemas=[
            {"filename": "y_data.bin", "golden_filename": "y.bin", "schema": S32},
        ],
        jsons=[
            {"filename": "regmap_status.json", "compare_fields": ["y"]},
        ],
        output_dir="results/vitis",
        output_artifact="vitis_dir",
        report_path="results/verify_csim.json",
    ))

    dag.add(CSynthStep(name="csynth"))
    dag.add(InspectSynthStep(name="inspect_synth"))
    dag.add(ExtractCosimTimingStep(
        name="extract_cosim_timing",
        top="simp_fun",
        report_dir_artifact="report_dir",
    ))
    dag.add(ValidateTimingStep(
        name="validate_timing",
        py_timing_artifact="py_timing",
        cosim_timing_artifact="cosim_timing",
        tolerance_cycles=4,
    ))
    dag.add(GenerateTimingDiagramStep(name="generate_timing_diagram"))
    return dag


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the vitis_regmap_simp_fun example.")
    parser.add_argument("--through", metavar="STEP", default="extract_py_timing",
                        help="Run the DAG up to and including this step.")
    parser.add_argument("--list-steps", action="store_true",
                        help="Print all step names in execution order and exit.")
    parser.add_argument("--list-steps-verbose", action="store_true",
                        help="Print step names with descriptions and artifacts.")
    parser.add_argument("--list-artifacts", action="store_true",
                        help="Print all artifacts with their producer and file path.")
    parser.add_argument("--status", action="store_true",
                        help="Print pre-build freshness status and exit.")
    parser.add_argument("--x", type=int, default=DEFAULT_VECTOR["x"])
    parser.add_argument("--a", type=int, default=DEFAULT_VECTOR["a"])
    parser.add_argument("--b", type=int, default=DEFAULT_VECTOR["b"])
    parser.add_argument("--clk-freq", type=float, default=100e6,
                        metavar="HZ", help="Target clock frequency in Hz.")
    parser.add_argument("--latency-cycles", type=int, default=4)
    parser.add_argument("--log", metavar="FILE", default="results/sim_log.csv",
                        help="Log filename relative to the build root.")
    parser.add_argument("--live-output", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Force all steps to rebuild.")
    parser.add_argument("--force-step", metavar="STEP", action="append", default=[],
                        help="Force a specific step to rebuild (repeatable).")
    args = parser.parse_args()

    dag = build_simp_fun_dag()

    if args.list_steps:
        for name in dag.step_names():
            print(name)
        return

    if args.list_steps_verbose:
        for step in dag.steps():
            consumes = ", ".join(step.consumes) if step.consumes else "(none)"
            produces = ", ".join(step.produces) if step.produces else "(none)"
            print(f"{step.name}")
            if step.description:
                print(f"    {step.description}")
            print(f"    consumes: {consumes}")
            print(f"    produces: {produces}")
        return

    config = BuildConfig(
        root_dir=_SOURCE_DIR,
        params={
            "x": args.x,
            "a": args.a,
            "b": args.b,
            "clk_freq": args.clk_freq,
            "latency_cycles": args.latency_cycles,
            "log_file": args.log,
            "live_output": args.live_output,
        },
    )

    if args.list_artifacts:
        all_paths = dag.artifact_paths(config)
        for artifact, step_name in dag.artifact_owners().items():
            p = all_paths.get(artifact)
            if p is not None:
                try:
                    display = p.relative_to(config.root_dir)
                except ValueError:
                    display = p
                print(f"  {artifact:<24} {step_name:<26} {display}")
            else:
                print(f"  {artifact:<24} {step_name:<26} (object)")
        return

    if args.status:
        now = _time.time()
        for entry in dag.results_status(config):
            age = f"{(now - entry['mtime']) / 3600:.1f}h ago" if entry["mtime"] else "—"
            exists_mark = "✓" if entry["exists"] else "✗"
            stale_note = (f"  STALE ({', '.join(entry['stale_because'])} newer)"
                          if entry["stale"] else "")
            print(f"  {entry['artifact']:<16} {entry['produced_by']:<22} "
                  f"{exists_mark}  {age:<12}{stale_note}")
        return

    force: bool | list[str] = True if args.force else (args.force_step or False)

    def on_step_begin(step, will_run, paths):
        print(f"{step.name}:")
        for artifact, p in paths.items():
            try:
                display = p.relative_to(config.root_dir)
            except ValueError:
                display = p
            print(f"    {display}")
        if will_run:
            print("    RUNNING...")

    def on_step_end(step, result):
        if not result.success:
            print(f"    FAILED: {result.message}")
        elif result.skipped:
            print("    UP-TO-DATE")
        else:
            print("    PASSED")

    dag.run(config, through=args.through, force=force,
            on_step_begin=on_step_begin, on_step_end=on_step_end)


if __name__ == "__main__":
    main()
