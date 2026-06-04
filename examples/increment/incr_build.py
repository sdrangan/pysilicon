"""Build DAG for the increment-buffer toy: Python golden → m_axi kernel codegen
→ Vitis C-sim functional verification (against the Python model).

Mirrors examples/poly/poly_build.py, trimmed to the C-sim milestone (decision
11): the generated kernel is proven with the hand-written incr_tb.cpp before TB
codegen (Phase 5).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pysilicon.build.build import BuildConfig, BuildDag, BuildStep, SourceStep
from pysilicon.build.hwcodegen_steps import HlsCodegenStep
from pysilicon.build.streamutils import MemMgrStep, StreamUtilsStep
from pysilicon.build.verify_steps import FunctionalVerifyStep
from pysilicon.hw.arrayutils import ArrayUtilsStep, write_uint32_file
from pysilicon.hw.dataschema import DataSchemaStep
from pysilicon.toolchain import toolchain

try:
    from examples.increment.incr import (
        IncrAccel, IncrCmd, IncrResp, SCHEMA_CLASSES, Uint32Field,
        WORD_BW_SUPPORTED, build_inputs, run_sim,
    )
except ModuleNotFoundError:  # direct execution
    from incr import (  # type: ignore[no-redef]
        IncrAccel, IncrCmd, IncrResp, SCHEMA_CLASSES, Uint32Field,
        WORD_BW_SUPPORTED, build_inputs, run_sim,
    )


_SOURCE_DIR = Path(__file__).resolve().parent


@dataclass(kw_only=True)
class BuildInputsStep(BuildStep):
    description = "Write in.bin, cmd.bin, and params.json for the testbench."
    consumes    = ["incr_source"]
    produces    = {
        "in_bin":   Path("data/in.bin"),
        "cmd_bin":  Path("data/cmd.bin"),
        "data_dir": Path("data"),
    }
    params      = {"n": 37, "seed": 7}

    def run(self, config: BuildConfig, n, seed, **_) -> dict:
        rng = np.random.default_rng(int(seed))
        input_buf = rng.integers(0, 1000, size=int(n), dtype=np.uint32)
        data_dir = build_inputs(config.root_dir / "data", input_buf)
        return {
            "in_bin":   data_dir / "in.bin",
            "cmd_bin":  data_dir / "cmd.bin",
            "data_dir": data_dir,
        }


@dataclass(kw_only=True)
class PySimStep(BuildStep):
    description = "Run the SimPy increment model; write golden out.bin + resp.bin."
    consumes    = ["incr_source", "in_bin", "cmd_bin"]
    produces    = {"sim_dir": Path("results/sim")}
    params      = {"clk_freq": 1e9}

    def run(self, config: BuildConfig, in_bin, cmd_bin, clk_freq, **_) -> dict:
        cmd = IncrCmd()
        cmd.read_uint32_file(cmd_bin)
        n = int(cmd.n)
        input_buf = np.fromfile(in_bin, dtype="<u4")[:n].astype(np.uint32)

        res = run_sim(input_buf, clk_freq=clk_freq)
        if not res.passed:
            raise RuntimeError(
                f"SimPy increment model failed: status={res.status}, "
                f"expected={res.expected}, got={res.result}"
            )

        sim_dir = config.root_dir / "results" / "sim"
        sim_dir.mkdir(parents=True, exist_ok=True)
        write_uint32_file(res.result, elem_type=Uint32Field,
                          file_path=sim_dir / "out.bin", nwrite=n)
        resp = IncrResp()
        resp.status = res.status
        resp.write_uint32_file(sim_dir / "resp.bin")
        return {"sim_dir": sim_dir}


@dataclass(kw_only=True)
class HlsGenIncludeStep(BuildStep):
    description = "Generate schema + utility headers for the Vitis flow."
    consumes    = ["incr_source"]
    params      = {}
    include_dir: str = "include"

    @property
    def produces(self) -> dict:  # type: ignore[override]
        return {"include_dir": Path(self.include_dir)}

    def run(self, config: BuildConfig, **_) -> dict:
        inner = BuildDag()
        inner.add(StreamUtilsStep(output_dir=self.include_dir))
        inner.add(MemMgrStep(output_dir=self.include_dir))
        for cls in SCHEMA_CLASSES:
            inner.add(DataSchemaStep(cls, word_bw_supported=WORD_BW_SUPPORTED,
                                     include_dir=self.include_dir))
        inner.add(ArrayUtilsStep(Uint32Field, WORD_BW_SUPPORTED))
        results = inner.run(config)
        failed = [n for n, r in results.items() if not r.success]
        if failed:
            raise RuntimeError(f"Code generation failed: {failed}")
        return {"include_dir": config.root_dir / self.include_dir}


@dataclass(kw_only=True)
class CSimStep(BuildStep):
    description = "Invoke Vitis HLS C-simulation of the generated kernel."
    consumes    = ["incr_cpp", "incr_hpp", "incr_transform_impl",
                   "incr_respond_impl", "include_dir", "data_dir"]
    produces    = {"csim_data_dir": "data_dir"}
    params      = {"live_output": False, "clk_freq": 1e9}

    def run(self, config: BuildConfig, include_dir, data_dir, live_output, clk_freq, **_) -> dict:
        env = {"PYSILICON_INCR_COSIM": "0",
               "PYSILICON_INCR_CLK_PERIOD_NS": f"{1e9 / clk_freq:g}"}
        result = toolchain.run_vitis_hls(
            config.root_dir / "run.tcl",
            work_dir=config.root_dir,
            capture_output=not live_output,
            env=env,
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return {"csim_data_dir": data_dir}


def build_incr_dag() -> BuildDag:
    dag = BuildDag()
    dag.add(SourceStep(
        artifact="incr_source", path=_SOURCE_DIR / "incr.py",
        description="Python source for schemas, accelerator, and testbench.",
    ))

    # Python golden model
    dag.add(BuildInputsStep(name="build_inputs"))
    dag.add(PySimStep(name="py_sim"))

    # HLS code generation
    dag.add(HlsGenIncludeStep(name="gen_include"))
    dag.add(HlsCodegenStep(
        name="gen_kernel",
        comp_class=IncrAccel,
        source_artifact="incr_source",
        output_dir="gen",
        impl_dir=".",
    ))

    # C-sim functional verification vs the Python model
    dag.add(CSimStep(name="csim"))
    dag.add(FunctionalVerifyStep(
        name="validate_csim",
        golden_dir_artifact="sim_dir",
        actual_dir_artifact="csim_data_dir",
        extra_artifacts=["cmd_bin"],
        schemas=[
            {"filename": "resp_data.bin", "golden_filename": "resp.bin",
             "schema": IncrResp},
        ],
        arrays=[
            {"filename": "out_data.bin", "golden_filename": "out.bin",
             "elem_type": Uint32Field,
             "count_from_extra": "cmd_bin", "count_schema": IncrCmd,
             "count_field": "n",
             "rtol": 0.0, "atol": 0.0},
        ],
        output_dir="results/vitis",
        output_artifact="vitis_dir",
        report_path="results/verify_csim.json",
    ))
    return dag


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the increment accelerator example.")
    parser.add_argument("--through", metavar="STEP", default="gen_kernel",
                        help="Run the DAG up to and including this step.")
    parser.add_argument("--n", type=int, default=37)
    parser.add_argument("--live-output", action="store_true")
    args = parser.parse_args()

    config = BuildConfig(
        root_dir=_SOURCE_DIR,
        params={"n": args.n, "live_output": args.live_output},
    )
    dag = build_incr_dag()
    results = dag.run(config, through=args.through)
    for name, r in results.items():
        status = "PASS" if r.success else f"FAIL: {r.message}"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
