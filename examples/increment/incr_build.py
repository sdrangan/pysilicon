"""Build DAG for the increment-buffer toy: Python golden → m_axi kernel codegen
→ Vitis C-sim functional verification (against the Python model).

Mirrors examples/stream_inband/poly_build.py, trimmed to the C-sim milestone (decision
11): the generated kernel is proven with the hand-written incr_tb.cpp before TB
codegen (Phase 5).
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pysilicon.build.build import BuildConfig, BuildDag, BuildStep, SourceStep
from pysilicon.build.hwcodegen_steps import HlsCodegenStep
from pysilicon.build.streamutils import MemMgrStep, StreamUtilsStep
from pysilicon.build.verify_steps import FunctionalVerifyStep
from pysilicon.hw.arrayutils import ArrayUtilsStep, get_nwords, write_uint32_file
from pysilicon.hw.dataschema import DataSchemaStep
from pysilicon.toolchain import toolchain


# Memory geometry of the generated kernel (must match IncrAccel.mem_bw / the
# AXI-MM addressing convention).  Byte-addressed, 32-bit words.
MEM_DWIDTH = 32
DEFAULT_MAX_BURST_LENGTH = 16


# ---------------------------------------------------------------------------
# AXI-MM burst expectation (mirrors examples/histogram/hist_demo.py)
# ---------------------------------------------------------------------------

def _detect_axi_max_burst_lengths(example_dir: Path) -> tuple[int, int]:
    """Read MAX_{READ,WRITE}_BURST_LENGTH from the synthesized RTL, else default.

    Same approach as ``HistTest._detect_axi_max_burst_lengths`` — the values are
    set by Vitis on the m_axi adapter and appear in the generated Verilog/VHDL.
    """
    candidates = [
        example_dir / "pysilicon_incr_proj" / "solution1" / "impl" / "verilog" / "incr.v",
        example_dir / "pysilicon_incr_proj" / "solution1" / "impl" / "vhdl" / "incr.vhd",
    ]
    patterns = {
        "read": [
            re.compile(r"MAX_READ_BURST_LENGTH\s*=>\s*(\d+)"),
            re.compile(r"\.MAX_READ_BURST_LENGTH\s*\(\s*(\d+)\s*\)"),
        ],
        "write": [
            re.compile(r"MAX_WRITE_BURST_LENGTH\s*=>\s*(\d+)"),
            re.compile(r"\.MAX_WRITE_BURST_LENGTH\s*\(\s*(\d+)\s*\)"),
        ],
    }
    read_length = DEFAULT_MAX_BURST_LENGTH
    write_length = DEFAULT_MAX_BURST_LENGTH
    for cand in candidates:
        if not cand.exists():
            continue
        text = cand.read_text(encoding="utf-8", errors="ignore")
        for pat in patterns["read"]:
            m = pat.search(text)
            if m is not None:
                read_length = int(m.group(1))
                break
        for pat in patterns["write"]:
            m = pat.search(text)
            if m is not None:
                write_length = int(m.group(1))
                break
        if read_length and write_length:
            break
    return read_length, write_length


def _split_region_into_bursts(
    region: dict, max_burst_length: int, bytes_per_word: int,
) -> list[dict]:
    """Split a contiguous ``{name, addr, nwords}`` region into ≤max-length bursts."""
    addr = int(region["addr"])
    nwords = int(region["nwords"])
    name = str(region["name"])
    bursts: list[dict] = []
    remaining = nwords
    offset_words = 0
    while remaining > 0:
        burst_words = min(remaining, max_burst_length)
        bursts.append({
            "name": name,
            "addr": addr + offset_words * bytes_per_word,
            "nwords": burst_words,
        })
        offset_words += burst_words
        remaining -= burst_words
    return bursts


def _burst_layout(bursts: list[dict], len_key: str) -> list[dict]:
    """Reduce extracted bursts to ``[{addr, nwords}]`` for layout comparison."""
    layout = []
    for burst in bursts:
        addr = burst.get("addr")
        if addr is None:
            continue
        raw_len = burst.get(len_key)
        nwords = len(burst.get("data", []))
        if raw_len is not None:
            nwords = int(raw_len) + 1   # AxLEN is beats-minus-one
        layout.append({"addr": int(addr), "nwords": int(nwords)})
    return layout


def _expected_burst_summary(n: int, addr: int, example_dir: Path) -> dict:
    """Expected AXI-MM traffic for the increment toy: one read region and one
    write region, each ``n`` words at ``addr``, split by max burst length.

    The kernel reads the buffer (``read_array``) and writes it back in place
    (``write_array``) at the same address — so read and write regions coincide.
    """
    max_read, max_write = _detect_axi_max_burst_lengths(example_dir)
    bytes_per_word = MEM_DWIDTH // 8
    nwords = int(get_nwords(Uint32Field, word_bw=MEM_DWIDTH, shape=n))
    read_region = {"name": "buf", "addr": addr, "nwords": nwords}
    write_region = {"name": "buf", "addr": addr, "nwords": nwords}
    read_bursts = _split_region_into_bursts(read_region, max_read, bytes_per_word)
    write_bursts = _split_region_into_bursts(write_region, max_write, bytes_per_word)
    return {
        "max_read_burst_length": max_read,
        "max_write_burst_length": max_write,
        "read_burst_count": len(read_bursts),
        "write_burst_count": len(write_bursts),
        "read_bursts": read_bursts,
        "write_bursts": write_bursts,
    }

try:
    from examples.increment.incr import (
        IncrAccel, IncrCmd, IncrResp, IncrTBHls, SCHEMA_CLASSES, Uint32Field,
        WORD_BW_SUPPORTED, build_inputs, run_sim,
    )
except ModuleNotFoundError:  # direct execution
    from incr import (  # type: ignore[no-redef]
        IncrAccel, IncrCmd, IncrResp, IncrTBHls, SCHEMA_CLASSES, Uint32Field,
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
                   "incr_respond_impl", "incr_tb", "include_dir", "data_dir"]
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


@dataclass(kw_only=True)
class CosimStep(BuildStep):
    description = "Run Vitis HLS C-synthesis + RTL co-simulation (with trace)."
    consumes    = ["incr_cpp", "incr_hpp", "incr_transform_impl",
                   "incr_respond_impl", "incr_tb", "include_dir", "csim_data_dir"]
    produces    = {"report_dir": Path("pysilicon_incr_proj/solution1")}
    params      = {"live_output": False, "clk_freq": 1e9, "trace_level": "port"}

    def run(self, config: BuildConfig, include_dir, csim_data_dir,
            live_output, clk_freq, trace_level, **_) -> dict:
        env = {"PYSILICON_INCR_COSIM": "1",
               "PYSILICON_INCR_TRACE_LEVEL": trace_level,
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
        return {"report_dir": config.root_dir / "pysilicon_incr_proj" / "solution1"}


@dataclass(kw_only=True)
class GenerateVcdStep(BuildStep):
    description = "Re-run the traced RTL co-sim to emit a VCD of the AXI-MM bus."
    consumes    = ["report_dir"]
    produces    = {"vcd_path": Path("vcd/dump.vcd")}
    params      = {"trace_level": "port"}

    def run(self, config: BuildConfig, report_dir, trace_level, **_) -> dict:
        from pysilicon.scripts.xsim_vcd import run_xsim_vcd
        tl = trace_level if trace_level in ("all", "port") else "*"
        vcd = run_xsim_vcd(
            top="incr",
            comp="pysilicon_incr_proj",
            out="dump.vcd",
            soln="solution1",
            trace_level=tl,
            workdir=config.root_dir,
        )
        return {"vcd_path": vcd}


@dataclass(kw_only=True)
class ExtractBurstsStep(BuildStep):
    description = "Extract AXI-MM bursts from the VCD and validate counts + layout."
    consumes    = ["vcd_path", "cmd_bin"]
    produces    = {"burst_report": Path("results/burst_info.json")}
    params      = {}

    def run(self, config: BuildConfig, vcd_path, cmd_bin, **_) -> dict:
        from vcdvcd import VCDVCD

        from pysilicon.utils.vcd import VcdParser

        cmd = IncrCmd()
        cmd.read_uint32_file(cmd_bin)
        n = int(cmd.n)
        # The testbench performs a single first-fit allocation, so the buffer
        # lands at word index 0 -> byte address 0 (read == write region).
        addr = 0
        expected = _expected_burst_summary(n, addr, config.root_dir)

        vcd_path = Path(vcd_path)
        if not vcd_path.exists():
            raise FileNotFoundError(f"VCD file not found: {vcd_path}")
        vcd = VCDVCD(str(vcd_path), signals=None, store_tvs=True)
        vp = VcdParser(vcd)
        clk_sig = vp.add_clock_signal()
        aximm_sigs, aximm_bw = vp.add_aximm_signals(
            prefix="m_axi_gmem_", dir="both", lite_only=False,
            short_name_prefix="gmem_",
        )
        write_bursts, read_bursts, clk_period = vp.extract_aximm_bursts(
            clk_name=clk_sig, aximm_sigs=aximm_sigs,
        )

        actual = {
            "read_burst_count": len(read_bursts),
            "write_burst_count": len(write_bursts),
            "read_burst_layout": _burst_layout(read_bursts, "arlen"),
            "write_burst_layout": _burst_layout(write_bursts, "awlen"),
        }
        checks = {
            "read_burst_count_matches":
                actual["read_burst_count"] == expected["read_burst_count"],
            "write_burst_count_matches":
                actual["write_burst_count"] == expected["write_burst_count"],
            "read_burst_layout_matches": actual["read_burst_layout"] == [
                {"addr": b["addr"], "nwords": b["nwords"]} for b in expected["read_bursts"]
            ],
            "write_burst_layout_matches": actual["write_burst_layout"] == [
                {"addr": b["addr"], "nwords": b["nwords"]} for b in expected["write_bursts"]
            ],
        }
        report = {
            "n": n,
            "clk_period_ns": float(clk_period),
            "aximm_bitwidths": {k: int(v) for k, v in aximm_bw.items()},
            "expected": expected,
            "actual": actual,
            "checks": checks,
            "validated": all(checks.values()),
        }
        out_path = config.root_dir / "results" / "burst_info.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        if not report["validated"]:
            raise RuntimeError(
                "Unexpected AXI-MM bursts in the increment co-sim VCD. "
                f"Expected read/write counts "
                f"{expected['read_burst_count']}/{expected['write_burst_count']}, "
                f"got {actual['read_burst_count']}/{actual['write_burst_count']}. "
                f"See {out_path}."
            )
        return {"burst_report": out_path}


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
    # Testbench codegen (Phase 5): lower IncrTBHls.main() to gen/incr_tb.cpp,
    # producing the ``incr_tb`` artifact CSimStep consumes (replaces the
    # hand-written incr_tb.cpp).
    dag.add(HlsCodegenStep(
        name="gen_tb",
        comp_class=IncrTBHls,
        source_artifact="incr_source",
        output_dir="gen",
        is_testbench=True,
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

    # C-synth + RTL co-sim + AXI-MM burst validation (Phase 6, stretch).
    # C-sim above runs the kernel against a plain C array — it never exercises
    # the m_axi bus.  Co-sim drives the synthesized RTL master, and the burst
    # extraction proves the generated kernel issues the expected AXI-MM reads
    # and writes.
    dag.add(CosimStep(name="cosim"))
    dag.add(GenerateVcdStep(name="generate_vcd"))
    dag.add(ExtractBurstsStep(name="extract_bursts"))
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
