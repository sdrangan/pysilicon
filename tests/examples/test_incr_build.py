"""Phase 4 tests: the increment build DAG.

The non-Vitis tests exercise the Python golden model + m_axi kernel codegen
branches.  The Vitis-gated test runs C-simulation of the generated kernel
against the Python model (the milestone — decision 11).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from examples.increment.incr_build import build_incr_dag
from pysilicon.build.build import BuildConfig
from pysilicon.toolchain import toolchain

import pytest


INCR_DIR = Path(__file__).resolve().parents[2] / "examples" / "increment"


def _copy_vitis_resources(dst: Path) -> None:
    """Seed the non-generated Vitis sources (run.tcl + hook impls); the build
    DAG generates the kernel AND the testbench into gen/ and include/."""
    for name in ("run.tcl", "incr_transform_impl.cpp", "incr_respond_impl.tpp"):
        shutil.copy(INCR_DIR / name, dst / name)


# ---------------------------------------------------------------------------
# Non-Vitis: golden model + codegen branches
# ---------------------------------------------------------------------------

def test_build_inputs_and_pysim_golden(tmp_path):
    cfg = BuildConfig(root_dir=tmp_path, params={"n": 37})
    res = build_incr_dag().run(cfg, through="py_sim")
    assert res["build_inputs"].success
    assert res["py_sim"].success
    # py_sim wrote the golden out.bin = in + 1.
    in_buf = np.fromfile(tmp_path / "data" / "in.bin", dtype="<u4")[:37]
    out_buf = np.fromfile(tmp_path / "results" / "sim" / "out.bin", dtype="<u4")[:37]
    np.testing.assert_array_equal(out_buf, in_buf + 1)


def test_kernel_codegen_branch(tmp_path):
    cfg = BuildConfig(root_dir=tmp_path, params={"n": 37})
    res = build_incr_dag().run(cfg, through="gen_kernel")
    assert res["gen_kernel"].success
    cpp = (tmp_path / "gen" / "incr.cpp").read_text()
    assert "#pragma HLS INTERFACE m_axi port=m_mem offset=slave bundle=gmem depth=1024" in cpp
    assert "#pragma HLS INTERFACE ap_ctrl_hs port=return" in cpp
    assert "uint32_array_utils::read_array<32>(" in cpp
    assert "uint32_array_utils::write_array<32>(" in cpp
    assert "memmgr::byte_addr_to_word_index<32>(cmd.addr)" in cpp


def test_gen_include_provisions_memmgr_and_headers(tmp_path):
    cfg = BuildConfig(root_dir=tmp_path, params={"n": 37})
    res = build_incr_dag().run(cfg, through="gen_include")
    assert res["gen_include"].success
    inc = tmp_path / "include"
    for name in ("memmgr.hpp", "memmgr_tb.hpp", "incr_cmd.h", "incr_resp.h",
                 "incr_error.h", "uint32_array_utils.h"):
        assert (inc / name).exists(), f"missing {name}"


# ---------------------------------------------------------------------------
# Vitis-gated: the milestone — generated kernel passes C-sim
# ---------------------------------------------------------------------------

@pytest.mark.vitis
def test_incr_vitis_csim_matches_python(tmp_path):
    if not toolchain.find_vitis_path():
        pytest.skip("Vitis installation not found; skipping increment C-sim.")

    _copy_vitis_resources(tmp_path)
    cfg = BuildConfig(root_dir=tmp_path, params={"n": 37})
    try:
        results = build_incr_dag().run(cfg, through="validate_csim")
    except subprocess.CalledProcessError as exc:
        pytest.fail(f"Vitis C-sim failed: {exc}\n{exc.stdout}\n{exc.stderr}")

    # Vitis was found (gate above), so a csim failure is a real failure — do not
    # mask it as a skip. (An earlier soft-skip here hid a build-wiring bug that
    # made C-sim fail to compile while the test still reported "passed".)
    csim = results.get("csim")
    assert csim is not None and csim.success, (
        f"C-sim failed: {csim.message if csim else 'csim step did not run'}"
    )
    assert results["validate_csim"].success, results["validate_csim"].message


@pytest.mark.vitis
def test_incr_vitis_cosim_bursts_match_expectation(tmp_path):
    """C-synth + RTL co-sim + AXI-MM burst validation (Phase 6).

    Unlike C-sim (which runs against a plain C array and never touches the
    m_axi bus), co-sim drives the synthesized RTL master.  The burst-extraction
    step asserts the generated kernel issues exactly the expected AXI-MM reads
    and writes (one read region + one write region of n words, split by the
    kernel's max burst length).
    """
    if not toolchain.find_vitis_path():
        pytest.skip("Vitis installation not found; skipping increment co-sim.")

    _copy_vitis_resources(tmp_path)
    cfg = BuildConfig(root_dir=tmp_path, params={"n": 37, "trace_level": "port"})
    try:
        results = build_incr_dag().run(cfg, through="extract_bursts")
    except subprocess.CalledProcessError as exc:
        pytest.fail(f"Vitis co-sim failed: {exc}\n{exc.stdout}\n{exc.stderr}")

    # Toolchain gate has cleared, so every Vitis step is a hard assertion — do
    # not soft-skip a real co-sim or burst-validation failure.
    for step in ("cosim", "generate_vcd", "extract_bursts"):
        r = results.get(step)
        assert r is not None and r.success, (
            f"{step} failed: {r.message if r else f'{step} did not run'}"
        )

    report = json.loads(results["extract_bursts"].path("burst_report").read_text())
    assert report["validated"] is True, report["checks"]
    assert report["expected"]["read_burst_count"] == report["actual"]["read_burst_count"]
    assert report["expected"]["write_burst_count"] == report["actual"]["write_burst_count"]
