"""Phase 6: the GENERATED histogram kernel + TB pass RTL co-simulation, and the
AXI-MM burst report shows the multi-buffer pattern (data + edges reads, counts
write) the increment toy never exercised.

Drives gen/hist.cpp + gen/hist_tb.cpp through Vitis csim -> csynth -> cosim, then
reuses HistTest's generate_vcd / extract_bursts machinery to validate the bursts
and CosimReportParser for the measured latency. Vitis + Vivado/xsim gated.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from examples.shared_mem.hist_build import HistCase, generate_vitis_sources
from waveflow.toolchain import toolchain

SHARED_DIR = Path(__file__).resolve().parents[2] / "examples" / "shared_mem"
_RESOURCES = (
    "run.tcl",
    "hist_validate_impl.cpp",
    "hist_compute_impl.cpp",
    "hist_respond_impl.tpp",
)


@pytest.mark.vitis
def test_generated_hist_cosim_and_bursts(tmp_path):
    """RTL cosim of the generated artifacts matches the golden; the burst report
    shows two read regions (data, bin_edges) and one write region (counts) with
    sizes matching the test vector."""
    from examples.shared_mem.hist_build import HistTest

    if not toolchain.find_vitis_path():
        pytest.skip("Vitis installation not found; skipping generated hist cosim.")

    for name in _RESOURCES:
        shutil.copy(SHARED_DIR / name, tmp_path / name)
    generate_vitis_sources(tmp_path)

    # One valid vector rich enough to split the data read into several bursts.
    case = HistCase(ndata=37, nbins=6)
    data, edges = case.write_inputs(tmp_path / "data")

    # csim -> csynth -> cosim with port tracing (so a VCD can be generated).
    try:
        toolchain.run_vitis_hls(
            tmp_path / "run.tcl", work_dir=tmp_path, capture_output=True,
            env={
                "WAVEFLOW_HIST_START_AT": "csim",
                "WAVEFLOW_HIST_THROUGH": "cosim",
                "WAVEFLOW_HIST_TRACE_LEVEL": "port",
            },
        )
    except subprocess.CalledProcessError as exc:
        pytest.fail(f"Vitis cosim failed:\n{exc.stdout}\n{exc.stderr}")

    # Cosim wrote the response + counts; confirm they still match the golden.
    ok, detail = case.check_outputs(tmp_path / "data", data, edges)
    assert ok, f"cosim output mismatch: {detail}"

    # Burst extraction via the HistTest machinery (expected summary is computed
    # from the same vector; addresses line up because the generated TB allocs in
    # the same order/size as the SimPy controller).
    ht = HistTest(example_dir=tmp_path, ndata=37, nbins=6)
    ht.simulate()
    # No soft-skip on the VCD/burst step: cosim already ran above, so a failure
    # here is a real failure to surface, not something to mask.
    vcd_path = ht.generate_vcd(trace_level="port")
    report = ht.extract_bursts(vcd_path=vcd_path)

    exp = report["expected"]
    act = report["actual"]
    read_regions = [r["name"] for r in exp["read_regions"]]
    write_regions = [r["name"] for r in exp["write_regions"]]
    print(f"  read regions : {read_regions}")
    print(f"  write regions: {write_regions}")
    print(f"  read bursts  : {act['read_burst_count']} (expected {exp['read_burst_count']})")
    print(f"  write bursts : {act['write_burst_count']} (expected {exp['write_burst_count']})")
    print(f"  read layout  : {act['read_burst_layout']}")
    print(f"  write layout : {act['write_burst_layout']}")

    # The multi-buffer m_axi pattern: data + bin_edges reads, counts write.
    assert read_regions == ["data", "bin_edges"], read_regions
    assert write_regions == ["counts"], write_regions
    # Sizes sensible for the vector: 37 data words + 5 edge words read, 6 counts
    # words written (extract_bursts already asserts the per-burst layout matches).
    assert sum(r["nwords"] for r in exp["read_regions"]) == 37 + 5
    assert sum(r["nwords"] for r in exp["write_regions"]) == 6
    assert report["validated"]

    # Timing: the cosim per-transaction latency (existing machinery).
    from waveflow.utils.cosimparse import CosimReportParser
    sol = tmp_path / "waveflow_hist_proj" / "solution1"
    cycles = CosimReportParser(sol_path=sol, top="hist").get_transaction_cycles()
    print(f"  cosim latency: {cycles} cycles @ {report['clk_period_ns']} ns")
    assert cycles is None or cycles > 0
