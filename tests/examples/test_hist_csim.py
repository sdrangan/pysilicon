"""The GENERATED histogram kernel + GENERATED testbench pass Vitis C-sim.

Generates the kernel + header from HistAccel and the testbench from HistTBHls
(into gen/), compiles them with the hand-written datapath hooks via run.tcl, and
runs C-sim across coverage cases — asserting status and counts match the
HistogramAccel golden. Vitis-gated; the non-vitis suite skips it.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from examples.shared_mem.hist_build import CSIM_CASES, generate_vitis_sources
from waveflow.toolchain import toolchain

SHARED_DIR = Path(__file__).resolve().parents[2] / "examples" / "shared_mem"
_RESOURCES = (
    "run.tcl",
    "hist_validate_impl.cpp",
    "hist_compute_impl.cpp",
    "hist_respond_impl.tpp",
)


@pytest.mark.vitis
def test_generated_hist_kernel_passes_csim(tmp_path):
    """The generated kernel matches the golden under C-sim across nbins==1,
    nbins>1 (multiple bins), and a validation-failure case."""
    if not toolchain.find_vitis_path():
        pytest.skip("Vitis installation not found; skipping generated hist C-sim.")

    for name in _RESOURCES:
        shutil.copy(SHARED_DIR / name, tmp_path / name)
    generate_vitis_sources(tmp_path)

    data_dir = tmp_path / "data"
    failures = []
    for case in CSIM_CASES:
        data, edges = case.write_inputs(data_dir)
        try:
            toolchain.run_vitis_hls(tmp_path / "run.tcl", work_dir=tmp_path,
                                    capture_output=True)
        except subprocess.CalledProcessError as exc:
            # Vitis was found (gate above), so a csim failure is a real failure —
            # do NOT soft-skip it.
            pytest.fail(
                f"Vitis C-sim failed for ndata={case.ndata} nbins={case.nbins}:\n"
                f"{exc.stdout}\n{exc.stderr}"
            )
        ok, detail = case.check_outputs(data_dir, data, edges)
        marker = "ok" if ok else "FAIL"
        print(f"  [{marker}] ndata={case.ndata} nbins={case.nbins}: {detail}")
        if not ok:
            failures.append(f"ndata={case.ndata} nbins={case.nbins}: {detail}")

    assert not failures, "Generated kernel C-sim mismatches:\n" + "\n".join(failures)
