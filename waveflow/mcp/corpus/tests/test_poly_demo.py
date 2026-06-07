import csv
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from examples.poly.poly_demo import PolyError, build_poly_dag
from waveflow.build.build import BuildConfig
from waveflow.toolchain import toolchain


def _read_event_times(log_file: Path) -> dict[str, float]:
    """Return first occurrence of each event name → simulation time (seconds)."""
    events: dict[str, float] = {}
    with open(log_file, newline='') as f:
        for row in csv.DictReader(f):
            ev = row['event']
            if ev not in events:
                events[ev] = float(row['time'])
    return events


POLY_EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "poly"


def _copy_poly_vitis_resources(dst_dir: Path) -> None:
    """Copy the canonical poly Vitis sources into a temporary test directory."""
    for name in ("poly.cpp", "poly.hpp", "poly_tb.cpp", "run.tcl"):
        shutil.copy(POLY_EXAMPLE_DIR / name, dst_dir / name)


def test_poly_simulate_matches_expected_outputs() -> None:
    results = build_poly_dag(nsamp=100).run(BuildConfig(), through='ValidateTimingStep')

    assert results['BuildInputsStep'].success
    assert results['PySimStep'].success

    sim_result = results['PySimStep'].object('sim_result')

    assert sim_result.cmd_hdr is not None
    assert sim_result.samp_in is not None
    assert sim_result.resp_hdr is not None
    assert sim_result.samp_out is not None
    assert sim_result.resp_ftr is not None
    assert sim_result.passed is True
    assert sim_result.resp_ftr.error is PolyError.NO_ERROR
    assert sim_result.samp_out.dtype == np.float32


def test_poly_timing_bandwidth_and_unroll(tmp_path: Path) -> None:
    """Timing scales with unroll_factor and interface bitwidth.

    Three configurations at nsamp=100, proc_latency=10, clk=1 GHz:

    Case 1 — uf=1, bw=32: each sample is one 32-bit word; bandwidth-limited
      at 1 sample/cycle.  Expected duration ≈ (nsamp + proc_latency) cycles.

    Case 2 — uf=2, bw=32: unroll=2 but the 32-bit interface only carries one
      Float32 per clock, so still bandwidth-limited at 1 sample/cycle.
      Duration should match case 1.

    Case 3 — uf=2, bw=64: two Float32 samples packed per 64-bit word; both
      compute and bandwidth scale with unroll_factor.
      Expected duration ≈ (nsamp/uf + proc_latency) cycles ≈ half of case 1.
    """
    nsamp = 100
    proc_latency = 10
    period = 1e-9  # 1 GHz

    configs = [
        dict(in_bw=32, out_bw=32, unroll_factor=1,
             log_file=tmp_path / "python_sim_log_uf1_bw32.csv"),
        dict(in_bw=32, out_bw=32, unroll_factor=2,
             log_file=tmp_path / "python_sim_log_uf2_bw32.csv"),
        dict(in_bw=64, out_bw=64, unroll_factor=2,
             log_file=tmp_path / "python_sim_log_uf2_bw64.csv"),
    ]

    durations: list[float] = []
    for cfg in configs:
        results = build_poly_dag(
            nsamp=nsamp,
            in_bw=cfg['in_bw'],
            out_bw=cfg['out_bw'],
            unroll_factor=cfg['unroll_factor'],
            log_file=cfg['log_file'],
        ).run(BuildConfig(), through='ValidateTimingStep')
        sim_result = results['PySimStep'].object('sim_result')
        assert sim_result.passed, f"Simulation failed for config {cfg}"
        events = _read_event_times(cfg['log_file'])
        durations.append(events['samp_out_write_end'] - events['samp_read_begin'])

    dur_uf1_bw32, dur_uf2_bw32, dur_uf2_bw64 = durations
    tol = 5 * period  # ±5 clock cycles to allow for timing discretisation

    # Cases 1 and 2: bandwidth-limited — timing should be identical
    expected_bw_limited = (nsamp + proc_latency) * period
    assert abs(dur_uf1_bw32 - expected_bw_limited) < tol, (
        f"Case 1 (uf=1, bw=32): expected ~{expected_bw_limited*1e9:.0f} ns, "
        f"got {dur_uf1_bw32*1e9:.1f} ns"
    )
    assert abs(dur_uf2_bw32 - expected_bw_limited) < tol, (
        f"Case 2 (uf=2, bw=32): expected ~{expected_bw_limited*1e9:.0f} ns (BW-limited), "
        f"got {dur_uf2_bw32*1e9:.1f} ns"
    )
    assert abs(dur_uf2_bw32 - dur_uf1_bw32) < tol, (
        f"Cases 1 and 2 should match (both BW-limited): "
        f"{dur_uf1_bw32*1e9:.1f} ns vs {dur_uf2_bw32*1e9:.1f} ns"
    )

    # Case 3: compute + BW not limited — duration halved
    expected_bw64 = (nsamp // 2 + proc_latency) * period
    assert abs(dur_uf2_bw64 - expected_bw64) < tol, (
        f"Case 3 (uf=2, bw=64): expected ~{expected_bw64*1e9:.0f} ns, "
        f"got {dur_uf2_bw64*1e9:.1f} ns"
    )
    assert dur_uf2_bw64 < dur_uf1_bw32 * 0.75, (
        f"Case 3 should be significantly faster than case 1: "
        f"{dur_uf2_bw64*1e9:.1f} ns vs {dur_uf1_bw32*1e9:.1f} ns"
    )


@pytest.mark.vitis
def test_poly_vitis_cosim_matches_python_model(tmp_path: Path) -> None:
    if not toolchain.find_vitis_path():
        pytest.skip("Vitis installation not found; skipping poly Vitis co-sim regression.")

    _copy_poly_vitis_resources(tmp_path)

    results = build_poly_dag(
        nsamp=100,
        example_dir=tmp_path,
        log_file=tmp_path / "sim_log.csv",
    ).run(BuildConfig(root_dir=tmp_path))

    csim_result = results.get('CSimStep')
    if csim_result is None or not csim_result.success:
        msg = csim_result.message if csim_result else "CSimStep did not run"
        pytest.skip(f"Vitis execution unavailable in current setup: {msg}")

    assert results['ValidateCSimStep'].success, results['ValidateCSimStep'].message

    vitis_result = results['ValidateCSimStep'].object('vitis_result')
    sim_result = results['PySimStep'].object('sim_result')
    # Vitis 2025.1 writes poly_cosim.rpt; older versions wrote cosim.log
    sim_report_dir = tmp_path / "waveflow_poly_proj" / "solution1" / "sim" / "report"
    cosim_reports = list(sim_report_dir.glob("*cosim*")) if sim_report_dir.exists() else []

    assert vitis_result.passed is True
    assert vitis_result.resp_ftr.error is PolyError.NO_ERROR
    assert np.allclose(vitis_result.samp_out, sim_result.samp_out[:vitis_result.samp_out.size], rtol=1e-6, atol=1e-6)
    assert cosim_reports, f"No cosim report found in {sim_report_dir}"
