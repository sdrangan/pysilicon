import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples.histogram.hist_demo import HistError, HistTest
from pysilicon.toolchain import toolchain


HIST_EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "histogram"


def _copy_hist_vitis_resources(dst_dir: Path) -> None:
    """Copy the canonical histogram Vitis sources into a temporary test directory."""
    for name in ("hist.cpp", "hist.hpp", "hist_tb.cpp", "run.tcl"):
        shutil.copy(HIST_EXAMPLE_DIR / name, dst_dir / name)


@pytest.mark.parametrize("mem_dwidth", [32, 64, 128])
def test_hist_test_simulate_matches_expected_counts(mem_dwidth: int) -> None:
    hist_test = HistTest(seed=11, ndata=41, nbins=7, mem_dwidth=mem_dwidth)

    result = hist_test.simulate()

    assert hist_test.mem is not None
    assert hist_test.hist_accel is not None
    assert hist_test.cmd is not None
    assert hist_test.resp is not None
    assert hist_test.counts is not None
    assert hist_test.expected is not None

    assert hist_test.mem.word_size == mem_dwidth
    assert result.cmd is hist_test.cmd
    assert result.resp is hist_test.resp
    assert result.counts is hist_test.counts
    assert result.expected is hist_test.expected
    assert result.passed is True

    assert hist_test.resp.tx_id == hist_test.cmd.tx_id
    assert hist_test.resp.status is HistError.NO_ERROR
    assert hist_test.counts.dtype == np.uint32
    assert hist_test.expected.dtype == np.uint32
    assert np.array_equal(hist_test.counts, hist_test.expected)


def test_hist_test_gen_test_data_initializes_state_before_simulate() -> None:
    hist_test = HistTest(seed=5, ndata=13, nbins=4, mem_dwidth=64)

    hist_test.gen_test_data()

    assert hist_test.cmd is None
    assert hist_test.data is not None
    assert hist_test.bin_edges is not None
    assert hist_test.data.shape == (13,)
    assert hist_test.bin_edges.shape == (3,)

    result = hist_test.simulate()

    assert hist_test.cmd is not None
    assert result.passed is True


def test_hist_test_vitis_stage_range_passes_tcl_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _copy_hist_vitis_resources(tmp_path)
    hist_test = HistTest(seed=11, ndata=41, nbins=7, example_dir=tmp_path)
    expected_result = hist_test.simulate()

    captured: dict[str, object] = {}

    def fake_gen_vitis_code() -> list[Path]:
        return []

    def fake_write_input_files(data_dir: Path | None = None) -> Path:
        target_dir = tmp_path / "data"
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def fake_run_vitis_hls(
        run_tcl: Path,
        work_dir: Path,
        capture_output: bool,
        env: dict[str, str] | None = None,
    ):
        captured["run_tcl"] = run_tcl
        captured["work_dir"] = work_dir
        captured["capture_output"] = capture_output
        captured["env"] = env
        return SimpleNamespace(stdout="hist csynth ok", stderr="")

    def fake_read_vitis_outputs(data_dir: Path):
        captured["data_dir"] = data_dir
        return expected_result

    monkeypatch.setattr(hist_test, "gen_vitis_code", fake_gen_vitis_code)
    monkeypatch.setattr(hist_test, "write_input_files", fake_write_input_files)
    monkeypatch.setattr(hist_test, "read_vitis_outputs", fake_read_vitis_outputs)
    monkeypatch.setattr(toolchain, "run_vitis_hls", fake_run_vitis_hls)

    result = hist_test.test_vitis(start_at="csim", through="csynth")

    assert result is expected_result
    assert captured["run_tcl"] == tmp_path / "run.tcl"
    assert captured["work_dir"] == tmp_path
    assert captured["capture_output"] is True
    assert captured["data_dir"] == tmp_path / "data"
    assert captured["env"] == {
        "PYSILICON_HIST_START_AT": "csim",
        "PYSILICON_HIST_THROUGH": "csynth",
        "PYSILICON_HIST_TRACE_LEVEL": "none",
    }


def test_hist_test_vitis_rejects_invalid_stage_range() -> None:
    hist_test = HistTest()

    with pytest.raises(ValueError, match="must not come after"):
        hist_test.test_vitis(start_at="cosim", through="csynth")


def test_hist_test_generate_vcd_stage_skips_vitis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _copy_hist_vitis_resources(tmp_path)
    hist_test = HistTest(seed=11, ndata=41, nbins=7, example_dir=tmp_path)
    project_dir = tmp_path / "pysilicon_hist_proj" / "solution1"
    project_dir.mkdir(parents=True, exist_ok=True)

    captured: dict[str, object] = {}

    def fake_generate_vcd(output_vcd: str = "dump.vcd", soln: str | None = "solution1", trace_level: str = "*") -> Path:
        captured["output_vcd"] = output_vcd
        captured["soln"] = soln
        captured["trace_level"] = trace_level
        return tmp_path / "vcd" / output_vcd

    def fail_run_vitis_hls(*args, **kwargs):
        raise AssertionError("Vitis should not be invoked for the generate_vcd-only stage.")

    monkeypatch.setattr(hist_test, "generate_vcd", fake_generate_vcd)
    monkeypatch.setattr(toolchain, "run_vitis_hls", fail_run_vitis_hls)

    result = hist_test.test_vitis(start_at="generate_vcd", through="generate_vcd")

    assert result is None
    assert captured == {
        "output_vcd": "dump.vcd",
        "soln": "solution1",
        "trace_level": "*",
    }


@pytest.mark.vitis
def test_hist_test_vitis_matches_python_model(tmp_path: Path) -> None:
    vitis_path = toolchain.find_vitis_path()
    if not vitis_path:
        pytest.skip("Vitis installation not found; skipping histogram Vitis regression.")

    _copy_hist_vitis_resources(tmp_path)
    hist_test = HistTest(seed=11, ndata=41, nbins=7, example_dir=tmp_path)

    try:
        result = hist_test.test_vitis()
    except RuntimeError as exc:
        pytest.skip(f"Vitis execution unavailable in current setup: {exc}")
    except subprocess.CalledProcessError as exc:
        pytest.fail(
            "Vitis execution failed for histogram regression.\n"
            f"Command: {exc.cmd}\n"
            f"Return code: {exc.returncode}\n"
            f"Stdout:\n{exc.stdout}\n"
            f"Stderr:\n{exc.stderr}"
        )

    assert hist_test.resp is not None
    assert hist_test.expected is not None
    assert result.passed is True
    assert result.resp.status is HistError.NO_ERROR
    assert np.array_equal(result.counts, hist_test.expected)