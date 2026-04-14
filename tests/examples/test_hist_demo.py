import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples.histogram import hist_demo
from examples.histogram.hist_demo import HistError, HistSimResult, HistTest
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


def test_hist_kernel_only_scans_programmed_bin_edges() -> None:
    hist_cpp = (HIST_EXAMPLE_DIR / "hist.cpp").read_text(encoding="utf-8")

    assert "for (int b = 0; b < nbins - 1; ++b)" in hist_cpp
    assert "for (int b = 0; b < max_nbins; b++)" not in hist_cpp


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


def test_hist_test_generate_vcd_clamps_vitis_through_to_cosim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
        captured["env"] = env
        return SimpleNamespace(stdout="hist cosim ok", stderr="")

    def fake_read_vitis_outputs(data_dir: Path):
        return expected_result

    def fake_generate_vcd(output_vcd: str = "dump.vcd", soln: str | None = "solution1", trace_level: str = "*") -> Path:
        captured["generate_vcd"] = {
            "output_vcd": output_vcd,
            "soln": soln,
            "trace_level": trace_level,
        }
        return tmp_path / "vcd" / output_vcd

    monkeypatch.setattr(hist_test, "gen_vitis_code", fake_gen_vitis_code)
    monkeypatch.setattr(hist_test, "write_input_files", fake_write_input_files)
    monkeypatch.setattr(hist_test, "read_vitis_outputs", fake_read_vitis_outputs)
    monkeypatch.setattr(hist_test, "generate_vcd", fake_generate_vcd)
    monkeypatch.setattr(toolchain, "run_vitis_hls", fake_run_vitis_hls)

    result = hist_test.test_vitis(start_at="csim", through="generate_vcd", trace_level="port")

    assert result is expected_result
    assert captured["env"] == {
        "PYSILICON_HIST_START_AT": "csim",
        "PYSILICON_HIST_THROUGH": "cosim",
        "PYSILICON_HIST_TRACE_LEVEL": "port",
    }
    assert captured["generate_vcd"] == {
        "output_vcd": "dump.vcd",
        "soln": "solution1",
        "trace_level": "port",
    }


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


def test_hist_test_extract_bursts_only_stage_skips_vitis_and_vcd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _copy_hist_vitis_resources(tmp_path)
    hist_test = HistTest(seed=11, ndata=41, nbins=7, example_dir=tmp_path)
    captured: dict[str, object] = {}

    def fake_extract_bursts(vcd_path: str | Path | None = None, output_json: str | Path | None = None) -> dict[str, object]:
        captured["vcd_path"] = vcd_path
        captured["output_json"] = output_json
        return {
            "actual": {"read_burst_count": 2, "write_burst_count": 1},
            "output_json": str(tmp_path / "vcd" / "burst_info.json"),
        }

    def fail_generate_vcd(*args, **kwargs):
        raise AssertionError("generate_vcd should not run for the extract_bursts-only stage.")

    def fail_run_vitis_hls(*args, **kwargs):
        raise AssertionError("Vitis should not be invoked for the extract_bursts-only stage.")

    monkeypatch.setattr(hist_test, "extract_bursts", fake_extract_bursts)
    monkeypatch.setattr(hist_test, "generate_vcd", fail_generate_vcd)
    monkeypatch.setattr(toolchain, "run_vitis_hls", fail_run_vitis_hls)

    report = hist_test.test_vitis(start_at="extract_bursts", through="extract_bursts")

    assert report == {
        "actual": {"read_burst_count": 2, "write_burst_count": 1},
        "output_json": str(tmp_path / "vcd" / "burst_info.json"),
    }
    assert captured == {"vcd_path": None, "output_json": None}


def test_hist_test_extract_bursts_stage_runs_after_generate_vcd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _copy_hist_vitis_resources(tmp_path)
    hist_test = HistTest(seed=11, ndata=41, nbins=7, example_dir=tmp_path)
    project_dir = tmp_path / "pysilicon_hist_proj" / "solution1"
    project_dir.mkdir(parents=True, exist_ok=True)

    captured: dict[str, object] = {}

    def fake_generate_vcd(output_vcd: str = "dump.vcd", soln: str | None = "solution1", trace_level: str = "*") -> Path:
        captured["generate_vcd"] = {
            "output_vcd": output_vcd,
            "soln": soln,
            "trace_level": trace_level,
        }
        return tmp_path / "vcd" / output_vcd

    def fake_extract_bursts(vcd_path: str | Path | None = None, output_json: str | Path | None = None) -> dict[str, object]:
        captured["extract_bursts"] = {
            "vcd_path": vcd_path,
            "output_json": output_json,
        }
        return {
            "actual": {"read_burst_count": 2, "write_burst_count": 1},
            "output_json": str(tmp_path / "vcd" / "burst_info.json"),
        }

    def fail_run_vitis_hls(*args, **kwargs):
        raise AssertionError("Vitis should not be invoked when starting at generate_vcd.")

    monkeypatch.setattr(hist_test, "generate_vcd", fake_generate_vcd)
    monkeypatch.setattr(hist_test, "extract_bursts", fake_extract_bursts)
    monkeypatch.setattr(toolchain, "run_vitis_hls", fail_run_vitis_hls)

    report = hist_test.test_vitis(start_at="generate_vcd", through="extract_bursts", trace_level="port")

    assert report == {
        "actual": {"read_burst_count": 2, "write_burst_count": 1},
        "output_json": str(tmp_path / "vcd" / "burst_info.json"),
    }
    assert captured == {
        "generate_vcd": {
            "output_vcd": "dump.vcd",
            "soln": "solution1",
            "trace_level": "port",
        },
        "extract_bursts": {
            "vcd_path": tmp_path / "vcd" / "dump.vcd",
            "output_json": None,
        },
    }


def test_hist_main_reports_burst_report_without_accessing_counts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = {
        "actual": {"read_burst_count": 4, "write_burst_count": 1},
        "output_json": "C:/tmp/burst_info.json",
    }

    class FakeHistTest:
        def __init__(self, seed: int, ndata: int, nbins: int):
            self.seed = seed
            self.ndata = ndata
            self.nbins = nbins

        def simulate(self):
            raise AssertionError("simulate should not be called when starting at extract_bursts")

        def test_vitis(self, start_at: str, through: str, trace_level: str, live_output: bool):
            assert start_at == "extract_bursts"
            assert through == "extract_bursts"
            assert trace_level == "none"
            assert live_output is True
            return report

    monkeypatch.setattr(hist_demo, "HistTest", FakeHistTest)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hist_demo.py",
            "--start_at",
            "extract_bursts",
            "--through",
            "extract_bursts",
            "--live_output",
        ],
    )

    hist_demo.main()

    out = capsys.readouterr().out
    assert "Burst extraction report generated." in out
    assert "output_json=C:/tmp/burst_info.json" in out


def test_hist_main_reports_counts_for_hist_sim_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    counts = np.array([1, 2, 3], dtype=np.uint32)
    sim_result = HistSimResult(
        cmd=SimpleNamespace(),
        resp=SimpleNamespace(tx_id=11, status=SimpleNamespace(name="NO_ERROR")),
        counts=counts,
        expected=counts.copy(),
    )

    class FakeHistTest:
        def __init__(self, seed: int, ndata: int, nbins: int):
            self.seed = seed
            self.ndata = ndata
            self.nbins = nbins

        def simulate(self):
            return sim_result

        def test_vitis(self, start_at: str, through: str, trace_level: str, live_output: bool):
            assert start_at == "csim"
            assert through == "csim"
            return sim_result

    monkeypatch.setattr(hist_demo, "HistTest", FakeHistTest)
    monkeypatch.setattr(sys, "argv", ["hist_demo.py"])

    hist_demo.main()

    out = capsys.readouterr().out
    assert "Python simulation: tx_id=11, status=NO_ERROR, passed=True" in out
    assert "Vitis simulation matched Python model. counts=[1 2 3]" in out


def test_burst_to_jsonable_adds_fixed_width_hex_words() -> None:
    burst = {
        "addr": 0,
        "start_idx": 1,
        "tstart": 10.0,
        "data_start_idx": 2,
        "data_end_idx": 3,
        "data_tstart": 20.0,
        "data_tend": 30.0,
        "queue_wait_cycles": 1,
        "beat_type": [0, 1],
        "data": np.array([-1, 16], dtype=np.int32),
        "awlen": None,
        "arlen": 1,
    }

    result = HistTest._burst_to_jsonable(burst, data_bitwidth=32)

    assert result["data"] == [-1, 16]
    assert result["data_hex"] == ["0xffffffff", "0x00000010"]
    assert result["beat_type_names"] == ["transfer", "idle"]


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