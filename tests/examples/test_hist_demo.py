import shutil
import subprocess
from pathlib import Path

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