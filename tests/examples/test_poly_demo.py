import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from examples.poly.poly_demo import PolyError, PolyTest
from pysilicon.toolchain import toolchain


POLY_EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "poly"


def _copy_poly_vitis_resources(dst_dir: Path) -> None:
    """Copy the canonical poly Vitis sources into a temporary test directory."""
    for name in ("poly.cpp", "poly.hpp", "poly_tb.cpp", "run.tcl"):
        shutil.copy(POLY_EXAMPLE_DIR / name, dst_dir / name)


def test_poly_test_simulate_matches_expected_outputs() -> None:
    poly_test = PolyTest(nsamp=100)

    result = poly_test.simulate()

    assert poly_test.cmd_hdr is not None
    assert poly_test.samp_in is not None
    assert poly_test.resp_hdr is not None
    assert poly_test.samp_out is not None
    assert poly_test.resp_ftr is not None

    assert result.cmd_hdr is poly_test.cmd_hdr
    assert result.samp_in is poly_test.samp_in
    assert result.resp_hdr is poly_test.resp_hdr
    assert result.samp_out is poly_test.samp_out
    assert result.resp_ftr is poly_test.resp_ftr
    assert result.passed is True
    assert result.resp_ftr.error is PolyError.NO_ERROR
    assert result.samp_out.dtype == np.float32


@pytest.mark.vitis
def test_poly_test_vitis_cosim_matches_python_model(tmp_path: Path) -> None:
    vitis_path = toolchain.find_vitis_path()
    if not vitis_path:
        pytest.skip("Vitis installation not found; skipping poly Vitis co-sim regression.")

    _copy_poly_vitis_resources(tmp_path)
    poly_test = PolyTest(nsamp=100, example_dir=tmp_path)

    try:
        result = poly_test.test_vitis(cosim=True, trace_level="none")
    except RuntimeError as exc:
        pytest.skip(f"Vitis execution unavailable in current setup: {exc}")
    except subprocess.CalledProcessError as exc:
        pytest.fail(
            "Vitis execution failed for poly co-sim regression.\n"
            f"Command: {exc.cmd}\n"
            f"Return code: {exc.returncode}\n"
            f"Stdout:\n{exc.stdout}\n"
            f"Stderr:\n{exc.stderr}"
        )

    cosim_log = tmp_path / "pysilicon_poly_proj" / "solution1" / "sim" / "report" / "cosim.log"

    assert poly_test.resp_ftr is not None
    assert poly_test.samp_out is not None
    assert result.passed is True
    assert result.resp_ftr.error is PolyError.NO_ERROR
    assert np.allclose(result.samp_out, poly_test.samp_out[: result.samp_out.size], rtol=1e-6, atol=1e-6)
    assert cosim_log.exists()
