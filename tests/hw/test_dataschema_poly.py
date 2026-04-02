import re
import shutil
import subprocess
from enum import IntEnum
from pathlib import Path

import numpy as np
import pytest

from pysilicon.codegen.build import CodeGenConfig
from pysilicon.codegen.streamutils import copy_streamutils
from pysilicon.hw.arrayutils import gen_array_utils, write_uint32_file
from pysilicon.hw.dataschema import DataArray, DataList, EnumField, FloatField, IntField
from pysilicon.xilinxutils import toolchain


TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent.parent
POLY_EXAMPLE_DIR = REPO_ROOT / "examples" / "poly"
POLY_HPP_PATH = REPO_ROOT / "examples" / "poly" / "poly.hpp"

INCLUDE_DIR = "include"
WORD_BW_SUPPORTED = [32, 64]
U16 = IntField.specialize(bitwidth=16, signed=False)
Float32 = FloatField.specialize(bitwidth=32, include_dir=INCLUDE_DIR)


class PolyError(IntEnum):
    NO_ERROR = 0
    WRONG_NSAMP = 1


PolyErrorField = EnumField.specialize(enum_type=PolyError, include_dir=INCLUDE_DIR)


class CoeffArray(DataArray):
    element_type = Float32
    static = True
    max_shape = (4,)
    include_dir = INCLUDE_DIR


class PolyCmdHdr(DataList):
    elements = {
        "tx_id": {
            "schema": U16,
            "description": "Transaction ID",
        },
        "coeffs": {
            "schema": CoeffArray,
            "description": "Polynomial coefficients",
        },
        "nsamp": {
            "schema": U16,
            "description": "Number of samples",
        },
    }
    include_dir = INCLUDE_DIR


class PolyRespHdr(DataList):
    elements = {
        "tx_id": {
            "schema": U16,
            "description": "Echo of the Transaction ID sent in the command",
        },
    }
    include_dir = INCLUDE_DIR


class PolyRespFtr(DataList):
    elements = {
        "nsamp_read": {
            "schema": U16,
            "description": "Number of samples actually read and returned in the response",
        },
        "error": {
            "schema": PolyErrorField,
            "description": "Error code indicating success or type of failure",
        },
    }
    include_dir = INCLUDE_DIR


def polynomial_eval(
    cmd_hdr: PolyCmdHdr,
    samp_in: np.ndarray,
) -> tuple[PolyRespHdr, np.ndarray, PolyRespFtr]:
    resp_hdr = PolyRespHdr()
    resp_hdr.tx_id = cmd_hdr.tx_id

    coeffs = np.asarray(cmd_hdr.coeffs, dtype=np.float32)
    x = np.asarray(samp_in, dtype=np.float32)
    y = np.zeros_like(x)
    power = np.ones_like(x)
    for coeff in coeffs:
        y += coeff * power
        power *= x

    resp_ftr = PolyRespFtr()
    resp_ftr.nsamp_read = len(x)
    resp_ftr.error = PolyError.NO_ERROR if len(x) == int(cmd_hdr.nsamp) else PolyError.WRONG_NSAMP

    return resp_hdr, y, resp_ftr


def _poly_include_names() -> set[str]:
    content = POLY_HPP_PATH.read_text(encoding="utf-8")
    return set(re.findall(r'#include "include/([^"]+)"', content))


def _write_and_read_words_array(arr: np.ndarray, path: Path, **kwargs) -> np.ndarray:
    write_uint32_file(arr, elem_type=Float32, file_path=path, **kwargs)
    return np.fromfile(path, dtype="<u4")


def _float32_words(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype="<f4").view("<u4")


def _copy_poly_vitis_resources(dst_dir: Path) -> None:
    for name in ["poly.hpp", "poly.cpp", "poly_tb.cpp", "run.tcl"]:
        shutil.copy(POLY_EXAMPLE_DIR / name, dst_dir / name)


def test_poly_notebook_flow_generates_headers_vectors_and_expected_outputs(tmp_path: Path):
    cfg = CodeGenConfig(root_dir=tmp_path, util_dir=INCLUDE_DIR)

    schema_classes = [
        PolyErrorField,
        CoeffArray,
        PolyCmdHdr,
        PolyRespHdr,
        PolyRespFtr,
    ]
    for schema_class in schema_classes:
        schema_class.gen_include(cfg=cfg, word_bw_supported=WORD_BW_SUPPORTED)
    gen_array_utils(Float32, WORD_BW_SUPPORTED, cfg=cfg)
    copy_streamutils(cfg)

    include_root = tmp_path / INCLUDE_DIR
    generated_headers = {path.name for path in include_root.iterdir() if path.is_file()}
    assert _poly_include_names().issubset(generated_headers)

    coeffs = CoeffArray()
    coeffs.val = np.array([1.0, -2.0, -3.0, 4.0], dtype=np.float32)

    nsamp = 100
    cmd_hdr = PolyCmdHdr()
    cmd_hdr.tx_id = 42
    cmd_hdr.coeffs = coeffs.val
    cmd_hdr.nsamp = nsamp

    samp_in = np.linspace(0.0, 1.0, nsamp, dtype=np.float32)

    resp_hdr, samp_out, resp_ftr = polynomial_eval(cmd_hdr, samp_in)

    expected_y = np.polynomial.polynomial.polyval(samp_in, coeffs.val)
    assert int(resp_hdr.tx_id) == 42
    assert int(resp_ftr.nsamp_read) == nsamp
    assert resp_ftr.error is PolyError.NO_ERROR
    assert np.allclose(samp_out, expected_y, rtol=1e-6, atol=1e-6)

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    cmd_words = np.fromfile(cmd_hdr.write_uint32_file(data_dir / "cmd_hdr_data.bin"), dtype="<u4")
    samp_in_words = _write_and_read_words_array(samp_in, data_dir / "samp_in_data.bin", nwrite=cmd_hdr.nsamp)
    resp_hdr_words = np.fromfile(resp_hdr.write_uint32_file(data_dir / "resp_hdr_data.bin"), dtype="<u4")
    samp_out_words = _write_and_read_words_array(samp_out, data_dir / "samp_out_data.bin", nwrite=cmd_hdr.nsamp)
    resp_ftr_words = np.fromfile(resp_ftr.write_uint32_file(data_dir / "resp_ftr_data.bin"), dtype="<u4")

    assert np.array_equal(cmd_words, np.asarray(cmd_hdr.serialize(word_bw=32), dtype="<u4"))
    assert np.array_equal(samp_in_words, _float32_words(samp_in))
    assert np.array_equal(resp_hdr_words, np.asarray(resp_hdr.serialize(word_bw=32), dtype="<u4"))
    assert np.array_equal(samp_out_words, _float32_words(samp_out))
    assert np.array_equal(resp_ftr_words, np.asarray(resp_ftr.serialize(word_bw=32), dtype="<u4"))

    restored_cmd = PolyCmdHdr().read_uint32_file(data_dir / "cmd_hdr_data.bin")
    restored_resp_hdr = PolyRespHdr().read_uint32_file(data_dir / "resp_hdr_data.bin")
    restored_resp_ftr = PolyRespFtr().read_uint32_file(data_dir / "resp_ftr_data.bin")

    assert restored_cmd.is_close(cmd_hdr)
    assert restored_resp_hdr.is_close(resp_hdr)
    assert restored_resp_ftr.is_close(resp_ftr)


@pytest.mark.vitis
def test_poly_notebook_flow_runs_vitis_csim(tmp_path: Path):
    vitis_path = toolchain.find_vitis_path()
    if not vitis_path:
        pytest.skip("Vitis installation not found; skipping poly Vitis regression.")

    cfg = CodeGenConfig(root_dir=tmp_path, util_dir=INCLUDE_DIR)
    schema_classes = [
        PolyErrorField,
        CoeffArray,
        PolyCmdHdr,
        PolyRespHdr,
        PolyRespFtr,
    ]
    for schema_class in schema_classes:
        schema_class.gen_include(cfg=cfg, word_bw_supported=WORD_BW_SUPPORTED)
    gen_array_utils(Float32, WORD_BW_SUPPORTED, cfg=cfg)
    copy_streamutils(cfg)
    _copy_poly_vitis_resources(tmp_path)

    coeffs = CoeffArray()
    coeffs.val = np.array([1.0, -2.0, -3.0, 4.0], dtype=np.float32)

    nsamp = 100
    cmd_hdr = PolyCmdHdr()
    cmd_hdr.tx_id = 42
    cmd_hdr.coeffs = coeffs.val
    cmd_hdr.nsamp = nsamp

    samp_in = np.linspace(0.0, 1.0, nsamp, dtype=np.float32)

    expected_resp_hdr, expected_samp_out, expected_resp_ftr = polynomial_eval(cmd_hdr, samp_in)

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cmd_hdr.write_uint32_file(data_dir / "cmd_hdr_data.bin")
    write_uint32_file(samp_in, elem_type=Float32, file_path=data_dir / "samp_in_data.bin", nwrite=cmd_hdr.nsamp)

    try:
        toolchain.run_vitis_hls(tmp_path / "run.tcl", work_dir=tmp_path)
    except RuntimeError as exc:
        pytest.skip(f"Vitis execution unavailable in current setup: {exc}")
    except subprocess.CalledProcessError as exc:
        pytest.fail(
            "Vitis execution failed for poly dataschema regression.\n"
            f"Command: {exc.cmd}\n"
            f"Return code: {exc.returncode}\n"
            f"Stdout:\n{exc.stdout}\n"
            f"Stderr:\n{exc.stderr}"
        )

    got_resp_hdr = PolyRespHdr().read_uint32_file(data_dir / "resp_hdr_data.bin")
    got_resp_ftr = PolyRespFtr().read_uint32_file(data_dir / "resp_ftr_data.bin")
    got_samp_out_words = np.fromfile(data_dir / "samp_out_data.bin", dtype="<u4")
    got_samp_out = got_samp_out_words.view("<f4")

    assert got_resp_hdr.is_close(expected_resp_hdr)
    assert got_resp_ftr.is_close(expected_resp_ftr)
    assert np.allclose(got_samp_out, expected_samp_out[: got_samp_out.size], rtol=1e-6, atol=1e-6)