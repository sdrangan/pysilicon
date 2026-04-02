import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pysilicon.codegen.build import CodeGenConfig
from pysilicon.codegen.streamutils import copy_streamutils
from pysilicon.hw.arrayutils import gen_array_utils, read_array, write_array
from pysilicon.hw.dataschema import FloatField, IntField
from pysilicon.xilinxutils import toolchain


F32 = FloatField.specialize(bitwidth=32, include_dir="include")
S16 = IntField.specialize(bitwidth=16, signed=True, include_dir="include")

TEST_DIR = Path(__file__).parent
RESOURCE_DIR = TEST_DIR / "resources"
ROUNDTRIP_CPP_PATH = RESOURCE_DIR / "arrayutils_roundtrip_test.cpp"
ROUNDTRIP_TCL_PATH = RESOURCE_DIR / "arrayutils_roundtrip_run.tcl"


def _run_vitis_tcl(tcl_path: Path, work_dir: Path, failure_prefix: str) -> None:
    try:
        toolchain.run_vitis_hls(tcl_path, work_dir=work_dir)
    except RuntimeError as exc:
        pytest.skip(f"Vitis execution unavailable in current setup: {exc}")
    except subprocess.CalledProcessError as exc:
        pytest.fail(
            f"{failure_prefix}\n"
            f"Command: {exc.cmd}\n"
            f"Return code: {exc.returncode}\n"
            f"Stdout:\n{exc.stdout}\n"
            f"Stderr:\n{exc.stderr}"
        )


@pytest.mark.vitis
@pytest.mark.parametrize("word_bw", [32, 64])
def test_arrayutils_float_roundtrip_vitis(tmp_path: Path, word_bw: int):
    vitis_path = toolchain.find_vitis_path()
    if not vitis_path:
        pytest.skip("Vitis installation not found; skipping arrayutils integration test.")

    length = 19
    data = np.linspace(-2.5, 3.5, length, dtype=np.float32)
    in_words = write_array(data, elem_type=F32, word_bw=word_bw)

    in_words_path = tmp_path / "array_words.txt"
    out_words_path = tmp_path / "array_words_out.txt"
    save_dtype = np.uint32 if word_bw <= 32 else np.uint64
    np.savetxt(in_words_path, in_words.astype(save_dtype), fmt="%u")

    cfg = CodeGenConfig(root_dir=tmp_path, util_dir="include")
    generated_header = gen_array_utils(F32, [word_bw], cfg=cfg)
    copy_streamutils(cfg)

    header_include = generated_header.relative_to(tmp_path).as_posix()
    namespace_name = generated_header.stem
    cpp_src = (
        ROUNDTRIP_CPP_PATH.read_text(encoding="utf-8")
        .replace("__HEADER__", header_include)
        .replace("__NAMESPACE__", namespace_name)
        .replace("__WORD_BW__", str(word_bw))
        .replace("__ARRAY_LEN__", str(length))
        .replace("__NWORDS__", str(in_words.shape[0]))
    )
    (tmp_path / "arrayutils_roundtrip_test.cpp").write_text(cpp_src, encoding="utf-8")
    shutil.copy(ROUNDTRIP_TCL_PATH, tmp_path / "arrayutils_roundtrip_run.tcl")

    _run_vitis_tcl(
        tmp_path / "arrayutils_roundtrip_run.tcl",
        work_dir=tmp_path,
        failure_prefix="Vitis execution failed for arrayutils float roundtrip integration test.",
    )

    out_words = np.loadtxt(out_words_path, dtype=save_dtype)
    out_words = np.asarray(out_words)
    if out_words.ndim == 0:
        out_words = out_words.reshape(1)

    got = read_array(out_words, elem_type=F32, word_bw=word_bw, shape=length)
    got = np.asarray(got, dtype=np.float32)

    assert np.array_equal(out_words, in_words.astype(save_dtype))
    assert np.allclose(got, data, rtol=1e-6, atol=1e-6)


@pytest.mark.vitis
def test_arrayutils_int16_pf2_roundtrip_vitis(tmp_path: Path):
    vitis_path = toolchain.find_vitis_path()
    if not vitis_path:
        pytest.skip("Vitis installation not found; skipping arrayutils integration test.")

    word_bw = 32
    data = np.array([-32768, -12345, -17, -1, 0, 1, 23, 255, 1024, 32767], dtype=np.int16)
    in_words = write_array(data, elem_type=S16, word_bw=word_bw)

    in_words_path = tmp_path / "array_words.txt"
    out_words_path = tmp_path / "array_words_out.txt"
    np.savetxt(in_words_path, in_words.astype(np.uint32), fmt="%u")

    cfg = CodeGenConfig(root_dir=tmp_path, util_dir="include")
    generated_header = gen_array_utils(S16, [word_bw], cfg=cfg)
    copy_streamutils(cfg)

    header_include = generated_header.relative_to(tmp_path).as_posix()
    namespace_name = generated_header.stem
    cpp_src = (
        ROUNDTRIP_CPP_PATH.read_text(encoding="utf-8")
        .replace("__HEADER__", header_include)
        .replace("__NAMESPACE__", namespace_name)
        .replace("__WORD_BW__", str(word_bw))
        .replace("__ARRAY_LEN__", str(data.size))
        .replace("__NWORDS__", str(in_words.shape[0]))
    )
    (tmp_path / "arrayutils_roundtrip_test.cpp").write_text(cpp_src, encoding="utf-8")
    shutil.copy(ROUNDTRIP_TCL_PATH, tmp_path / "arrayutils_roundtrip_run.tcl")

    _run_vitis_tcl(
        tmp_path / "arrayutils_roundtrip_run.tcl",
        work_dir=tmp_path,
        failure_prefix="Vitis execution failed for arrayutils int16 pf=2 roundtrip integration test.",
    )

    out_words = np.loadtxt(out_words_path, dtype=np.uint32)
    out_words = np.asarray(out_words)
    if out_words.ndim == 0:
        out_words = out_words.reshape(1)

    got = read_array(out_words, elem_type=S16, word_bw=word_bw, shape=data.size)
    got = np.asarray(got, dtype=np.int16)

    assert np.array_equal(out_words, in_words.astype(np.uint32))
    assert np.array_equal(got, data)