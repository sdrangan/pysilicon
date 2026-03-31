import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pysilicon.codegen.build import CodeGenConfig
from pysilicon.codegen.streamutils import copy_streamutils
from pysilicon.hw.dataschema import DataArray, DataList, FloatField, IntField
from pysilicon.xilinxutils import toolchain


S16 = IntField.specialize(bitwidth=16, signed=True)
U8 = IntField.specialize(bitwidth=8, signed=False)
S13 = IntField.specialize(bitwidth=13, signed=True)
F32 = FloatField.specialize(bitwidth=32)


class DemoPacket(DataList):
    elements = {
        "count": {
            "schema": S16,
            "description": "Signed sample count",
        },
        "gain": {
            "schema": F32,
            "description": "Linear gain value",
        },
        "mode": {
            "schema": U8,
            "description": "Unsigned mode selector",
        },
    }


class NestedPacket(DataList):
    elements = {
        "seq": {
            "schema": U8,
            "description": "Sequence number",
        },
        "sample": {
            "schema": DemoPacket,
            "description": "Embedded sample payload",
        },
        "tag": {
            "schema": U8,
            "description": "Packet tag",
        },
    }


class SampData(DataArray):
    element_type = S13
    max_shape = (16,)
    static = True


class DynSampData(DataArray):
    element_type = S13
    max_shape = (16,)
    static = False


TEST_DIR = Path(__file__).parent
RESOURCE_DIR = TEST_DIR / "resources"
SERIALIZE_CPP_PATH = RESOURCE_DIR / "serialize_test.cpp"
SERIALIZE_TCL_PATH = RESOURCE_DIR / "serialize_run.tcl"
DESERIALIZE_CPP_PATH = RESOURCE_DIR / "deserialize_test.cpp"
DESERIALIZE_TCL_PATH = RESOURCE_DIR / "deserialize_run.tcl"
UINT32_FILE_READ_CPP_PATH = RESOURCE_DIR / "uint32_file_read_test.cpp"
UINT32_FILE_READ_TCL_PATH = RESOURCE_DIR / "uint32_file_read_run.tcl"


def _packet_payload(packet_type: type[DataList] | type[DataArray]) -> list[int] | dict[str, object]:
    if packet_type in {SampData, DynSampData}:
        return [
            -2048, -1024, -17, -1,
            0, 1, 7, 63,
            127, 255, 511, 1023,
            1535, 1792, 2046, 2047,
        ]

    if packet_type is DemoPacket:
        return {
            "count": -21,
            "gain": 1.75,
            "mode": 5,
        }

    if packet_type is NestedPacket:
        return {
            "seq": 19,
            "sample": {
                "count": -7,
                "gain": 0.625,
                "mode": 3,
            },
            "tag": 201,
        }

    raise ValueError(f"Unsupported packet_type: {packet_type}")


def _make_packet(packet_type: type[DataList] | type[DataArray]):
    return packet_type().from_dict(_packet_payload(packet_type))


def _rw_args(packet_or_type: object) -> str:
    packet_type = packet_or_type if isinstance(packet_or_type, type) else packet_or_type.__class__
    if issubclass(packet_type, DataArray) and not packet_type.static:
        shape_args = ", ".join(str(int(dim)) for dim in tuple(packet_type.max_shape))
        return f", {shape_args}" if shape_args else ""
    return ""


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


def _generate_include_tree(
    schema_type: type[DataList] | type[DataArray],
    cfg: CodeGenConfig,
    word_bw: int,
    seen: set[type[object]] | None = None,
) -> None:
    if seen is None:
        seen = set()

    if schema_type in seen:
        return

    for dependency in schema_type.get_dependencies():
        _generate_include_tree(dependency, cfg=cfg, word_bw=word_bw, seen=seen)

    schema_type.gen_include(cfg=cfg, word_bw_supported=[word_bw])
    seen.add(schema_type)


@pytest.mark.vitis
@pytest.mark.parametrize("packet_type", [SampData, DemoPacket, NestedPacket])
@pytest.mark.parametrize("word_bw", [32, 64])
def test_dataschema2_python_to_vitis_serialization(
    tmp_path: Path,
    packet_type: type[DataList] | type[DataArray],
    word_bw: int,
):
    vitis_path = toolchain.find_vitis_path()
    if not vitis_path:
        pytest.skip("Vitis installation not found; skipping dataschema2 integration test.")

    words_path = tmp_path / "packet_words.txt"
    output_json_path = tmp_path / "packet_out.json"

    packet = _make_packet(packet_type)
    packed = packet.serialize(word_bw=word_bw)
    save_dtype = np.uint32 if word_bw <= 32 else np.uint64
    np.savetxt(words_path, packed.astype(save_dtype), fmt="%u")

    cfg = CodeGenConfig(root_dir=tmp_path)
    _generate_include_tree(packet_type, cfg=cfg, word_bw=word_bw)
    copy_streamutils(cfg)

    cpp_src = (
        SERIALIZE_CPP_PATH.read_text(encoding="utf-8")
        .replace("__HEADER__", packet_type.resolved_tb_include_filename())
        .replace("__EXTRA_INCLUDES__", "")
        .replace("__PACKET_CLASS__", packet_type.__name__)
        .replace("__WORD_BW__", str(word_bw))
        .replace("__RW_ARGS__", _rw_args(packet_type))
    )
    (tmp_path / "serialize_test.cpp").write_text(cpp_src, encoding="utf-8")
    shutil.copy(SERIALIZE_TCL_PATH, tmp_path / "serialize_run.tcl")

    try:
        toolchain.run_vitis_hls(tmp_path / "serialize_run.tcl", work_dir=tmp_path)
    except RuntimeError as exc:
        pytest.skip(f"Vitis execution unavailable in current setup: {exc}")
    except subprocess.CalledProcessError as exc:
        pytest.fail(
            "Vitis execution failed for dataschema2 serialization integration test.\n"
            f"Command: {exc.cmd}\n"
            f"Return code: {exc.returncode}\n"
            f"Stdout:\n{exc.stdout}\n"
            f"Stderr:\n{exc.stderr}"
        )

    got_packet = packet_type().from_json(output_json_path)
    assert got_packet.is_close(packet, rel_tol=1e-6, abs_tol=1e-6)


@pytest.mark.vitis
@pytest.mark.parametrize("packet_type", [SampData, DemoPacket, NestedPacket])
@pytest.mark.parametrize("word_bw", [32, 64])
def test_dataschema2_vitis_to_python_serialization(
    tmp_path: Path,
    packet_type: type[DataList] | type[DataArray],
    word_bw: int,
):
    vitis_path = toolchain.find_vitis_path()
    if not vitis_path:
        pytest.skip("Vitis installation not found; skipping dataschema2 integration test.")

    input_json_path = tmp_path / "packet_src.json"
    words_path = tmp_path / "packet_from_vitis_words.txt"

    ref_packet = _make_packet(packet_type)
    ref_packet.to_json(file_path=input_json_path, indent=2)

    cfg = CodeGenConfig(root_dir=tmp_path)
    _generate_include_tree(packet_type, cfg=cfg, word_bw=word_bw)
    copy_streamutils(cfg)

    cpp_src = (
        DESERIALIZE_CPP_PATH.read_text(encoding="utf-8")
        .replace("__HEADER__", packet_type.resolved_tb_include_filename())
        .replace("__EXTRA_INCLUDES__", "")
        .replace("__PACKET_CLASS__", packet_type.__name__)
        .replace("__WORD_BW__", str(word_bw))
        .replace("__NWORDS__", str(packet_type.nwords_per_inst(word_bw=word_bw)))
        .replace("__RW_ARGS__", _rw_args(packet_type))
    )
    (tmp_path / "deserialize_test.cpp").write_text(cpp_src, encoding="utf-8")
    shutil.copy(DESERIALIZE_TCL_PATH, tmp_path / "deserialize_run.tcl")

    _run_vitis_tcl(
        tmp_path / "deserialize_run.tcl",
        work_dir=tmp_path,
        failure_prefix="Vitis execution failed for dataschema2 reverse serialization integration test.",
    )

    load_dtype = np.uint32 if word_bw <= 32 else np.uint64
    packed = np.loadtxt(words_path, dtype=load_dtype)
    packed = np.asarray(packed)
    if packed.ndim == 0:
        packed = packed.reshape(1)

    got_packet = packet_type().deserialize(packed, word_bw=word_bw)
    assert got_packet.is_close(ref_packet, rel_tol=1e-6, abs_tol=1e-6)


@pytest.mark.vitis
def test_streamutils_read_uint32_file_loopback(tmp_path: Path):
    vitis_path = toolchain.find_vitis_path()
    if not vitis_path:
        pytest.skip("Vitis installation not found; skipping uint32 file loopback test.")

    packet = _make_packet(DemoPacket)
    in_bin_path = tmp_path / "packet_words.bin"
    out_json_path = tmp_path / "packet_out.json"
    packet.write_uint32_file(in_bin_path)

    cfg = CodeGenConfig(root_dir=tmp_path)
    _generate_include_tree(DemoPacket, cfg=cfg, word_bw=32)
    copy_streamutils(cfg)

    cpp_src = (
        UINT32_FILE_READ_CPP_PATH.read_text(encoding="utf-8")
        .replace("__HEADER__", DemoPacket.resolved_tb_include_filename())
        .replace("__EXTRA_INCLUDES__", "")
        .replace("__PACKET_CLASS__", DemoPacket.__name__)
        .replace("__READ_CALL__", 'streamutils::read_uint32_file(pkt, in_bin_path)')
    )
    (tmp_path / "uint32_file_read_test.cpp").write_text(cpp_src, encoding="utf-8")
    shutil.copy(UINT32_FILE_READ_TCL_PATH, tmp_path / "uint32_file_read_run.tcl")

    _run_vitis_tcl(
        tmp_path / "uint32_file_read_run.tcl",
        work_dir=tmp_path,
        failure_prefix="Vitis execution failed for uint32 file loopback test.",
    )

    got_packet = DemoPacket().from_json(out_json_path)
    assert got_packet.is_close(packet, rel_tol=1e-6, abs_tol=1e-6)


@pytest.mark.vitis
def test_streamutils_read_uint32_file_len_loopback(tmp_path: Path):
    vitis_path = toolchain.find_vitis_path()
    if not vitis_path:
        pytest.skip("Vitis installation not found; skipping uint32 file loopback test.")

    nread = 6
    packet = _make_packet(DynSampData)
    in_bin_path = tmp_path / "packet_words.bin"
    out_json_path = tmp_path / "packet_out.json"
    packet.write_uint32_file(in_bin_path, nwrite=nread)

    cfg = CodeGenConfig(root_dir=tmp_path)
    _generate_include_tree(DynSampData, cfg=cfg, word_bw=32)
    copy_streamutils(cfg)

    cpp_src = (
        UINT32_FILE_READ_CPP_PATH.read_text(encoding="utf-8")
        .replace("__HEADER__", DynSampData.resolved_tb_include_filename())
        .replace("__EXTRA_INCLUDES__", "")
        .replace("__PACKET_CLASS__", DynSampData.__name__)
        .replace("__READ_CALL__", f'streamutils::read_uint32_file_len(pkt, in_bin_path, {nread})')
    )
    (tmp_path / "uint32_file_read_test.cpp").write_text(cpp_src, encoding="utf-8")
    shutil.copy(UINT32_FILE_READ_TCL_PATH, tmp_path / "uint32_file_read_run.tcl")

    _run_vitis_tcl(
        tmp_path / "uint32_file_read_run.tcl",
        work_dir=tmp_path,
        failure_prefix="Vitis execution failed for uint32 file loopback len test.",
    )

    expected = DynSampData()
    expected.val = np.zeros(DynSampData.max_shape[0], dtype=np.int64)
    expected.val[:nread] = np.asarray(packet.val[:nread], dtype=np.int64)

    got_packet = DynSampData().from_json(out_json_path)
    assert got_packet.is_close(expected, rel_tol=1e-6, abs_tol=1e-6)