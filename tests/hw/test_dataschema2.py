from enum import IntEnum
from pathlib import Path

import numpy as np
import pytest

from pysilicon.codegen.build import CodeGenConfig
from pysilicon.hw.dataschema2 import DataList, EnumField, FloatField, IntField


class Mode(IntEnum):
    OFF = 0
    ON = 1
    AUTO = 2


class OpCode(IntEnum):
    NOP = 0
    ADD = 1


U16 = IntField.specialize(bitwidth=16, signed=False)
S16 = IntField.specialize(bitwidth=16, signed=True)
F32 = FloatField.specialize(bitwidth=32)
ModeField = EnumField.specialize(enum_type=Mode, default=Mode.AUTO)


class Complex(DataList):
    elements = {
        "real": S16,
        "imag": S16,
    }


class Packet(DataList):
    elements = {
        "count": U16,
        "gain": F32,
        "mode": ModeField,
        "z": Complex,
    }


class DescribedComplex(DataList):
    elements = {
        "real": {
            "schema": F32,
            "description": "Real component",
        },
        "imag": {
            "schema": F32,
            "description": "Imaginary component of the complex sample in Q-format form.",
        },
    }


def test_intfield_specialize_same_args_returns_same_class():
    a = IntField.specialize(bitwidth=16, signed=False)
    b = IntField.specialize(bitwidth=16, signed=False)

    assert a is b


def test_intfield_specialize_include_metadata_affects_cache_key():
    a = IntField.specialize(bitwidth=16, signed=False, include_dir="a")
    b = IntField.specialize(bitwidth=16, signed=False, include_dir="b")

    assert a is not b
    assert a.include_dir == "a"
    assert b.include_dir == "b"


def test_init_value_semantics():
    uint16_type = IntField.specialize(bitwidth=16, signed=False)
    float32_type = FloatField.specialize(bitwidth=32)
    enum_type = EnumField.specialize(enum_type=Mode, default=Mode.ON)

    int_init = uint16_type.init_value()
    float_init = float32_type.init_value()
    enum_init = enum_type.init_value()

    assert isinstance(int_init, np.uint32)
    assert int(int_init) == 0
    assert isinstance(float_init, np.float32)
    assert float(float_init) == pytest.approx(0.0)
    assert enum_init is Mode.ON


def test_enumfield_defaults_follow_enum_type_metadata():
    mode_field = EnumField.specialize(enum_type=Mode)
    opcode_field = EnumField.specialize(enum_type=OpCode)

    assert mode_field.cpp_class_name() == "Mode"
    assert mode_field.resolved_include_filename() == "mode.h"
    assert opcode_field.cpp_class_name() == "OpCode"
    assert opcode_field.resolved_include_filename() == "op_code.h"


def test_enumfield_gen_include_emits_guard_and_members(tmp_path: Path):
    mode_field = EnumField.specialize(enum_type=Mode)
    out_path = mode_field.gen_include(cfg=CodeGenConfig(root_dir=tmp_path))
    content = out_path.read_text(encoding="utf-8")

    assert out_path == tmp_path / "mode.h"
    assert "#ifndef MODE_H" in content
    assert "#define MODE_H" in content
    assert "enum class Mode" in content
    assert "OFF = 0," in content
    assert "ON = 1," in content
    assert "AUTO = 2," in content
    assert "#endif // MODE_H" in content


def test_enumfield_explicit_overrides_win():
    mode_field = EnumField.specialize(
        enum_type=Mode,
        cpp_repr="CustomMode",
        include_filename="custom_mode.h",
    )

    assert mode_field.cpp_class_name() == "CustomMode"
    assert mode_field.resolved_include_filename() == "custom_mode.h"


def test_inline_enum_specialization_keeps_expected_metadata():
    class Instruction(DataList):
        elements = {
            "mode": EnumField.specialize(enum_type=Mode),
        }

    mode_schema = Instruction.elements["mode"]
    assert mode_schema.cpp_class_name() == "Mode"
    assert mode_schema.resolved_include_filename() == "mode.h"


def test_datalist_get_dependencies_only_returns_generated_types():
    deps = Packet.get_dependencies()

    assert deps == [ModeField, Complex]
    assert U16 not in deps
    assert F32 not in deps


def test_datalist_backward_compatible_elements_form_still_works():
    packet = Packet(count=5, gain=2.0)

    assert isinstance(packet.z, Complex)
    assert int(packet.count) == 5
    assert float(packet.gain) == pytest.approx(2.0)


def test_datalist_metadata_form_initializes_and_assigns_normally():
    sample = DescribedComplex(real=1.5, imag=2.5)

    assert isinstance(sample.real, np.float32)
    assert isinstance(sample.imag, np.float32)
    assert float(sample.real) == pytest.approx(1.5)
    assert float(sample.imag) == pytest.approx(2.5)


def test_datalist_serialize_deserialize_roundtrip_word32():
    packet = Packet(count=7, gain=3.5, mode=Mode.ON)
    packet.z.real = -5
    packet.z.imag = 9

    packed = packet.serialize(word_bw=32)
    restored = Packet().deserialize(packed, word_bw=32)

    assert packed.dtype == np.uint32
    assert packed.shape == (4,)
    assert restored.is_close(packet)


def test_datalist_serialize_deserialize_roundtrip_word128():
    packet = Packet(count=11, gain=1.25, mode=Mode.AUTO)
    packet.z.real = -2
    packet.z.imag = 4

    packed = packet.serialize(word_bw=128)
    restored = Packet().deserialize(packed, word_bw=128)

    assert packed.dtype == np.uint64
    assert packed.shape == (1, 2)
    assert restored.is_close(packet)


def test_described_complex_serialize_deserialize_roundtrip():
    sample = DescribedComplex(real=1.5, imag=2.5)

    packed = sample.serialize(word_bw=64)
    restored = DescribedComplex().deserialize(packed, word_bw=64)

    assert packed.dtype == np.uint64
    assert packed.shape == (1,)
    assert restored.is_close(sample)


def test_serialize_rejects_non_positive_word_width():
    with pytest.raises(ValueError, match="word_bw must be positive"):
        Packet().serialize(word_bw=0)


def test_deserialize_rejects_invalid_shape_for_large_word_width():
    with pytest.raises(ValueError, match="packed must be a 2D array-like"):
        Packet().deserialize(np.array([1], dtype=np.uint64), word_bw=128)


def test_write_uint32_file_and_read_uint32_file_roundtrip(tmp_path: Path):
    packet = Packet(count=13, gain=2.75, mode=Mode.ON)
    packet.z.real = -7
    packet.z.imag = 12

    out_path = packet.write_uint32_file(tmp_path / "packet.bin")
    restored = Packet().read_uint32_file(out_path)

    assert out_path == tmp_path / "packet.bin"
    assert out_path.exists()
    assert restored.is_close(packet)


def test_write_uint32_file_creates_parent_directories(tmp_path: Path):
    packet = Packet(count=1)
    out_path = packet.write_uint32_file(tmp_path / "nested" / "dir" / "packet.bin")

    assert out_path.exists()


def test_datalist_element_normalization_accessors_work_for_both_forms():
    assert Complex.get_element_schema("real") is S16
    assert Complex.get_element_description("real") is None
    assert Complex.get_element_definition("real") == {
        "schema": S16,
        "description": None,
    }

    assert DescribedComplex.get_element_schema("real") is F32
    assert DescribedComplex.get_element_description("real") == "Real component"
    assert DescribedComplex.get_element_definition("imag") == {
        "schema": F32,
        "description": "Imaginary component of the complex sample in Q-format form.",
    }


def test_datalist_metadata_validation_rejects_missing_schema():
    class BadList(DataList):
        elements = {
            "x": {"description": "missing schema"},
        }

    with pytest.raises(TypeError, match="must define a 'schema' entry"):
        BadList.get_bitwidth()


def test_datalist_metadata_validation_rejects_unknown_keys():
    class BadList(DataList):
        elements = {
            "x": {"schema": U16, "units": "V"},
        }

    with pytest.raises(TypeError, match="unsupported metadata key"):
        BadList.get_bitwidth()


def test_datalist_metadata_validation_rejects_invalid_schema_value():
    class BadList(DataList):
        elements = {
            "x": {"schema": 123},
        }

    with pytest.raises(TypeError, match="must be a DataSchema subclass"):
        BadList.get_bitwidth()


def test_datalist_get_dependencies_works_with_metadata_form():
    class WithMetadataDeps(DataList):
        elements = {
            "mode": {
                "schema": ModeField,
                "description": "Mode field",
            },
            "nested": {
                "schema": Complex,
                "description": "Nested complex sample",
            },
            "gain": {
                "schema": F32,
                "description": "Local gain",
            },
        }

    assert WithMetadataDeps.get_dependencies() == [ModeField, Complex]


def test_datalist_gen_include_emits_dependency_includes_and_members(tmp_path: Path):
    out_path = Packet.gen_include(cfg=CodeGenConfig(root_dir=tmp_path))
    content = out_path.read_text(encoding="utf-8")

    assert out_path == tmp_path / "packet.h"
    assert '#include "streamutils.h"' in content
    assert "#ifndef PACKET_H" in content
    assert '#include "mode.h"' in content
    assert '#include "complex.h"' in content
    assert "struct Packet {" in content
    assert "ap_uint<16> count;" in content
    assert "float gain;" in content
    assert "Mode mode;" in content
    assert "Complex z;" in content
    assert "static constexpr int bitwidth = 82;" in content
    assert "static ap_uint<bitwidth> pack_to_uint(const Packet& data) {" in content
    assert "static Packet unpack_from_uint(const ap_uint<bitwidth>& packed) {" in content
    assert "#endif // PACKET_H" in content


def test_datalist_gen_include_emits_inline_and_block_comments(tmp_path: Path):
    out_path = DescribedComplex.gen_include(cfg=CodeGenConfig(root_dir=tmp_path))
    content = out_path.read_text(encoding="utf-8")

    assert "float real;  // Real component" in content
    assert "// Imaginary component of the complex sample in Q-format form." in content
    assert "float imag;" in content


def test_datalist_gen_include_emits_read_helpers_when_requested(tmp_path: Path):
    out_path = Packet.gen_include(
        cfg=CodeGenConfig(root_dir=tmp_path),
        word_bw_supported=[32],
    )
    content = out_path.read_text(encoding="utf-8")

    assert "template<int word_bw>" in content
    assert "void write_array(ap_uint<word_bw> x[]) const {" in content
    assert "void write_stream(hls::stream<ap_uint<word_bw>> &s) const {" in content
    assert "void write_axi4_stream(hls::stream<hls::axis<ap_uint<word_bw>, 0, 0, 0>> &s, bool tlast = true) const {" in content
    assert "x[0].range(15, 0) = this->count;" in content
    assert "streamutils::write_axi4_word<32>(s, w, tlast);" in content
    assert "void read_array(const ap_uint<word_bw> x[]) {" in content
    assert "void read_stream(hls::stream<ap_uint<word_bw>> &s) {" in content
    assert "void read_axi4_stream(hls::stream<hls::axis<ap_uint<word_bw>, 0, 0, 0>> &s) {" in content
    assert "this->count = (ap_uint<16>)(x[0].range(15, 0));" in content
    assert "w = s.read().data;" in content


def test_gen_include_rejects_non_positive_word_widths():
    with pytest.raises(ValueError, match="word_bw values must be positive"):
        Packet.gen_include(word_bw_supported=[0])


def test_gen_include_writes_under_cfg_root_and_include_dir(tmp_path: Path):
    class Instruction(DataList):
        include_dir = "isa"
        elements = {
            "count": U16,
        }

    out_path = Instruction.gen_include(cfg=CodeGenConfig(root_dir=tmp_path))
    content = out_path.read_text(encoding="utf-8")

    assert out_path == tmp_path / "isa" / "instruction.h"
    assert out_path.exists()
    assert '#include "../streamutils.h"' in content


def test_gen_include_uses_cfg_util_dir_for_streamutils_include(tmp_path: Path):
    cfg = CodeGenConfig(root_dir=tmp_path, util_dir="common")
    out_path = Packet.gen_include(cfg=cfg)
    content = out_path.read_text(encoding="utf-8")

    assert '#include "common/streamutils.h"' in content


def test_gen_include_overwrites_existing_file(tmp_path: Path):
    out_path = tmp_path / "packet.h"
    out_path.write_text("stale", encoding="utf-8")

    written_path = Packet.gen_include(cfg=CodeGenConfig(root_dir=tmp_path))

    assert written_path == out_path
    assert "stale" not in out_path.read_text(encoding="utf-8")


def test_primitive_field_pack_unpack_helpers_are_empty():
    assert U16.gen_pack() == ""
    assert U16.gen_unpack() == ""


def test_datalist_pack_emits_expected_member_slices():
    content = Packet.gen_pack()

    assert "static ap_uint<bitwidth> pack_to_uint(const Packet& data) {" in content
    assert "ap_uint<bitwidth> res = 0;" in content
    assert "res.range(15, 0) = data.count;" in content
    assert "res.range(47, 16) = streamutils::float_to_uint(data.gain);" in content
    assert "res.range(49, 48) = (ap_uint<2>)(data.mode);" in content
    assert "res.range(81, 50) = Complex::pack_to_uint(data.z);" in content
    assert "return res;" in content


def test_datalist_unpack_emits_expected_member_slices():
    content = Packet.gen_unpack()

    assert "static Packet unpack_from_uint(const ap_uint<bitwidth>& packed) {" in content
    assert "Packet data;" in content
    assert "data.count = (ap_uint<16>)(packed.range(15, 0));" in content
    assert "data.gain = streamutils::uint_to_float((uint32_t)(packed.range(47, 16)));" in content
    assert "data.mode = (Mode)(packed.range(49, 48));" in content
    assert "data.z = Complex::unpack_from_uint(packed.range(81, 50));" in content
    assert "return data;" in content


def test_datalist_gen_write_array_emits_expected_slices():
    content = Packet.gen_write(word_bw=32, dst_type="array")

    assert "template<int word_bw>" in content
    assert "void write_array(ap_uint<word_bw> x[]) const {" in content
    assert "if constexpr (word_bw == 32) {" in content
    assert "x[0] = 0;" in content
    assert "x[0].range(15, 0) = this->count;" in content
    assert "x[1] = streamutils::float_to_uint(this->gain);" in content
    assert "x[2] = 0;" in content
    assert "x[2].range(1, 0) = (ap_uint<2>)(this->mode);" in content
    assert "x[2].range(17, 2) = this->z.real;" in content
    assert "x[3].range(15, 0) = this->z.imag;" in content


def test_datalist_gen_write_stream_flushes_words():
    content = Packet.gen_write(word_bw=32, dst_type="stream")

    assert "void write_stream(hls::stream<ap_uint<word_bw>> &s) const {" in content
    assert "ap_uint<32> w = 0;" in content
    assert "w.range(15, 0) = this->count;" in content
    assert "s.write(w);" in content
    assert "w = 0;" in content
    assert "w = streamutils::float_to_uint(this->gain);" in content
    assert "w.range(1, 0) = (ap_uint<2>)(this->mode);" in content
    assert "w.range(17, 2) = this->z.real;" in content
    assert "w.range(15, 0) = this->z.imag;" in content


def test_datalist_gen_write_axi4_stream_uses_tlast_on_final_word():
    content = Packet.gen_write(word_bw=32, dst_type="axi4_stream")

    assert "void write_axi4_stream(hls::stream<hls::axis<ap_uint<word_bw>, 0, 0, 0>> &s, bool tlast = true) const {" in content
    assert "ap_uint<32> w = 0;" in content
    assert "streamutils::write_axi4_word<32>(s, w, false);" in content
    assert "streamutils::write_axi4_word<32>(s, w, tlast);" in content


def test_gen_write_requires_word_width_configuration():
    with pytest.raises(ValueError, match="word_bw must be provided"):
        Packet.gen_write()


def test_datalist_gen_read_array_emits_expected_slices():
    content = Packet.gen_read(word_bw=32, src_type="array")

    assert "template<int word_bw>" in content
    assert "void read_array(const ap_uint<word_bw> x[]) {" in content
    assert "if constexpr (word_bw == 32) {" in content
    assert "this->count = (ap_uint<16>)(x[0].range(15, 0));" in content
    assert "this->gain = streamutils::uint_to_float((uint32_t)(x[1]));" in content
    assert "this->mode = (Mode)(x[2].range(1, 0));" in content
    assert "this->z.real = (ap_int<16>)(x[2].range(17, 2));" in content
    assert "this->z.imag = (ap_int<16>)(x[3].range(15, 0));" in content


def test_datalist_gen_read_stream_emits_word_reads_at_boundaries():
    content = Packet.gen_read(word_bw=32, src_type="stream")

    assert "void read_stream(hls::stream<ap_uint<word_bw>> &s) {" in content
    assert "ap_uint<32> w = 0;" in content
    assert "w = s.read();" in content
    assert "this->count = (ap_uint<16>)(w.range(15, 0));" in content
    assert "this->gain = streamutils::uint_to_float((uint32_t)(w));" in content
    assert "this->mode = (Mode)(w.range(1, 0));" in content


def test_datalist_gen_read_axi4_stream_uses_data_field():
    content = Packet.gen_read(word_bw=32, src_type="axi4_stream")

    assert "void read_axi4_stream(hls::stream<hls::axis<ap_uint<word_bw>, 0, 0, 0>> &s) {" in content
    assert "ap_uint<32> w = 0;" in content
    assert "w = s.read().data;" in content
    assert "this->gain = streamutils::uint_to_float((uint32_t)(w));" in content


def test_gen_read_requires_word_width_configuration():
    with pytest.raises(ValueError, match="word_bw must be provided"):
        Packet.gen_read()


def test_datalist_initialization_nested_snapshot_and_types():
    packet = Packet()
    packet_ref = Packet()

    assert packet.is_close(packet_ref)
    assert isinstance(packet.count, np.uint32)
    assert isinstance(packet.gain, np.float32)
    assert packet.mode is Mode.AUTO
    assert isinstance(packet.z, Complex)
    assert isinstance(packet.z.real, np.int32)
    assert isinstance(packet.z.imag, np.int32)


def test_assignment_conversion_for_scalar_and_enum_fields():
    packet = Packet()

    packet.mode = 1
    packet.gain = 3
    packet.z.real = -5
    packet.count = 7

    assert packet.mode is Mode.ON
    assert isinstance(packet.gain, np.float32)
    assert float(packet.gain) == pytest.approx(3.0)
    assert isinstance(packet.z.real, np.int32)
    assert int(packet.z.real) == -5
    assert isinstance(packet.count, np.uint32)
    assert int(packet.count) == 7


def test_default_include_filename_uses_snake_case_class_name():
    class MyPacketHeader(DataList):
        elements = {}

    assert MyPacketHeader.default_include_filename() == "my_packet_header.h"


def test_root_include_dir_resolves_to_filename_only():
    class Instruction(DataList):
        elements = {}

    assert Instruction.include_path() == "instruction.h"


def test_non_root_include_dir_resolves_to_dir_filename():
    class Instruction(DataList):
        include_dir = "isa"
        elements = {}

    assert Instruction.include_path() == "isa/instruction.h"


def test_relative_include_path_uses_current_header_directory():
    class CommonMode(DataList):
        include_dir = "common"
        elements = {}

    class Instruction(DataList):
        include_dir = "isa"
        elements = {}

    assert Instruction.relative_include_path_to(CommonMode) == "../common/common_mode.h"


def test_invalid_specialize_kwargs_are_rejected():
    with pytest.raises(TypeError, match="Unknown specialization keyword"):
        IntField.specialize(bitwidth=16, include_dri="isa")


@pytest.mark.parametrize("field_type", [IntField.specialize(bitwidth=16), FloatField.specialize(bitwidth=32)])
def test_primitive_field_gen_include_raises(field_type):
    with pytest.raises(ValueError, match="does not support standalone include generation"):
        field_type.gen_include()