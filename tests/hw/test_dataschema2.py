from enum import IntEnum

import numpy as np
import pytest

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


def test_enumfield_gen_include_emits_guard_and_members():
    mode_field = EnumField.specialize(enum_type=Mode)
    content = mode_field.gen_include()

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


def test_datalist_gen_include_emits_dependency_includes_and_members():
    content = Packet.gen_include()

    assert "#ifndef PACKET_H" in content
    assert '#include "mode.h"' in content
    assert '#include "complex.h"' in content
    assert "struct Packet {" in content
    assert "ap_uint<16> count;" in content
    assert "float gain;" in content
    assert "Mode mode;" in content
    assert "Complex z;" in content
    assert "#endif // PACKET_H" in content


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