from enum import IntEnum

import numpy as np
import pytest

from pysilicon.hw.dataschema import DataList, EnumField, FloatField, IntField


class Mode(IntEnum):
	OFF = 0
	ON = 1
	AUTO = 2


class SimplePacket(DataList):
	def __init__(self, name=None):
		super().__init__(name=name, description="Simple test packet")
		self.add_elem(IntField(name="count", bitwidth=16, signed=True))
		self.add_elem(FloatField(name="gain", bitwidth=32))
		self.add_elem(EnumField(name="mode", enum_type=Mode))


class InnerPacket(DataList):
	def __init__(self, name=None):
		super().__init__(name=name, description="Nested inner packet")
		self.add_elem(IntField(name="i", bitwidth=16, signed=True))
		self.add_elem(FloatField(name="q", bitwidth=32))


class OuterPacket(DataList):
	def __init__(self, name=None):
		super().__init__(name=name, description="Nested outer packet")
		self.add_elem(IntField(name="seq", bitwidth=8, signed=False))
		self.add_elem(InnerPacket(name="sample"))
		self.add_elem(EnumField(name="mode", enum_type=Mode))


@pytest.mark.parametrize("word_bw", [32, 64])
def test_dataschema_serialize_deserialize_roundtrip(word_bw):
	packet = SimplePacket(name="pkt")
	packet.count = -7
	packet.gain = 1.5
	packet.mode = Mode.AUTO

	packed = packet.serialize(word_bw=word_bw)

	restored = SimplePacket(name="restored")
	restored.deserialize(packed, word_bw=word_bw)

	assert restored.is_close(packet)


@pytest.mark.parametrize("word_bw", [32, 64])
def test_dataschema_nested_datalist_roundtrip(word_bw):
	packet = OuterPacket(name="outer")
	packet.seq = 17
	packet.sample.i = -23
	packet.sample.q = 0.625
	packet.mode = Mode.ON

	packed = packet.serialize(word_bw=word_bw)

	restored = OuterPacket(name="restored_outer")
	restored.deserialize(packed, word_bw=word_bw)

	assert restored.is_close(packet)


def test_gen_include_default_filename_snake_case(tmp_path):
	class CmdHdr(DataList):
		def __init__(self, name=None):
			super().__init__(name=name)
			self.add_elem(IntField(name="opcode", bitwidth=8, signed=False))

	schema = CmdHdr(name=None)
	out_path = schema.gen_include(include_dir=tmp_path, word_bw_supported=[32])

	assert out_path.endswith("cmd_hdr.h")
	assert (tmp_path / "cmd_hdr.h").exists()


def test_gen_include_closes_class_brace(tmp_path):
	class ComplexSample(DataList):
		def __init__(self, name=None):
			super().__init__(name=name)
			self.add_elem(FloatField(name="r", bitwidth=32))
			self.add_elem(FloatField(name="i", bitwidth=32))

	schema = ComplexSample(name=None)
	out_path = schema.gen_include(include_dir=tmp_path, word_bw_supported=[32])
	content = (tmp_path / "complex_sample.h").read_text(encoding="utf-8")

	assert "\n};\n" in content
	assert "void dump_json(std::ostream& os, int indent = 2) const" in content
	assert "void load_json(std::istream& is)" in content
	assert "void dump_json_file(const char* file_path, int indent = 2) const" in content
	assert "void load_json_file(const char* file_path)" in content
	assert "static std::string _json_parse_string(const std::string& s, size_t& pos)" in content
	assert "static double _json_parse_number(const std::string& s, size_t& pos)" in content
	assert "NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE" not in content
	assert "\n#endif // COMPLEX_SAMPLE_H" in content
	assert str((tmp_path / "complex_sample.h")) == out_path


def test_dataschema_from_dict_roundtrip():
	packet = SimplePacket(name="pkt")
	packet.count = -11
	packet.gain = 2.25
	packet.mode = Mode.ON

	payload = packet.to_dict()

	restored = SimplePacket(name="pkt")
	restored.from_dict(payload)

	assert restored.is_close(packet)


def test_dataschema_to_json_from_json_roundtrip(tmp_path):
	packet = OuterPacket(name="outer")
	packet.seq = 31
	packet.sample.i = -17
	packet.sample.q = 0.5
	packet.mode = Mode.AUTO

	json_path = tmp_path / "outer_packet.json"
	json_str = packet.to_json(file_path=json_path)

	restored = OuterPacket(name="outer")
	restored.from_json(json_str)
	assert restored.is_close(packet)

	restored2 = OuterPacket(name="outer")
	restored2.from_json(json_path)
	assert restored2.is_close(packet)
