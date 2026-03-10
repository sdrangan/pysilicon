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

	assert "\n};\n\n#endif // COMPLEX_SAMPLE_H" in content
	assert str((tmp_path / "complex_sample.h")) == out_path
