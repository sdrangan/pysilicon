from enum import IntEnum

import numpy as np
import pytest

from pysilicon.hw.dataschema import DataArray, DataList, EnumField, FloatField, IntField


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


def test_gen_include_prefers_class_name_for_datalist_subclass(tmp_path):
	class CmdHdr(DataList):
		def __init__(self, name=None):
			super().__init__(name=name)
			self.add_elem(IntField(name="opcode", bitwidth=8, signed=False))

	schema = CmdHdr(name="instance_specific_name")
	out_path = schema.gen_include(include_dir=tmp_path, word_bw_supported=[32])

	assert out_path.endswith("cmd_hdr.h")
	assert (tmp_path / "cmd_hdr.h").exists()
	assert not (tmp_path / "instance_specific_name.h").exists()


def test_gen_include_prefers_class_name_for_dataarray_subclass(tmp_path):
	class CoeffArray(DataArray):
		def __init__(self, name=None):
			super().__init__(
				name=name,
				element_type=FloatField(name="coeff", bitwidth=32),
				max_shape=[4],
				static=True,
			)

	schema = CoeffArray(name="coeffs")
	out_path = schema.gen_include(include_dir=tmp_path, word_bw_supported=[32])

	assert out_path.endswith("coeff_array.h")
	assert (tmp_path / "coeff_array.h").exists()
	assert not (tmp_path / "coeffs.h").exists()


def test_gen_include_raw_datalist_raises(tmp_path):
	schema = DataList(name="cmd_hdr")
	schema.add_elem(IntField(name="opcode", bitwidth=8, signed=False))

	with pytest.raises(ValueError, match="does not support standalone include generation"):
		schema.gen_include(include_dir=tmp_path, word_bw_supported=[32])


def test_gen_include_raw_dataarray_raises(tmp_path):
	arr = DataArray(
		name="coeffs",
		element_type=IntField(name="x", bitwidth=8, signed=False),
		max_shape=[8],
		static=True,
	)

	with pytest.raises(ValueError, match="does not support standalone include generation"):
		arr.gen_include(include_dir=tmp_path, word_bw_supported=[32])


def test_gen_include_field_raises(tmp_path):
	field = IntField(name="opcode", bitwidth=8, signed=False)

	with pytest.raises(ValueError, match="does not support standalone include generation"):
		field.gen_include(include_dir=tmp_path, word_bw_supported=[32])


def test_gen_include_closes_class_brace(tmp_path):
	class ComplexSample(DataList):
		def __init__(self, name=None):
			super().__init__(name=name)
			self.add_elem(FloatField(name="r", bitwidth=32))
			self.add_elem(FloatField(name="i", bitwidth=32))

	schema = ComplexSample(name=None)
	out_path = schema.gen_include(include_dir=tmp_path, word_bw_supported=[32, 64])
	content = (tmp_path / "complex_sample.h").read_text(encoding="utf-8")

	assert "\n};\n" in content
	assert "void dump_json(std::ostream& os, int indent = 2, int level = 0) const" in content
	assert "void load_json(std::istream& is)" in content
	assert "void dump_json_file(const char* file_path, int indent = 2) const" in content
	assert "void load_json_file(const char* file_path)" in content
	assert "streamutils::json_parse_string" in content
	assert "streamutils::json_parse_number" in content
	assert "template<int word_bw>" in content
	assert "static constexpr int nwords()" in content
	assert "if constexpr (word_bw == 32)" in content
	assert "return 2;" in content
	assert "else if constexpr (word_bw == 64)" in content
	assert "return 1;" in content
	assert "static std::string _json_parse_string" not in content
	assert "static double _json_parse_number" not in content
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


def test_write_uint32_file_matches_serialize(tmp_path):
	packet = SimplePacket(name="pkt")
	packet.count = -7
	packet.gain = 1.5
	packet.mode = Mode.AUTO

	out_path = tmp_path / "pkt_words.bin"
	written_path = packet.write_uint32_file(out_path)

	assert written_path == out_path
	assert out_path.exists()

	file_words = np.fromfile(out_path, dtype="<u4")
	expected_words = np.asarray(packet.serialize(word_bw=32), dtype="<u4")

	assert np.array_equal(file_words, expected_words)


def test_read_uint32_file_roundtrip(tmp_path):
	packet = SimplePacket(name="pkt")
	packet.count = -11
	packet.gain = 2.25
	packet.mode = Mode.ON

	out_path = tmp_path / "pkt_words.bin"
	packet.write_uint32_file(out_path)

	restored = SimplePacket(name="restored")
	ret = restored.read_uint32_file(out_path)

	assert ret is restored
	assert restored.is_close(packet)

	restored2 = SimplePacket(name="restored2")
	ret2 = restored2.read_uint32_File(out_path)

	assert ret2 is restored2
	assert restored2.is_close(packet)


def test_dataarray_write_uint32_file_nwrite(tmp_path):
	arr = DataArray(
		name="arr",
		element_type=IntField(name="x", bitwidth=8, signed=False),
		max_shape=[8],
		static=True,
	)
	arr.val = np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8)

	out_path = tmp_path / "arr_nwrite.bin"
	arr.write_uint32_file(out_path, nwrite=4)

	file_words = np.fromfile(out_path, dtype="<u4")

	ref = DataArray(
		name="arr_ref",
		element_type=IntField(name="x", bitwidth=8, signed=False),
		max_shape=[4],
		static=True,
	)
	ref.val = np.asarray([1, 2, 3, 4], dtype=np.uint8)
	expected_words = np.asarray(ref.serialize(word_bw=32), dtype="<u4")

	assert np.array_equal(file_words, expected_words)


def test_dataarray_write_uint32_file_write_slice(tmp_path):
	arr = DataArray(
		name="arr2d",
		element_type=IntField(name="x", bitwidth=16, signed=False),
		max_shape=[4, 4],
		static=True,
	)
	arr.val = np.arange(16, dtype=np.uint16).reshape(4, 4)

	out_path = tmp_path / "arr_slice.bin"
	arr.write_uint32_file(out_path, write_slice=np.s_[:2, 1:3])

	file_words = np.fromfile(out_path, dtype="<u4")

	ref = DataArray(
		name="arr2d_ref",
		element_type=IntField(name="x", bitwidth=16, signed=False),
		max_shape=[2, 2],
		static=True,
	)
	ref.val = np.asarray(arr.val[:2, 1:3], dtype=np.uint16)
	expected_words = np.asarray(ref.serialize(word_bw=32), dtype="<u4")

	assert np.array_equal(file_words, expected_words)


def test_dataarray_subclass_gen_include_has_nwords_len_for_dynamic(tmp_path):
	class DynArr(DataArray):
		def __init__(self, name=None):
			super().__init__(
				name=name,
				element_type=IntField(name="x", bitwidth=8, signed=False),
				max_shape=[16],
				static=False,
			)

	arr = DynArr(name="dyn_arr")
	out_path = arr.gen_include(include_dir=tmp_path, word_bw_supported=[32, 64])
	content = (tmp_path / "dyn_arr.h").read_text(encoding="utf-8")

	assert out_path.endswith("dyn_arr.h")
	assert "static int nwords_len(int n0=1)" in content
	assert "if constexpr (word_bw == 32)" in content
	assert "const int n0_eff = (n0 < 0) ? 0 : ((n0 > 16) ? 16 : n0);" in content
	assert "return (n_total + 4 - 1) / 4;" in content


def test_gen_include_dump_json_nested_calls(tmp_path):
	schema = OuterPacket(name="outer")
	out_path = schema.gen_include(include_dir=tmp_path, word_bw_supported=[32])
	content = (tmp_path / "outer_packet.h").read_text(encoding="utf-8")

	assert out_path.endswith("outer_packet.h")
	assert "this->sample.dump_json(os, step, level + 1);" in content
	assert "this->sample.load_json(json_text, pos);" in content
