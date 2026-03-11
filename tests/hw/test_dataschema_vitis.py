import json
import subprocess
import shutil
from pathlib import Path

import numpy as np
import pytest

from pysilicon.codegen.streamutils import copy_streamutils
from pysilicon.hw.dataschema import DataList, FloatField, IntField
from pysilicon.xilinxutils import toolchain

TEST_DIR = Path(__file__).parent
RESOURCE_DIR = TEST_DIR / "resources"


class DemoPacket(DataList):
	def __init__(self, name=None):
		super().__init__(name=name)
		self.add_elem(IntField(name="count", bitwidth=16, signed=True))
		self.add_elem(FloatField(name="gain", bitwidth=32))
		self.add_elem(IntField(name="mode", bitwidth=8, signed=False))


class NestedPacket(DataList):
	def __init__(self, name=None):
		super().__init__(name=name)
		self.add_elem(IntField(name="seq", bitwidth=8, signed=False))
		self.add_elem(DemoPacket(name="sample"))
		self.add_elem(IntField(name="tag", bitwidth=8, signed=False))


def _packet_payload(packet_type):
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


def _packet_name(packet_type):
	if packet_type is DemoPacket:
		return "demo_packet"
	if packet_type is NestedPacket:
		return "nested_packet"
	raise ValueError(f"Unsupported packet_type: {packet_type}")


def _extra_includes(packet_type):
	if packet_type is DemoPacket:
		return ""
	if packet_type is NestedPacket:
		return '#include "demo_packet.h"\n'
	raise ValueError(f"Unsupported packet_type: {packet_type}")

@pytest.mark.vitis
@pytest.mark.parametrize("packet_type", [DemoPacket, NestedPacket])
@pytest.mark.parametrize("word_bw", [32, 64])
def test_python_to_vitis_serialization(tmp_path, packet_type, word_bw):
	vitis_path = toolchain.find_vitis_path()
	if not vitis_path:
		pytest.skip("Vitis installation not found; skipping integration test.")

	input_json_path = tmp_path / "packet_in.json"
	words_path = tmp_path / "packet_words.txt"
	output_json_path = tmp_path / "packet_out.json"

	input_payload = _packet_payload(packet_type)
	input_json_path.write_text(json.dumps(input_payload, indent=2), encoding="utf-8")

	packet_name = _packet_name(packet_type)

	# Python side: create packet, load fields from JSON, serialize to words, and emit include.
	packet = packet_type(name=packet_name)
	packet.from_json(input_json_path)

	packed = packet.serialize(word_bw=word_bw)
	if word_bw <= 32:
		np.savetxt(words_path, packed.astype(np.uint32), fmt="%u")
	else:
		np.savetxt(words_path, packed.astype(np.uint64), fmt="%u")

	include_path = packet.gen_include(word_bw_supported=[word_bw], include_dir=tmp_path)
	assert Path(include_path).name == f"{packet_name}.h"

	if packet_type is NestedPacket:
		demo_include = DemoPacket(name="demo_packet").gen_include(
			word_bw_supported=[word_bw],
			include_dir=tmp_path,
		)
		assert Path(demo_include).name == "demo_packet.h"

	copy_streamutils(dst_path=tmp_path)

	cpp_template = (RESOURCE_DIR / "serialize_test.cpp").read_text(encoding="utf-8")
	cpp_src = (
		cpp_template
		.replace("__HEADER__", f"{packet_name}.h")
		.replace("__EXTRA_INCLUDES__", _extra_includes(packet_type))
		.replace("__PACKET_CLASS__", packet_type.__name__)
		.replace("__WORD_BW__", str(word_bw))
	)
	(tmp_path / "serialize_test.cpp").write_text(cpp_src, encoding="utf-8")

	shutil.copy(RESOURCE_DIR / "serialize_run.tcl", tmp_path / "serialize_run.tcl")
	tcl_src = tmp_path / "serialize_run.tcl"

	try:
		toolchain.run_vitis_hls(tcl_src, work_dir=tmp_path)
	except RuntimeError as exc:
		pytest.skip(f"Vitis execution unavailable in current setup: {exc}")
	except subprocess.CalledProcessError as exc:
		pytest.fail(
			"Vitis execution failed for serialization integration test.\n"
			f"Command: {exc.cmd}\n"
			f"Return code: {exc.returncode}\n"
			f"Stdout:\n{exc.stdout}\n"
			f"Stderr:\n{exc.stderr}"
		)

	got_packet = packet_type(name=packet_name)
	got_packet.from_json(output_json_path)

	ref_packet = packet_type(name=packet_name)
	ref_packet.from_json(input_json_path)

	assert got_packet.is_close(ref_packet, rel_tol=1e-6, abs_tol=1e-6)

@pytest.mark.vitis
@pytest.mark.parametrize("packet_type", [DemoPacket, NestedPacket])
@pytest.mark.parametrize("word_bw", [32, 64])
def test_vitis_to_python_serialization(tmp_path, packet_type, word_bw):
	vitis_path = toolchain.find_vitis_path()
	if not vitis_path:
		pytest.skip("Vitis installation not found; skipping integration test.")

	input_json_path = tmp_path / "packet_src.json"
	words_path = tmp_path / "packet_from_vitis_words.txt"

	packet_name = _packet_name(packet_type)
	payload = _packet_payload(packet_type)

	# Python side: create packet, populate values, and export JSON.
	ref_packet = packet_type(name=packet_name)
	ref_packet.from_dict(payload)
	exported = ref_packet.to_dict()
	if packet_name in exported:
		exported = exported[packet_name]
	exported = ref_packet._to_jsonable(exported)
	input_json_path.write_text(json.dumps(exported, indent=2), encoding="utf-8")

	include_path = ref_packet.gen_include(word_bw_supported=[word_bw], include_dir=tmp_path)
	assert Path(include_path).name == f"{packet_name}.h"

	if packet_type is NestedPacket:
		demo_include = DemoPacket(name="demo_packet").gen_include(
			word_bw_supported=[word_bw],
			include_dir=tmp_path,
		)
		assert Path(demo_include).name == "demo_packet.h"

	copy_streamutils(dst_path=tmp_path)

	nwords = ref_packet.nwords_per_inst(word_bw=word_bw)
	cpp_template = (RESOURCE_DIR / "deserialize_test.cpp").read_text(encoding="utf-8")
	cpp_src = (
		cpp_template
		.replace("__HEADER__", f"{packet_name}.h")
		.replace("__EXTRA_INCLUDES__", _extra_includes(packet_type))
		.replace("__PACKET_CLASS__", packet_type.__name__)
		.replace("__WORD_BW__", str(word_bw))
		.replace("__NWORDS__", str(nwords))
	)
	(tmp_path / "deserialize_test.cpp").write_text(cpp_src, encoding="utf-8")

	shutil.copy(RESOURCE_DIR / "deserialize_run.tcl", tmp_path / "deserialize_run.tcl")
	tcl_src = tmp_path / "deserialize_run.tcl"

	try:
		toolchain.run_vitis_hls(tcl_src, work_dir=tmp_path)
	except RuntimeError as exc:
		pytest.skip(f"Vitis execution unavailable in current setup: {exc}")
	except subprocess.CalledProcessError as exc:
		pytest.fail(
			"Vitis execution failed for reverse serialization integration test.\n"
			f"Command: {exc.cmd}\n"
			f"Return code: {exc.returncode}\n"
			f"Stdout:\n{exc.stdout}\n"
			f"Stderr:\n{exc.stderr}"
		)

	load_dtype = np.uint32 if word_bw <= 32 else np.uint64
	packed = np.loadtxt(words_path, dtype=load_dtype)
	packed = np.asarray(packed)
	if packed.ndim == 0:
		packed = packed.reshape(1)

	got_packet = packet_type(name=packet_name)
	got_packet.deserialize(packed, word_bw=word_bw)

	assert got_packet.is_close(ref_packet, rel_tol=1e-6, abs_tol=1e-6)

