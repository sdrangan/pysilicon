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
