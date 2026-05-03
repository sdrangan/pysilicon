"""Tests for SchemaTransferIF: PhysicalTransport, StreamTransport, master/slave endpoints."""
from __future__ import annotations

import pytest

from pysilicon.hw.clock import Clock
from pysilicon.hw.dataschema import DataList, IntField
from pysilicon.hw.dataunion import (
    DataUnion,
    DataUnionHdr,
    LengthField,
    SchemaIDField,
    SchemaRegistry,
    register_schema,
)
from pysilicon.hw.interface import StreamIF, StreamIFMaster, StreamIFSlave
from pysilicon.hw.schema_transfer_interface import (
    PhysicalTransport,
    SchemaTransferIF,
    SchemaTransferIFMaster,
    SchemaTransferIFSlave,
    StreamTransport,
)
from pysilicon.simulation.simulation import Simulation


# ---------------------------------------------------------------------------
# Test schemas
# ---------------------------------------------------------------------------

U8 = IntField.specialize(bitwidth=8, signed=False)
S16 = IntField.specialize(bitwidth=16, signed=True)
U16 = IntField.specialize(bitwidth=16, signed=False)


class SensorPacket(DataList):
    elements = {"temp_raw": S16, "sensor_id": U8}


_sensor_reg = SchemaRegistry("Sensor")


@register_schema(schema_id=1, registry=_sensor_reg)
class TempPacket(DataList):
    elements = {"temp_raw": S16, "sensor_id": U8}


@register_schema(schema_id=2, registry=_sensor_reg)
class PressPacket(DataList):
    elements = {"pressure_pa": U16, "sensor_id": U8}


@register_schema(schema_id=3, registry=_sensor_reg)
class AccelPacket(DataList):
    elements = {"ax": S16, "ay": S16, "az": S16}


_SensorSchemaID = SchemaIDField.specialize(registry=_sensor_reg, bitwidth=16)
_SensorHdr = DataUnionHdr.specialize(schema_id_type=_SensorSchemaID)
SensorDU = DataUnion.specialize(hdr_type=_SensorHdr)


# ---------------------------------------------------------------------------
# Simulation harness
# ---------------------------------------------------------------------------


class StreamScenario:
    """StreamIF + SchemaTransferIF round-trip harness."""

    def __init__(self, schema_type, bitwidth: int = 32) -> None:
        self.sim = Simulation()
        self.env = self.sim.env
        self.clk = Clock(freq=1.0)

        stream_if = StreamIF(sim=self.sim, clk=self.clk)
        self.stream_master = StreamIFMaster(sim=self.sim, bitwidth=bitwidth)
        self.stream_slave = StreamIFSlave(sim=self.sim, bitwidth=bitwidth)
        stream_if.bind("master", self.stream_master)
        stream_if.bind("slave", self.stream_slave)

        transport = StreamTransport(
            master_ep=self.stream_master,
            slave_ep=self.stream_slave,
        )

        self.received: list = []

        self.schema_master = SchemaTransferIFMaster(
            sim=self.sim, transport=transport, bitwidth=bitwidth
        )
        self.schema_slave = SchemaTransferIFSlave(
            sim=self.sim,
            transport=transport,
            schema_type=schema_type,
            bitwidth=bitwidth,
            rx_proc=self._on_obj,
        )

    def _on_obj(self, obj):
        self.received.append(obj)
        yield self.env.timeout(0)

    def run(self, payloads: list) -> None:
        env = self.env
        n = len(payloads)

        self.schema_slave.pre_sim()

        def tx_proc():
            for obj in payloads:
                yield from self.schema_master.write(obj)

        def rx_monitor():
            while len(self.received) < n:
                yield env.timeout(0.1)

        env.process(self.stream_slave.run_proc())
        tx_done = env.process(tx_proc())
        rx_done = env.process(rx_monitor())
        env.run(until=env.all_of([tx_done, rx_done]))


# ---------------------------------------------------------------------------
# TestPhysicalTransportABC
# ---------------------------------------------------------------------------


class TestPhysicalTransportABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            PhysicalTransport()

    def test_partial_implementation_raises(self):
        class OnlyWrite(PhysicalTransport):
            def write_words(self, words):
                yield

        with pytest.raises(TypeError):
            OnlyWrite()


# ---------------------------------------------------------------------------
# TestStreamTransport
# ---------------------------------------------------------------------------


class TestStreamTransport:
    def _make_env(self):
        sim = Simulation()
        clk = Clock(freq=1.0)
        stream_if = StreamIF(sim=sim, clk=clk)
        master = StreamIFMaster(sim=sim, bitwidth=32)
        slave = StreamIFSlave(sim=sim, bitwidth=32)
        stream_if.bind("master", master)
        stream_if.bind("slave", slave)
        return sim.env, master, slave

    def test_set_rx_callback_assigns_rx_proc(self):
        _, master, slave = self._make_env()
        transport = StreamTransport(master_ep=master, slave_ep=slave)

        def my_callback(words):
            yield

        transport.set_rx_callback(my_callback)
        assert slave.rx_proc is my_callback

    def test_write_words_round_trip(self):
        import numpy as np

        sim = Simulation()
        env = sim.env
        clk = Clock(freq=1.0)
        stream_if = StreamIF(sim=sim, clk=clk)
        master = StreamIFMaster(sim=sim, bitwidth=32)
        slave = StreamIFSlave(sim=sim, bitwidth=32)
        stream_if.bind("master", master)
        stream_if.bind("slave", slave)

        transport = StreamTransport(master_ep=master, slave_ep=slave)
        received = []

        def callback(words):
            received.append(words.copy())
            yield env.timeout(0)

        transport.set_rx_callback(callback)

        words_in = np.array([0xDEAD, 0xBEEF], dtype=np.uint32)

        def tx():
            yield from transport.write_words(words_in)

        env.process(slave.run_proc())
        tx_done = env.process(tx())
        env.run(until=tx_done)
        env.run(until=env.timeout(5))

        assert len(received) == 1
        np.testing.assert_array_equal(received[0], words_in)


# ---------------------------------------------------------------------------
# TestSchemaTransferIFBind
# ---------------------------------------------------------------------------


class TestSchemaTransferIFBind:
    def _make_sim_and_transport(self, bitwidth=32):
        sim = Simulation()
        clk = Clock(freq=1.0)
        stream_if = StreamIF(sim=sim, clk=clk)
        master_ep = StreamIFMaster(sim=sim, bitwidth=bitwidth)
        slave_ep = StreamIFSlave(sim=sim, bitwidth=bitwidth)
        stream_if.bind("master", master_ep)
        stream_if.bind("slave", slave_ep)
        transport = StreamTransport(master_ep=master_ep, slave_ep=slave_ep)
        return sim, transport

    def test_valid_bind_master_then_slave(self):
        sim, transport = self._make_sim_and_transport()
        iface = SchemaTransferIF(sim=sim)
        m = SchemaTransferIFMaster(sim=sim, transport=transport, bitwidth=32)
        s = SchemaTransferIFSlave(
            sim=sim, transport=transport, schema_type=SensorPacket, bitwidth=32
        )
        iface.bind("master", m)
        iface.bind("slave", s)
        assert iface.endpoints["master"] is m
        assert iface.endpoints["slave"] is s

    def test_valid_bind_slave_then_master(self):
        sim, transport = self._make_sim_and_transport()
        iface = SchemaTransferIF(sim=sim)
        m = SchemaTransferIFMaster(sim=sim, transport=transport, bitwidth=32)
        s = SchemaTransferIFSlave(
            sim=sim, transport=transport, schema_type=SensorPacket, bitwidth=32
        )
        iface.bind("slave", s)
        iface.bind("master", m)
        assert iface.endpoints["master"] is m
        assert iface.endpoints["slave"] is s

    def test_wrong_type_on_master_side_raises(self):
        sim, transport = self._make_sim_and_transport()
        iface = SchemaTransferIF(sim=sim)
        s = SchemaTransferIFSlave(
            sim=sim, transport=transport, schema_type=SensorPacket, bitwidth=32
        )
        with pytest.raises(TypeError):
            iface.bind("master", s)

    def test_wrong_type_on_slave_side_raises(self):
        sim, transport = self._make_sim_and_transport()
        iface = SchemaTransferIF(sim=sim)
        m = SchemaTransferIFMaster(sim=sim, transport=transport, bitwidth=32)
        with pytest.raises(TypeError):
            iface.bind("slave", m)

    def test_invalid_ep_name_raises(self):
        sim, transport = self._make_sim_and_transport()
        iface = SchemaTransferIF(sim=sim)
        m = SchemaTransferIFMaster(sim=sim, transport=transport, bitwidth=32)
        with pytest.raises(KeyError):
            iface.bind("tx", m)

    def test_double_bind_raises(self):
        sim, transport = self._make_sim_and_transport()
        iface = SchemaTransferIF(sim=sim)
        m1 = SchemaTransferIFMaster(sim=sim, transport=transport, bitwidth=32)
        m2 = SchemaTransferIFMaster(sim=sim, transport=transport, bitwidth=32)
        iface.bind("master", m1)
        with pytest.raises(ValueError):
            iface.bind("master", m2)

    def test_bitwidth_mismatch_raises(self):
        sim, transport32 = self._make_sim_and_transport(bitwidth=32)
        _, transport64 = self._make_sim_and_transport(bitwidth=64)
        iface = SchemaTransferIF(sim=sim)
        m = SchemaTransferIFMaster(sim=sim, transport=transport32, bitwidth=32)
        s = SchemaTransferIFSlave(
            sim=sim, transport=transport64, schema_type=SensorPacket, bitwidth=64
        )
        iface.bind("master", m)
        with pytest.raises(ValueError):
            iface.bind("slave", s)


# ---------------------------------------------------------------------------
# TestSingleTypeRoundTrip
# ---------------------------------------------------------------------------


class TestSingleTypeRoundTrip:
    def test_single_packet_fields_preserved(self):
        scenario = StreamScenario(SensorPacket)
        pkt = SensorPacket(temp_raw=-42, sensor_id=7)
        scenario.run([pkt])

        assert len(scenario.received) == 1
        rx = scenario.received[0]
        assert int(rx.temp_raw) == -42
        assert int(rx.sensor_id) == 7

    def test_multiple_packets_in_order(self):
        scenario = StreamScenario(SensorPacket)
        packets = [
            SensorPacket(temp_raw=i * 10, sensor_id=i) for i in range(5)
        ]
        scenario.run(packets)

        assert len(scenario.received) == 5
        for i, rx in enumerate(scenario.received):
            assert int(rx.temp_raw) == i * 10
            assert int(rx.sensor_id) == i

    def test_queue_receives_object(self):
        scenario = StreamScenario(SensorPacket)
        # Use queue-based receive instead of rx_proc
        scenario.schema_slave.rx_proc = None
        scenario.schema_slave.pre_sim()

        env = scenario.env

        pkt = SensorPacket(temp_raw=100, sensor_id=3)
        got = []

        def tx_and_read():
            yield from scenario.schema_master.write(pkt)
            event = scenario.schema_slave.queue.get()
            yield event
            got.append(event.value)

        env.process(scenario.stream_slave.run_proc())
        done = env.process(tx_and_read())
        env.run(until=done)

        assert len(got) == 1
        assert int(got[0].temp_raw) == 100
        assert int(got[0].sensor_id) == 3

    def test_negative_temp_raw_roundtrip(self):
        scenario = StreamScenario(SensorPacket)
        pkt = SensorPacket(temp_raw=-32768, sensor_id=255)
        scenario.run([pkt])

        rx = scenario.received[0]
        assert int(rx.temp_raw) == -32768
        assert int(rx.sensor_id) == 255


# ---------------------------------------------------------------------------
# TestMultiTypeRoundTrip
# ---------------------------------------------------------------------------


class TestMultiTypeRoundTrip:
    def _make_payloads(self):
        return [
            TempPacket(temp_raw=-42, sensor_id=7),
            AccelPacket(ax=100, ay=-200, az=980),
            PressPacket(pressure_pa=10132, sensor_id=3),
        ]

    def test_all_three_types_received(self):
        scenario = StreamScenario(SensorDU)
        payloads = self._make_payloads()
        dus = []
        for p in payloads:
            du = SensorDU()
            du.payload = p
            dus.append(du)
        scenario.run(dus)

        assert len(scenario.received) == 3

    def test_payload_types_preserved(self):
        scenario = StreamScenario(SensorDU)
        payloads = self._make_payloads()
        dus = []
        for p in payloads:
            du = SensorDU()
            du.payload = p
            dus.append(du)
        scenario.run(dus)

        rx = scenario.received
        assert type(rx[0].payload) is TempPacket
        assert type(rx[1].payload) is AccelPacket
        assert type(rx[2].payload) is PressPacket

    def test_payload_field_values_preserved(self):
        scenario = StreamScenario(SensorDU)

        du0 = SensorDU()
        du0.payload = TempPacket(temp_raw=-42, sensor_id=7)
        du1 = SensorDU()
        du1.payload = AccelPacket(ax=100, ay=-200, az=980)
        du2 = SensorDU()
        du2.payload = PressPacket(pressure_pa=10132, sensor_id=3)

        scenario.run([du0, du1, du2])

        rx = scenario.received
        p0 = rx[0].payload
        assert int(p0.temp_raw) == -42
        assert int(p0.sensor_id) == 7

        p1 = rx[1].payload
        assert int(p1.ax) == 100
        assert int(p1.ay) == -200
        assert int(p1.az) == 980

        p2 = rx[2].payload
        assert int(p2.pressure_pa) == 10132
        assert int(p2.sensor_id) == 3

    def test_dispatch_table_pattern(self):
        scenario = StreamScenario(SensorDU)
        dispatch_log: list[tuple[type, object]] = []

        def on_receive(du):
            dispatch_log.append((type(du.payload), du.payload))
            yield scenario.env.timeout(0)

        scenario.schema_slave.rx_proc = on_receive
        scenario.schema_slave.pre_sim()

        du = SensorDU()
        du.payload = AccelPacket(ax=1, ay=2, az=3)

        env = scenario.env
        env.process(scenario.stream_slave.run_proc())
        done = env.process(
            (lambda: (yield from scenario.schema_master.write(du)))()
        )
        env.run(until=env.timeout(10))

        assert len(dispatch_log) == 1
        assert dispatch_log[0][0] is AccelPacket

    def test_schema_id_in_header(self):
        scenario = StreamScenario(SensorDU)
        du = SensorDU()
        du.payload = TempPacket(temp_raw=0, sensor_id=0)
        scenario.run([du])

        rx = scenario.received[0]
        assert rx.schema_id == 1

    def test_queue_receives_dataunion(self):
        scenario = StreamScenario(SensorDU)
        scenario.schema_slave.rx_proc = None
        scenario.schema_slave.pre_sim()

        env = scenario.env
        du = SensorDU()
        du.payload = PressPacket(pressure_pa=500, sensor_id=2)
        got = []

        def tx_and_read():
            yield from scenario.schema_master.write(du)
            event = scenario.schema_slave.queue.get()
            yield event
            got.append(event.value)

        env.process(scenario.stream_slave.run_proc())
        done = env.process(tx_and_read())
        env.run(until=done)

        assert len(got) == 1
        assert type(got[0].payload) is PressPacket
        assert int(got[0].payload.pressure_pa) == 500
