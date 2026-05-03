"""Tests for SchemaTransferIF: PhysicalTransport, StreamTransport, master/slave endpoints."""
from __future__ import annotations

import numpy as np
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
    ArrayTransferIF,
    ArrayTransferIFMaster,
    ArrayTransferIFSlave,
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


# ---------------------------------------------------------------------------
# Pull model helpers
# ---------------------------------------------------------------------------

def _make_pull_scenario(schema_type, bitwidth=32):
    """StreamIF + SchemaTransferIF wired for pull mode (no rx_proc, no run_proc)."""
    sim = Simulation()
    env = sim.env
    clk = Clock(freq=1.0)
    stream_if = StreamIF(sim=sim, clk=clk)
    stream_master = StreamIFMaster(sim=sim, bitwidth=bitwidth)
    stream_slave = StreamIFSlave(sim=sim, bitwidth=bitwidth)
    stream_if.bind("master", stream_master)
    stream_if.bind("slave", stream_slave)
    transport = StreamTransport(master_ep=stream_master, slave_ep=stream_slave)
    schema_master = SchemaTransferIFMaster(sim=sim, transport=transport, bitwidth=bitwidth)
    schema_slave = SchemaTransferIFSlave(
        sim=sim, transport=transport, schema_type=schema_type,
        bitwidth=bitwidth, pull_mode=True,
    )
    return sim, env, stream_if, schema_master, schema_slave


def _make_array_scenario(element_type, bitwidth=32):
    """StreamIF + ArrayTransferIF wired for pull mode."""
    sim = Simulation()
    env = sim.env
    clk = Clock(freq=1.0)
    stream_if = StreamIF(sim=sim, clk=clk)
    stream_master = StreamIFMaster(sim=sim, bitwidth=bitwidth)
    stream_slave = StreamIFSlave(sim=sim, bitwidth=bitwidth)
    stream_if.bind("master", stream_master)
    stream_if.bind("slave", stream_slave)
    transport = StreamTransport(master_ep=stream_master, slave_ep=stream_slave)
    arr_master = ArrayTransferIFMaster(
        sim=sim, transport=transport, element_type=element_type, bitwidth=bitwidth
    )
    arr_slave = ArrayTransferIFSlave(
        sim=sim, transport=transport, element_type=element_type,
        bitwidth=bitwidth, pull_mode=True,
    )
    return sim, env, stream_if, arr_master, arr_slave


# ---------------------------------------------------------------------------
# TestPullModelRunProcNoOp
# ---------------------------------------------------------------------------


class TestPullModelRunProcNoOp:
    def test_run_proc_exits_immediately_when_rx_proc_none(self):
        """run_proc with rx_proc=None should finish without consuming from data_buffer."""
        sim = Simulation()
        env = sim.env
        clk = Clock(freq=1.0)
        stream_if = StreamIF(sim=sim, clk=clk)
        master = StreamIFMaster(sim=sim, bitwidth=32)
        slave = StreamIFSlave(sim=sim, bitwidth=32)  # rx_proc=None by default
        stream_if.bind("master", master)
        stream_if.bind("slave", slave)

        proc = env.process(slave.run_proc())
        env.run(until=proc)
        # process should be finished (not stuck waiting for data)
        assert proc.processed

    def test_get_receives_words_when_run_proc_not_started(self):
        """get() retrieves words written by the master without run_proc running."""
        import numpy as np

        sim = Simulation()
        env = sim.env
        clk = Clock(freq=1.0)
        stream_if = StreamIF(sim=sim, clk=clk)
        master = StreamIFMaster(sim=sim, bitwidth=32)
        slave = StreamIFSlave(sim=sim, bitwidth=32)
        stream_if.bind("master", master)
        stream_if.bind("slave", slave)

        words_in = np.array([10, 20, 30], dtype=np.uint32)
        received = []

        def proc():
            yield from master.write(words_in)
            words_out = yield from slave.get()
            received.append(words_out)

        done = env.process(proc())
        env.run(until=done)

        assert len(received) == 1
        np.testing.assert_array_equal(received[0], words_in)


# ---------------------------------------------------------------------------
# TestSchemaTransferIFSlavePullGet
# ---------------------------------------------------------------------------


class TestSchemaTransferIFSlavePullGet:
    def test_get_returns_deserialized_object(self):
        sim, env, _, schema_master, schema_slave = _make_pull_scenario(SensorPacket)
        got = []

        def proc():
            pkt = SensorPacket(temp_raw=-7, sensor_id=3)
            yield from schema_master.write(pkt)
            obj = yield from schema_slave.get()
            got.append(obj)

        done = env.process(proc())
        env.run(until=done)

        assert len(got) == 1
        assert int(got[0].temp_raw) == -7
        assert int(got[0].sensor_id) == 3

    def test_get_sequential_two_different_types(self):
        """Component can interleave two schema slaves on the same stream in sequence."""
        sim = Simulation()
        env = sim.env
        clk = Clock(freq=1.0)
        stream_if = StreamIF(sim=sim, clk=clk)
        sm = StreamIFMaster(sim=sim, bitwidth=32)
        ss = StreamIFSlave(sim=sim, bitwidth=32)
        stream_if.bind("master", sm)
        stream_if.bind("slave", ss)
        transport = StreamTransport(master_ep=sm, slave_ep=ss)

        sensor_master = SchemaTransferIFMaster(sim=sim, transport=transport, bitwidth=32)
        sensor_slave = SchemaTransferIFSlave(
            sim=sim, transport=transport, schema_type=SensorPacket,
            bitwidth=32, pull_mode=True,
        )
        accel_master = SchemaTransferIFMaster(sim=sim, transport=transport, bitwidth=32)
        accel_slave = SchemaTransferIFSlave(
            sim=sim, transport=transport, schema_type=AccelPacket,
            bitwidth=32, pull_mode=True,
        )

        got_sensor = []
        got_accel = []

        def proc():
            pkt1 = SensorPacket(temp_raw=100, sensor_id=1)
            pkt2 = AccelPacket(ax=1, ay=2, az=3)
            yield from sensor_master.write(pkt1)
            yield from accel_master.write(pkt2)
            got_sensor.append((yield from sensor_slave.get()))
            got_accel.append((yield from accel_slave.get()))

        done = env.process(proc())
        env.run(until=done)

        assert int(got_sensor[0].temp_raw) == 100
        assert int(got_accel[0].ax) == 1
        assert int(got_accel[0].az) == 3

    def test_pull_mode_pre_sim_does_not_set_callback(self):
        sim, env, _, _, schema_slave = _make_pull_scenario(SensorPacket)
        schema_slave.pre_sim()
        transport = schema_slave.transport
        assert transport.slave_ep.rx_proc is None


# ---------------------------------------------------------------------------
# TestArrayTransferIFBind
# ---------------------------------------------------------------------------


class TestArrayTransferIFBind:
    def _make_sim_and_transport(self, bitwidth=32):
        sim = Simulation()
        clk = Clock(freq=1.0)
        stream_if = StreamIF(sim=sim, clk=clk)
        me = StreamIFMaster(sim=sim, bitwidth=bitwidth)
        se = StreamIFSlave(sim=sim, bitwidth=bitwidth)
        stream_if.bind("master", me)
        stream_if.bind("slave", se)
        return sim, StreamTransport(master_ep=me, slave_ep=se)

    def test_valid_bind(self):
        sim, t = self._make_sim_and_transport()
        iface = ArrayTransferIF(sim=sim)
        m = ArrayTransferIFMaster(sim=sim, transport=t, element_type=U8, bitwidth=32)
        s = ArrayTransferIFSlave(sim=sim, transport=t, element_type=U8, bitwidth=32)
        iface.bind("master", m)
        iface.bind("slave", s)
        assert iface.endpoints["master"] is m
        assert iface.endpoints["slave"] is s

    def test_wrong_type_on_master_side_raises(self):
        sim, t = self._make_sim_and_transport()
        iface = ArrayTransferIF(sim=sim)
        s = ArrayTransferIFSlave(sim=sim, transport=t, element_type=U8, bitwidth=32)
        with pytest.raises(TypeError):
            iface.bind("master", s)

    def test_wrong_type_on_slave_side_raises(self):
        sim, t = self._make_sim_and_transport()
        iface = ArrayTransferIF(sim=sim)
        m = ArrayTransferIFMaster(sim=sim, transport=t, element_type=U8, bitwidth=32)
        with pytest.raises(TypeError):
            iface.bind("slave", m)

    def test_invalid_ep_name_raises(self):
        sim, t = self._make_sim_and_transport()
        iface = ArrayTransferIF(sim=sim)
        m = ArrayTransferIFMaster(sim=sim, transport=t, element_type=U8, bitwidth=32)
        with pytest.raises(KeyError):
            iface.bind("tx", m)

    def test_element_type_mismatch_raises(self):
        sim, t = self._make_sim_and_transport()
        iface = ArrayTransferIF(sim=sim)
        m = ArrayTransferIFMaster(sim=sim, transport=t, element_type=U8, bitwidth=32)
        s = ArrayTransferIFSlave(sim=sim, transport=t, element_type=S16, bitwidth=32)
        iface.bind("master", m)
        with pytest.raises(ValueError, match="element_type"):
            iface.bind("slave", s)

    def test_bitwidth_mismatch_raises(self):
        sim, t32 = self._make_sim_and_transport(bitwidth=32)
        _, t64 = self._make_sim_and_transport(bitwidth=64)
        iface = ArrayTransferIF(sim=sim)
        m = ArrayTransferIFMaster(sim=sim, transport=t32, element_type=U8, bitwidth=32)
        s = ArrayTransferIFSlave(sim=sim, transport=t64, element_type=U8, bitwidth=64)
        iface.bind("master", m)
        with pytest.raises(ValueError, match="bitwidth"):
            iface.bind("slave", s)


# ---------------------------------------------------------------------------
# TestArrayTransferIFRoundTrip
# ---------------------------------------------------------------------------

Float32 = None  # resolved lazily below to avoid import-time side-effects

def _float32_type():
    from pysilicon.hw.dataschema import FloatField
    return FloatField.specialize(bitwidth=32)


class TestArrayTransferIFRoundTrip:
    def test_u8_array_round_trip(self):
        """Scalar IntField: get() returns np.ndarray[uint8]."""
        sim, env, _, arr_master, arr_slave = _make_array_scenario(U8)
        result = []

        def proc():
            yield from arr_master.write(np.array([1, 2, 3, 4], dtype=np.uint8))
            elems = yield from arr_slave.get(count=4)
            result.append(elems)

        done = env.process(proc())
        env.run(until=done)

        assert isinstance(result[0], np.ndarray)
        assert result[0].dtype == np.uint8
        assert np.array_equal(result[0], [1, 2, 3, 4])

    def test_s16_array_round_trip(self):
        """Scalar signed IntField: get() returns np.ndarray[int16]."""
        sim, env, _, arr_master, arr_slave = _make_array_scenario(S16)
        result = []

        def proc():
            yield from arr_master.write(np.array([-100, 0, 100], dtype=np.int16))
            elems = yield from arr_slave.get(count=3)
            result.append(elems)

        done = env.process(proc())
        env.run(until=done)

        assert isinstance(result[0], np.ndarray)
        assert result[0].dtype == np.int16
        assert np.array_equal(result[0], [-100, 0, 100])

    def test_float32_array_round_trip(self):
        """FloatField: write/get accept/return np.ndarray[float32]."""
        F32 = _float32_type()
        sim, env, _, arr_master, arr_slave = _make_array_scenario(F32)
        values = np.array([1.0, -2.5, 0.0, 3.14], dtype=np.float32)
        result = []

        def proc():
            yield from arr_master.write(values)
            elems = yield from arr_slave.get(count=len(values))
            result.append(elems)

        done = env.process(proc())
        env.run(until=done)

        assert isinstance(result[0], np.ndarray)
        assert result[0].dtype == np.float32
        assert np.allclose(result[0], values, atol=1e-6)

    def test_schema_instance_write(self):
        """write() still accepts schema instances (slow path) and get() returns ndarray."""
        sim, env, _, arr_master, arr_slave = _make_array_scenario(U8)
        result = []

        def proc():
            yield from arr_master.write([U8(v) for v in [1, 2, 3, 4]])
            elems = yield from arr_slave.get(count=4)
            result.append(elems)

        done = env.process(proc())
        env.run(until=done)

        assert isinstance(result[0], np.ndarray)
        assert np.array_equal(result[0], [1, 2, 3, 4])

    def test_raw_value_write(self):
        """write() accepts raw Python values (slow path) and get() returns ndarray."""
        sim, env, _, arr_master, arr_slave = _make_array_scenario(U8)
        result = []

        def proc():
            yield from arr_master.write([10, 20, 30])  # raw ints → slow path
            elems = yield from arr_slave.get(count=3)
            result.append(elems)

        done = env.process(proc())
        env.run(until=done)

        assert isinstance(result[0], np.ndarray)
        assert np.array_equal(result[0], [10, 20, 30])

    def test_tlast_early_raises(self):
        """Burst shorter than count * nwords_per_elem raises RuntimeError."""
        import numpy as np
        sim, env, _, arr_master, arr_slave = _make_array_scenario(U8)
        errors = []

        def proc():
            # manually send a 2-word burst but ask get() for 4 elements
            stream_master = arr_master.transport.master_ep
            yield from stream_master.write(np.array([1, 2], dtype=np.uint32))
            try:
                yield from arr_slave.get(count=4)
            except RuntimeError as exc:
                errors.append(str(exc))

        done = env.process(proc())
        env.run(until=done)

        assert len(errors) == 1
        assert "TLAST early" in errors[0]

    def test_missing_tlast_raises(self):
        """Burst longer than count * nwords_per_elem raises RuntimeError."""
        import numpy as np
        sim, env, _, arr_master, arr_slave = _make_array_scenario(U8)
        errors = []

        def proc():
            stream_master = arr_master.transport.master_ep
            yield from stream_master.write(np.array([1, 2, 3, 4], dtype=np.uint32))
            try:
                yield from arr_slave.get(count=2)
            except RuntimeError as exc:
                errors.append(str(exc))

        done = env.process(proc())
        env.run(until=done)

        assert len(errors) == 1
        assert "Missing TLAST" in errors[0]

    def test_push_mode_rx_proc(self):
        """ArrayTransferIFSlave push mode delivers np.ndarray via rx_proc."""
        sim = Simulation()
        env = sim.env
        clk = Clock(freq=1.0)
        stream_if = StreamIF(sim=sim, clk=clk)
        sm = StreamIFMaster(sim=sim, bitwidth=32)
        ss = StreamIFSlave(sim=sim, bitwidth=32)
        stream_if.bind("master", sm)
        stream_if.bind("slave", ss)
        transport = StreamTransport(master_ep=sm, slave_ep=ss)

        received = []

        def on_elements(elems):
            received.append(elems)
            yield env.timeout(0)

        arr_master = ArrayTransferIFMaster(
            sim=sim, transport=transport, element_type=U8, bitwidth=32
        )
        arr_slave = ArrayTransferIFSlave(
            sim=sim, transport=transport, element_type=U8,
            bitwidth=32, rx_proc=on_elements,
        )
        arr_slave.pre_sim()

        def tx():
            yield from arr_master.write(np.array([5, 6, 7], dtype=np.uint8))

        env.process(ss.run_proc())
        done = env.process(tx())
        env.run(until=env.timeout(10))

        assert len(received) == 1
        assert isinstance(received[0], np.ndarray)
        assert np.array_equal(received[0], [5, 6, 7])


# ---------------------------------------------------------------------------
# TestPhysicalTransportABC (additions)
# ---------------------------------------------------------------------------


class TestPhysicalTransportABCGetWords:
    def test_partial_impl_without_get_words_raises(self):
        class WriteAndCallback(PhysicalTransport):
            def write_words(self, words):
                yield

            def set_rx_callback(self, cb):
                pass

        with pytest.raises(TypeError):
            WriteAndCallback()
