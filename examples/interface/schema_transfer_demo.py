"""schema_transfer_demo.py

End-to-end demonstration of SchemaTransferIF in two modes:

  Mode 1 — single type:  every transfer carries exactly one known schema.
            TempSensor sends TempPacket; DataLogger receives and prints them.

  Mode 2 — multi-type (DataUnion):  multiple packet schemas share one link.
            SensorHub sends TempPacket / AccelPacket / PressPacket wrapped
            in a SensorDU; SignalProcessor dispatches by payload type.

Architecture:

    Sender.run_proc()
         |
    SchemaTransferIFMaster.write(obj)
         |   obj.serialize(word_bw) → Words
    StreamTransport.write_words(words)
         |   StreamIFMaster.write(words) → bus latency
    StreamIF (physical)
         |
    StreamIFSlave.run_proc()  [started by Simulation.run_sim()]
         |   calls rx_callback(words)
    SchemaTransferIFSlave._on_words_received(words)
         |   schema_type().deserialize(words, word_bw) → obj
    Receiver.on_object(obj)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pysilicon.hw.clock import Clock
from pysilicon.hw.dataschema import DataList, IntField
from pysilicon.hw.dataunion import (
    DataUnion,
    DataUnionHdr,
    SchemaIDField,
    SchemaRegistry,
    register_schema,
)
from pysilicon.hw.interface import StreamIF, StreamIFMaster, StreamIFSlave
from pysilicon.hw.schema_transfer_interface import (
    SchemaTransferIF,
    SchemaTransferIFMaster,
    SchemaTransferIFSlave,
    StreamTransport,
)
from pysilicon.simulation.simulation import Simulation
from pysilicon.simulation.simobj import ProcessGen, SimObj


# ---------------------------------------------------------------------------
# Shared field types
# ---------------------------------------------------------------------------

U8  = IntField.specialize(bitwidth=8,  signed=False)
S16 = IntField.specialize(bitwidth=16, signed=True)
U16 = IntField.specialize(bitwidth=16, signed=False)


# ---------------------------------------------------------------------------
# Mode 1 schema
# ---------------------------------------------------------------------------

class TempPacket(DataList):
    elements = {"temp_raw": S16, "sensor_id": U8}


# ---------------------------------------------------------------------------
# Mode 2 schemas (DataUnion)
# ---------------------------------------------------------------------------

sensor_reg = SchemaRegistry("Sensor")


@register_schema(schema_id=1, registry=sensor_reg)
class TempMeasurement(DataList):
    elements = {"temp_raw": S16, "sensor_id": U8}


@register_schema(schema_id=2, registry=sensor_reg)
class AccelMeasurement(DataList):
    elements = {"ax": S16, "ay": S16, "az": S16}


@register_schema(schema_id=3, registry=sensor_reg)
class PressureMeasurement(DataList):
    elements = {"pressure_pa": U16, "sensor_id": U8}


SensorSchemaID = SchemaIDField.specialize(registry=sensor_reg, bitwidth=16)
SensorHdr      = DataUnionHdr.specialize(schema_id_type=SensorSchemaID)
SensorDU       = DataUnion.specialize(hdr_type=SensorHdr)


# ---------------------------------------------------------------------------
# Reusable SimObj components
# ---------------------------------------------------------------------------

@dataclass
class Sender(SimObj):
    """Serializes a list of objects and transmits them via SchemaTransferIF."""

    objects: list = field(default_factory=list)
    bitwidth: int = 32

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.stream_ep = StreamIFMaster(sim=self.sim, bitwidth=self.bitwidth)
        self.schema_ep: SchemaTransferIFMaster | None = None  # set during wire-up

    def run_proc(self) -> ProcessGen[None]:
        for obj in self.objects:
            yield from self.schema_ep.write(obj)


@dataclass
class Receiver(SimObj):
    """Receives deserialized objects from SchemaTransferIF via rx_proc callback."""

    bitwidth: int = 32

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.received: list = []
        self.stream_ep = StreamIFSlave(sim=self.sim, bitwidth=self.bitwidth)
        self.schema_ep: SchemaTransferIFSlave | None = None  # set during wire-up

    def on_object(self, obj: Any) -> ProcessGen[None]:
        """Default callback: collect every received object."""
        self.received.append(obj)
        yield self.env.timeout(0)


def wire_up(
    sim: Simulation,
    sender: Sender,
    receiver: Receiver,
    schema_type: type,
    clk: Clock,
    rx_proc=None,
) -> None:
    """Connect sender and receiver through a StreamIF + SchemaTransferIF."""
    stream_if = StreamIF(sim=sim, clk=clk)
    stream_if.bind("master", sender.stream_ep)
    stream_if.bind("slave",  receiver.stream_ep)

    transport = StreamTransport(
        master_ep=sender.stream_ep,
        slave_ep=receiver.stream_ep,
    )

    sender.schema_ep = SchemaTransferIFMaster(
        sim=sim, transport=transport, bitwidth=sender.bitwidth
    )
    receiver.schema_ep = SchemaTransferIFSlave(
        sim=sim,
        transport=transport,
        schema_type=schema_type,
        bitwidth=receiver.bitwidth,
        rx_proc=rx_proc if rx_proc is not None else receiver.on_object,
    )


# ---------------------------------------------------------------------------
# Mode 1: single-type demo
# ---------------------------------------------------------------------------

class SingleTypeDemo:
    """
    TempSensor sends TempPacket objects to DataLogger via SchemaTransferIF.

    No registry, no header — one word per transfer.
    """

    def __init__(self) -> None:
        self.sim = Simulation()
        clk = Clock(freq=1e9)  # 1 GHz

        self.sensor = Sender(
            sim=self.sim,
            objects=[
                TempPacket(temp_raw=-10, sensor_id=1),
                TempPacket(temp_raw=25,  sensor_id=2),
                TempPacket(temp_raw=75,  sensor_id=3),
            ],
        )
        self.logger = Receiver(sim=self.sim)

        wire_up(self.sim, self.sensor, self.logger, TempPacket, clk)

    def run_and_check(self) -> None:
        print("=" * 55)
        print("Mode 1: single-type  (TempPacket)")
        print(f"  {TempPacket.nwords_per_inst(32)} word(s) per transfer  |  no header")
        print("=" * 55)

        self.sim.run_sim()

        for pkt in self.logger.received:
            print(f"  RX  temp_raw={int(pkt.temp_raw):+6d}  sensor_id={int(pkt.sensor_id)}")

        assert len(self.logger.received) == len(self.sensor.objects)
        for tx, rx in zip(self.sensor.objects, self.logger.received):
            assert int(tx.temp_raw)  == int(rx.temp_raw)
            assert int(tx.sensor_id) == int(rx.sensor_id)

        print(f"\n  {len(self.logger.received)} packet(s) received — all verified.\n")


# ---------------------------------------------------------------------------
# Mode 2: multi-type (DataUnion) demo
# ---------------------------------------------------------------------------

class MultiTypeDemo:
    """
    SensorHub sends three different measurement types wrapped in SensorDU.
    SignalProcessor dispatches each incoming DataUnion by payload type.

    SensorDU.nwords_per_inst(32) = hdr_words + max_payload_words = 3 total.
    """

    def __init__(self) -> None:
        self.sim = Simulation()
        clk = Clock(freq=1e9)

        payloads: list[Any] = [
            TempMeasurement(temp_raw=-42, sensor_id=7),
            AccelMeasurement(ax=100, ay=-200, az=980),
            PressureMeasurement(pressure_pa=10132, sensor_id=3),
            TempMeasurement(temp_raw=20, sensor_id=1),
        ]

        objects = []
        for p in payloads:
            du = SensorDU()
            du.payload = p
            objects.append(du)

        self.hub       = Sender(sim=self.sim, objects=objects)
        self.processor = Receiver(sim=self.sim)
        self.dispatch_log: list[tuple[type, Any]] = []

        self._handlers = {
            TempMeasurement:     self._on_temp,
            AccelMeasurement:    self._on_accel,
            PressureMeasurement: self._on_press,
        }

        wire_up(self.sim, self.hub, self.processor, SensorDU, clk,
                rx_proc=self._dispatch)

    # ----- dispatch table -----

    def _dispatch(self, du: SensorDU) -> ProcessGen[None]:
        handler = self._handlers.get(type(du.payload))
        if handler is not None:
            yield from handler(du.payload)

    def _on_temp(self, p: TempMeasurement) -> ProcessGen[None]:
        self.dispatch_log.append((TempMeasurement, p))
        print(f"  RX  TempMeasurement     temp_raw={int(p.temp_raw):+6d}  sensor_id={int(p.sensor_id)}")
        yield self.processor.env.timeout(0)

    def _on_accel(self, p: AccelMeasurement) -> ProcessGen[None]:
        self.dispatch_log.append((AccelMeasurement, p))
        print(f"  RX  AccelMeasurement    ax={int(p.ax):+5d}  ay={int(p.ay):+5d}  az={int(p.az):+5d}")
        yield self.processor.env.timeout(0)

    def _on_press(self, p: PressureMeasurement) -> ProcessGen[None]:
        self.dispatch_log.append((PressureMeasurement, p))
        print(f"  RX  PressureMeasurement pressure_pa={int(p.pressure_pa)}  sensor_id={int(p.sensor_id)}")
        yield self.processor.env.timeout(0)

    def run_and_check(self) -> None:
        print("=" * 55)
        print("Mode 2: multi-type   (SensorDU = DataUnion)")
        print(f"  {SensorDU.nwords_per_inst(32)} word(s) per transfer  |  schema_id in header")
        print("=" * 55)

        self.sim.run_sim()

        assert len(self.dispatch_log) == len(self.hub.objects)
        assert self.dispatch_log[0][0] is TempMeasurement
        assert self.dispatch_log[1][0] is AccelMeasurement
        assert self.dispatch_log[2][0] is PressureMeasurement
        assert self.dispatch_log[3][0] is TempMeasurement

        print(f"\n  {len(self.dispatch_log)} DataUnion(s) dispatched — all verified.\n")


# ---------------------------------------------------------------------------
# Queue-based receive (alternative to rx_proc callback)
# ---------------------------------------------------------------------------

def queue_receive_demo() -> None:
    """
    Demonstrates polling the slave's simpy.Store queue instead of using rx_proc.
    Useful when the consumer prefers to pull objects in its own run_proc loop.
    """
    print("=" * 55)
    print("Queue-based receive")
    print("=" * 55)

    sim = Simulation()
    env = sim.env
    clk = Clock(freq=1e9)

    stream_if = StreamIF(sim=sim, clk=clk)
    stream_master = StreamIFMaster(sim=sim, bitwidth=32)
    stream_slave  = StreamIFSlave(sim=sim, bitwidth=32)
    stream_if.bind("master", stream_master)
    stream_if.bind("slave",  stream_slave)

    transport = StreamTransport(master_ep=stream_master, slave_ep=stream_slave)
    schema_master = SchemaTransferIFMaster(sim=sim, transport=transport, bitwidth=32)
    schema_slave  = SchemaTransferIFSlave(
        sim=sim, transport=transport, schema_type=TempPacket, bitwidth=32,
        # No rx_proc — consumer polls queue directly
    )

    received: list = []

    def consumer() -> ProcessGen[None]:
        for _ in range(3):
            event = schema_slave.queue.get()
            yield event
            pkt = event.value
            received.append(pkt)
            print(f"  Queue.get()  temp_raw={int(pkt.temp_raw):+6d}  sensor_id={int(pkt.sensor_id)}")

    def producer() -> ProcessGen[None]:
        for t, sid in [(-5, 1), (15, 2), (40, 3)]:
            yield from schema_master.write(TempPacket(temp_raw=t, sensor_id=sid))

    schema_slave.pre_sim()
    env.process(stream_slave.run_proc())
    c = env.process(consumer())
    env.process(producer())
    env.run(until=c)

    assert len(received) == 3
    print(f"\n  {len(received)} packet(s) dequeued — all verified.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SingleTypeDemo().run_and_check()
    MultiTypeDemo().run_and_check()
    queue_receive_demo()
