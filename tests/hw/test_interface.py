from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.testing as npt
import pytest

from pysilicon.hw.clock import Clock
from pysilicon.hw.interface import (
    CrossBarIF,
    CrossBarIFInput,
    CrossBarIFOutput,
    StreamIF,
    StreamIFMaster,
    StreamIFSlave,
    StreamType,
    TransferNotifyType,
    Words,
)
from pysilicon.simulation.simulation import Simulation


# ---------------------------------------------------------------------------
# Shared timing test infrastructure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransferCase:
    name: str
    packets: tuple[tuple[int, ...], ...]
    tx_gap: float
    queue_size: int | None
    rx_processing_delay: float
    expected_write_end: tuple[float, ...]
    expected_rx_start: tuple[float, ...]
    expected_rx_end: tuple[float, ...]


class StreamScenario:
    """1 master + 1 slave StreamIF scenario for timing / data tests."""

    def __init__(self, case: TransferCase) -> None:
        self.case = case
        self.sim = Simulation()
        self.iface = StreamIF(sim=self.sim, clk=Clock(freq=1))
        self.master = StreamIFMaster(sim=self.sim, bitwidth=32)
        self.slave = StreamIFSlave(
            sim=self.sim,
            bitwidth=32,
            queue_size=case.queue_size,
            rx_proc=self.rx_proc,
        )
        self.iface.bind("master", self.master)
        self.iface.bind("slave", self.slave)

        self.packets = [np.array(p, dtype=np.uint32) for p in case.packets]
        self.write_end_times: list[float] = []
        self.rx_start_times: list[float] = []
        self.rx_end_times: list[float] = []
        self.received_packets: list[np.ndarray] = []

    def rx_proc(self, words: Words):
        self.rx_start_times.append(self.sim.env.now)
        self.received_packets.append(np.array(words, copy=True))
        yield self.sim.env.timeout(self.case.rx_processing_delay)
        self.rx_end_times.append(self.sim.env.now)

    def run(self) -> None:
        env = self.sim.env
        master_done = env.event()
        rx_done = env.event()

        def master_proc():
            for packet in self.packets:
                yield env.process(self.master.write(packet))
                self.write_end_times.append(env.now)
                yield env.timeout(self.case.tx_gap)
            master_done.succeed()

        def rx_monitor():
            while len(self.rx_end_times) < len(self.packets):
                yield env.timeout(0.1)
            rx_done.succeed()

        env.process(self.slave.run_proc())
        env.process(master_proc())
        env.process(rx_monitor())
        env.run(until=env.all_of([master_done, rx_done]))

    @property
    def rx_ep(self):
        return self.slave


class CrossBarScenario11:
    """1×1 CrossBarIF scenario — same timing semantics as StreamScenario."""

    def __init__(self, case: TransferCase) -> None:
        self.case = case
        self.sim = Simulation()
        self.xbar = CrossBarIF(sim=self.sim, clk=Clock(freq=1), nports_in=1, nports_out=1)
        self.inp = CrossBarIFInput(sim=self.sim, bitwidth=32)
        self.out = CrossBarIFOutput(
            sim=self.sim,
            bitwidth=32,
            queue_size=case.queue_size,
            rx_proc=self.rx_proc,
        )
        self.xbar.bind("in_0", self.inp)
        self.xbar.bind("out_0", self.out)

        self.packets = [np.array(p, dtype=np.uint32) for p in case.packets]
        self.write_end_times: list[float] = []
        self.rx_start_times: list[float] = []
        self.rx_end_times: list[float] = []
        self.received_packets: list[np.ndarray] = []

    def rx_proc(self, words: Words):
        self.rx_start_times.append(self.sim.env.now)
        self.received_packets.append(np.array(words, copy=True))
        yield self.sim.env.timeout(self.case.rx_processing_delay)
        self.rx_end_times.append(self.sim.env.now)

    def run(self) -> None:
        env = self.sim.env
        master_done = env.event()
        rx_done = env.event()

        def sender_proc():
            for packet in self.packets:
                yield env.process(self.inp.write(packet))
                self.write_end_times.append(env.now)
                yield env.timeout(self.case.tx_gap)
            master_done.succeed()

        def rx_monitor():
            while len(self.rx_end_times) < len(self.packets):
                yield env.timeout(0.1)
            rx_done.succeed()

        env.process(self.out.run_proc())
        env.process(sender_proc())
        env.process(rx_monitor())
        env.run(until=env.all_of([master_done, rx_done]))

    @property
    def rx_ep(self):
        return self.out


TRANSFER_CASES = [
    pytest.param(
        TransferCase(
            name="spaced_packets_no_backpressure",
            packets=((10, 11, 12), (20, 21), (30,)),
            tx_gap=1.0,
            queue_size=None,
            rx_processing_delay=0.0,
            expected_write_end=(3.0, 6.0, 8.0),
            expected_rx_start=(3.0, 6.0, 8.0),
            expected_rx_end=(3.0, 6.0, 8.0),
        ),
        id="no-backpressure",
    ),
    pytest.param(
        TransferCase(
            name="deep_queue_absorbs_receiver_delay",
            packets=((1, 2), (3, 4), (5, 6)),
            tx_gap=0.0,
            queue_size=6,
            rx_processing_delay=5.0,
            expected_write_end=(2.0, 4.0, 6.0),
            expected_rx_start=(2.0, 7.0, 12.0),
            expected_rx_end=(7.0, 12.0, 17.0),
        ),
        id="deep-queue",
    ),
    pytest.param(
        TransferCase(
            name="shallow_queue_forces_backpressure",
            packets=((1, 2), (3, 4), (5, 6)),
            tx_gap=0.0,
            queue_size=2,
            rx_processing_delay=5.0,
            expected_write_end=(2.0, 4.0, 7.0),
            expected_rx_start=(2.0, 7.0, 12.0),
            expected_rx_end=(7.0, 12.0, 17.0),
        ),
        id="shallow-queue-backpressure",
    ),
]

SCENARIO_CLASSES = [
    pytest.param(StreamScenario, id="stream"),
    pytest.param(CrossBarScenario11, id="crossbar-1x1"),
]


@pytest.mark.parametrize("scenario_cls", SCENARIO_CLASSES)
@pytest.mark.parametrize("case", TRANSFER_CASES)
def test_timing(case: TransferCase, scenario_cls) -> None:
    sc = scenario_cls(case)
    sc.run()

    assert sc.write_end_times == pytest.approx(list(case.expected_write_end))
    assert sc.rx_start_times == pytest.approx(list(case.expected_rx_start))
    assert sc.rx_end_times == pytest.approx(list(case.expected_rx_end))
    assert sc.rx_ep.nrx.level == 0
    assert sc.rx_ep.ntx.level == 0
    for received, expected in zip(sc.received_packets, case.packets, strict=True):
        npt.assert_array_equal(received, expected)


# ---------------------------------------------------------------------------
# StreamIF-specific validation tests
# ---------------------------------------------------------------------------

def test_stream_missing_clock():
    sim = Simulation()
    with pytest.raises(ValueError, match="clock"):
        StreamIF(sim=sim)


def test_stream_bind_wrong_slave_type():
    sim = Simulation()
    iface = StreamIF(sim=sim, clk=Clock(freq=1))
    wrong = StreamIFMaster(sim=sim, bitwidth=32)
    with pytest.raises(TypeError):
        iface.bind("slave", wrong)


def test_stream_bind_wrong_master_type():
    sim = Simulation()
    iface = StreamIF(sim=sim, clk=Clock(freq=1))
    wrong = StreamIFSlave(sim=sim, bitwidth=32)
    with pytest.raises(TypeError):
        iface.bind("master", wrong)


def test_stream_bind_invalid_ep_name():
    sim = Simulation()
    iface = StreamIF(sim=sim, clk=Clock(freq=1))
    ep = StreamIFMaster(sim=sim, bitwidth=32)
    with pytest.raises(KeyError):
        iface.bind("in_0", ep)


def test_stream_bind_bitwidth_mismatch():
    sim = Simulation()
    iface = StreamIF(sim=sim, clk=Clock(freq=1), bitwidth=32)
    ep = StreamIFSlave(sim=sim, bitwidth=64)
    with pytest.raises(ValueError, match="bitwidth"):
        iface.bind("slave", ep)


def test_stream_bind_stream_type_mismatch():
    sim = Simulation()
    iface = StreamIF(sim=sim, clk=Clock(freq=1), stream_type=StreamType.axi4)
    ep = StreamIFSlave(sim=sim, bitwidth=32, stream_type=StreamType.hls)
    with pytest.raises(ValueError, match="stream type"):
        iface.bind("slave", ep)


def test_stream_bind_notify_type_mismatch():
    sim = Simulation()
    iface = StreamIF(
        sim=sim, clk=Clock(freq=1), notify_type=TransferNotifyType.end_only
    )
    ep = StreamIFSlave(sim=sim, bitwidth=32, notify_type=TransferNotifyType.begin_end)
    with pytest.raises(ValueError, match="notify type"):
        iface.bind("slave", ep)


def test_stream_write_without_slave():
    sim = Simulation()
    iface = StreamIF(sim=sim, clk=Clock(freq=1))
    master = StreamIFMaster(sim=sim, bitwidth=32)
    iface.bind("master", master)

    result: list[Exception] = []

    def _proc():
        try:
            yield from iface.write(np.array([1, 2], dtype=np.uint32))
        except RuntimeError as exc:
            result.append(exc)

    sim.env.process(_proc())
    sim.env.run()
    assert len(result) == 1
    assert "slave" in str(result[0]).lower()


def test_stream_master_write_without_binding():
    sim = Simulation()
    master = StreamIFMaster(sim=sim, bitwidth=32)
    words = np.array([1, 2], dtype=np.uint32)

    result: list[Exception] = []

    def _proc():
        try:
            yield from master.write(words)
        except RuntimeError as exc:
            result.append(exc)

    sim.env.process(_proc())
    sim.env.run()
    assert len(result) == 1
    assert "not bound" in str(result[0]).lower()
