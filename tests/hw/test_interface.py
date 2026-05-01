from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.testing as npt
import pytest
import simpy

from pysilicon.hw.clock import Clock
from pysilicon.hw.interface import StreamIF, StreamIFMaster, StreamIFSlave, Words
from pysilicon.simulation.simobj import SimConfig


@dataclass(frozen=True)
class StreamCase:
    name: str
    packets: tuple[tuple[int, ...], ...]
    tx_gap: float
    queue_size: int | None
    rx_processing_delay: float
    expected_write_end: tuple[float, ...]
    expected_rx_start: tuple[float, ...]
    expected_rx_end: tuple[float, ...]


class StreamScenario:
    def __init__(self, case: StreamCase) -> None:
        self.case = case
        self.env = simpy.Environment()
        self.sim_config = SimConfig(env=self.env)
        self.iface = StreamIF(sim_config=self.sim_config, clk=Clock(freq=1))
        self.master = StreamIFMaster(sim_config=self.sim_config, bitwidth=32)
        self.slave = StreamIFSlave(
            sim_config=self.sim_config,
            bitwidth=32,
            queue_size=case.queue_size,
            rx_proc=self.rx_proc,
        )
        self.iface.bind("master", self.master)
        self.iface.bind("slave", self.slave)
        self.env.process(self.slave.run_proc())

        self.packets = [np.array(packet, dtype=np.uint32) for packet in case.packets]
        self.write_start_times: list[float] = []
        self.write_end_times: list[float] = []
        self.rx_start_times: list[float] = []
        self.rx_end_times: list[float] = []
        self.received_packets: list[np.ndarray] = []
        self.master_done = self.env.event()
        self.rx_done = self.env.event()

    def master_proc(self):
        for packet in self.packets:
            self.write_start_times.append(self.env.now)
            yield self.env.process(self.master.write(packet))
            self.write_end_times.append(self.env.now)
            yield self.env.timeout(self.case.tx_gap)
        self.master_done.succeed()

    def rx_proc(self, words: Words):
        self.rx_start_times.append(self.env.now)
        self.received_packets.append(np.array(words, copy=True))
        yield self.env.timeout(self.case.rx_processing_delay)
        self.rx_end_times.append(self.env.now)
        if len(self.rx_end_times) == len(self.packets) and not self.rx_done.triggered:
            self.rx_done.succeed()

    def run(self) -> None:
        self.env.process(self.master_proc())
        self.env.run(until=simpy.events.AllOf(self.env, [self.master_done, self.rx_done]))


STREAM_CASES = [
    pytest.param(
        StreamCase(
            name="spaced_packets_no_backpressure",
            packets=((10, 11, 12), (20, 21), (30,)),
            tx_gap=1.0,
            queue_size=None,
            rx_processing_delay=0.0,
            expected_write_end=(3.0, 6.0, 8.0),
            expected_rx_start=(3.0, 6.0, 8.0),
            expected_rx_end=(3.0, 6.0, 8.0),
        ),
        id="spaced-packets-no-backpressure",
    ),
    pytest.param(
        StreamCase(
            name="deep_queue_absorbs_receiver_delay",
            packets=((1, 2), (3, 4), (5, 6)),
            tx_gap=0.0,
            queue_size=6,
            rx_processing_delay=5.0,
            expected_write_end=(2.0, 4.0, 6.0),
            expected_rx_start=(2.0, 7.0, 12.0),
            expected_rx_end=(7.0, 12.0, 17.0),
        ),
        id="deep-queue-absorbs-receiver-delay",
    ),
    pytest.param(
        StreamCase(
            name="shallow_queue_forces_backpressure",
            packets=((1, 2), (3, 4), (5, 6)),
            tx_gap=0.0,
            queue_size=2,
            rx_processing_delay=5.0,
            expected_write_end=(2.0, 4.0, 7.0),
            expected_rx_start=(2.0, 7.0, 12.0),
            expected_rx_end=(7.0, 12.0, 17.0),
        ),
        id="shallow-queue-forces-backpressure",
    ),
]


@pytest.mark.parametrize("case", STREAM_CASES)
def test_stream_interface_timing_cases(case: StreamCase) -> None:
    scenario = StreamScenario(case)

    scenario.run()

    assert len(scenario.write_start_times) == len(case.packets)
    assert scenario.write_end_times == list(case.expected_write_end)
    assert scenario.rx_start_times == list(case.expected_rx_start)
    assert scenario.rx_end_times == list(case.expected_rx_end)
    assert scenario.slave.nrx.level == 0
    assert scenario.slave.ntx.level == 0
    for received, expected in zip(scenario.received_packets, scenario.packets, strict=True):
        npt.assert_array_equal(received, expected)