"""Unit tests for CrossbarIF, CrossbarIFMaster, and CrossbarIFSlave."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.testing as npt
import pytest
import simpy

from pysilicon.hw.clock import Clock
from pysilicon.hw.interface import (
    CrossbarIF,
    CrossbarIFMaster,
    CrossbarIFSlave,
    StreamIFMaster,
    StreamIFSlave,
    Words,
)
from pysilicon.simulation.simulation import Simulation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_xbar(
    num_masters: int = 1,
    num_slaves: int = 1,
    bitwidth: int = 32,
    freq: float = 1.0,
    latency_init: float = 0.0,
    queue_size: int | None = None,
    rx_procs: dict[int, object] | None = None,
) -> tuple[Simulation, CrossbarIF, list[CrossbarIFMaster], list[CrossbarIFSlave]]:
    """Build and bind a full crossbar fixture; return (sim, xbar, masters, slaves)."""
    sim = Simulation()
    xbar = CrossbarIF(
        sim=sim,
        clk=Clock(freq=freq),
        num_masters=num_masters,
        num_slaves=num_slaves,
        bitwidth=bitwidth,
        latency_init=latency_init,
    )
    masters = [CrossbarIFMaster(sim=sim, bitwidth=bitwidth) for _ in range(num_masters)]
    slaves = [
        CrossbarIFSlave(
            sim=sim,
            bitwidth=bitwidth,
            queue_size=queue_size,
            rx_proc=(rx_procs or {}).get(i),
        )
        for i in range(num_slaves)
    ]
    for i, m in enumerate(masters):
        xbar.bind(f'master_{i}', m)
    for i, s in enumerate(slaves):
        xbar.bind(f'slave_{i}', s)
    return sim, xbar, masters, slaves


# ---------------------------------------------------------------------------
# Construction / validation tests
# ---------------------------------------------------------------------------

class TestCrossbarConstruction:
    def test_basic_construction(self) -> None:
        sim, xbar, masters, slaves = _make_xbar(num_masters=2, num_slaves=3)
        assert xbar.num_masters == 2
        assert xbar.num_slaves == 3
        assert xbar.bitwidth == 32
        assert set(xbar.endpoint_names) == {
            'master_0', 'master_1', 'slave_0', 'slave_1', 'slave_2'
        }

    def test_no_clock_raises(self) -> None:
        sim = Simulation()
        with pytest.raises(ValueError, match="clock must be provided"):
            CrossbarIF(sim=sim, num_masters=1, num_slaves=1)

    def test_invalid_num_masters(self) -> None:
        sim = Simulation()
        with pytest.raises(ValueError, match="num_masters"):
            CrossbarIF(sim=sim, clk=Clock(freq=1), num_masters=0, num_slaves=1)

    def test_invalid_num_slaves(self) -> None:
        sim = Simulation()
        with pytest.raises(ValueError, match="num_slaves"):
            CrossbarIF(sim=sim, clk=Clock(freq=1), num_masters=1, num_slaves=0)

    def test_bitwidth_inferred_from_first_endpoint(self) -> None:
        sim = Simulation()
        xbar = CrossbarIF(sim=sim, clk=Clock(freq=1), num_masters=1, num_slaves=1)
        xbar.bind('master_0', CrossbarIFMaster(sim=sim, bitwidth=64))
        assert xbar.bitwidth == 64
        xbar.bind('slave_0', CrossbarIFSlave(sim=sim, bitwidth=64))

    def test_bitwidth_mismatch_raises(self) -> None:
        sim = Simulation()
        xbar = CrossbarIF(sim=sim, clk=Clock(freq=1), num_masters=1, num_slaves=1, bitwidth=32)
        with pytest.raises(ValueError, match="bitwidth"):
            xbar.bind('master_0', CrossbarIFMaster(sim=sim, bitwidth=64))

    def test_wrong_type_for_slave_port_raises(self) -> None:
        sim = Simulation()
        xbar = CrossbarIF(sim=sim, clk=Clock(freq=1), num_masters=1, num_slaves=1)
        with pytest.raises(TypeError, match="CrossbarIFSlave"):
            xbar.bind('slave_0', CrossbarIFMaster(sim=sim, bitwidth=32))

    def test_wrong_type_for_master_port_raises(self) -> None:
        sim = Simulation()
        xbar = CrossbarIF(sim=sim, clk=Clock(freq=1), num_masters=1, num_slaves=1)
        with pytest.raises(TypeError, match="CrossbarIFMaster"):
            xbar.bind('master_0', CrossbarIFSlave(sim=sim, bitwidth=32))

    def test_invalid_port_name_raises(self) -> None:
        sim, xbar, _, _ = _make_xbar()
        with pytest.raises(KeyError, match="bad_name"):
            xbar.bind('bad_name', CrossbarIFSlave(sim=sim, bitwidth=32))

    def test_double_bind_raises(self) -> None:
        sim, xbar, _, _ = _make_xbar()
        with pytest.raises(ValueError, match="already bound"):
            xbar.bind('slave_0', CrossbarIFSlave(sim=sim, bitwidth=32))

    def test_stream_endpoints_rejected(self) -> None:
        sim = Simulation()
        xbar = CrossbarIF(sim=sim, clk=Clock(freq=1), num_masters=1, num_slaves=1)
        with pytest.raises(TypeError):
            xbar.bind('master_0', StreamIFMaster(sim=sim, bitwidth=32))
        with pytest.raises(TypeError):
            xbar.bind('slave_0', StreamIFSlave(sim=sim, bitwidth=32))


# ---------------------------------------------------------------------------
# Simulation scenario
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class XbarCase:
    name: str
    num_masters: int
    num_slaves: int
    # (master_idx, words_tuple, dest_slave_idx) — each master's list is sent in order
    transfers: tuple[tuple[int, tuple[int, ...], int], ...]
    tx_gap: float
    queue_size: int | None
    rx_processing_delay: float
    # expected write-end timestamps (sorted ascending) for timing cases
    expected_write_end: tuple[float, ...]
    # expected received packets per slave: slave_idx → tuple of word tuples (in order)
    expected_rx: dict[int, tuple[tuple[int, ...], ...]]


class XbarScenario:
    """
    Drives all masters concurrently and collects timing/data results.

    Each master fires its transfers sequentially with tx_gap between them.
    Slaves run their run_proc loop throughout.
    """

    def __init__(self, case: XbarCase) -> None:
        self.case = case

        # Initialise accumulators before building rx_procs (closures capture self)
        self.received: dict[int, list[np.ndarray]] = {i: [] for i in range(case.num_slaves)}
        self.rx_times: dict[int, list[float]] = {i: [] for i in range(case.num_slaves)}
        self.write_end_times: list[float] = []

        rx_procs = {i: self._make_rx_proc(i) for i in range(case.num_slaves)}

        sim, xbar, masters, slaves = _make_xbar(
            num_masters=case.num_masters,
            num_slaves=case.num_slaves,
            freq=1.0,
            queue_size=case.queue_size,
            rx_procs=rx_procs,
        )
        self.sim = sim
        self.env = sim.env
        self.xbar = xbar
        self.masters = masters
        self.slaves = slaves

        # Group transfers by master index (preserving order)
        self.master_transfers: dict[int, list[tuple[np.ndarray, int]]] = {
            i: [] for i in range(case.num_masters)
        }
        for m_idx, words_t, dest in case.transfers:
            self.master_transfers[m_idx].append(
                (np.array(words_t, dtype=np.uint32), dest)
            )

    def _make_rx_proc(self, slave_idx: int):
        def rx_proc(words: Words):
            self.received[slave_idx].append(np.array(words, copy=True))
            self.rx_times[slave_idx].append(self.env.now)
            yield self.env.timeout(self.case.rx_processing_delay)
        return rx_proc

    def _master_proc(self, m_idx: int, done: simpy.events.Event):
        for packet, dest in self.master_transfers[m_idx]:
            yield self.env.process(self.masters[m_idx].write(packet, dest))
            self.write_end_times.append(self.env.now)
            yield self.env.timeout(self.case.tx_gap)
        done.succeed()

    def run(self) -> None:
        for slave in self.slaves:
            self.env.process(slave.run_proc())

        done_events = [self.env.event() for _ in range(self.case.num_masters)]
        for i in range(self.case.num_masters):
            self.env.process(self._master_proc(i, done_events[i]))

        self.env.run(until=simpy.events.AllOf(self.env, done_events))
        # Allow in-flight rx_proc calls to complete
        if self.case.rx_processing_delay > 0:
            self.env.run(until=self.env.now + self.case.rx_processing_delay + 1)


# ---------------------------------------------------------------------------
# Parametrised routing and timing tests
# ---------------------------------------------------------------------------

XBAR_CASES = [
    pytest.param(
        XbarCase(
            name="single_master_three_slaves_no_backpressure",
            num_masters=1,
            num_slaves=3,
            transfers=(
                (0, (10, 11, 12), 0),
                (0, (20, 21), 1),
                (0, (30,), 2),
            ),
            tx_gap=1.0,
            queue_size=None,
            rx_processing_delay=0.0,
            expected_write_end=(3.0, 6.0, 8.0),
            expected_rx={0: ((10, 11, 12),), 1: ((20, 21),), 2: ((30,),)},
        ),
        id="single-master-three-slaves",
    ),
    pytest.param(
        XbarCase(
            name="two_masters_separate_slaves",
            num_masters=2,
            num_slaves=2,
            transfers=(
                (0, (1, 2, 3), 0),   # master_0 → slave_0 (3 words, done t=3)
                (1, (4, 5), 1),       # master_1 → slave_1 (2 words, done t=2)
            ),
            tx_gap=0.0,
            queue_size=None,
            rx_processing_delay=0.0,
            expected_write_end=(2.0, 3.0),
            expected_rx={0: ((1, 2, 3),), 1: ((4, 5),)},
        ),
        id="two-masters-separate-slaves",
    ),
    pytest.param(
        XbarCase(
            name="deep_queue_absorbs_receiver_delay",
            num_masters=1,
            num_slaves=1,
            transfers=(
                (0, (1, 2), 0),
                (0, (3, 4), 0),
                (0, (5, 6), 0),
            ),
            tx_gap=0.0,
            queue_size=6,
            rx_processing_delay=5.0,
            expected_write_end=(2.0, 4.0, 6.0),
            expected_rx={0: ((1, 2), (3, 4), (5, 6))},
        ),
        id="deep-queue-absorbs-receiver-delay",
    ),
    pytest.param(
        XbarCase(
            name="shallow_queue_forces_backpressure",
            num_masters=1,
            num_slaves=1,
            transfers=(
                (0, (1, 2), 0),
                (0, (3, 4), 0),
                (0, (5, 6), 0),
            ),
            tx_gap=0.0,
            queue_size=2,
            rx_processing_delay=5.0,
            expected_write_end=(2.0, 4.0, 7.0),
            expected_rx={0: ((1, 2), (3, 4), (5, 6))},
        ),
        id="shallow-queue-forces-backpressure",
    ),
]


@pytest.mark.parametrize("case", XBAR_CASES)
def test_crossbar_data_integrity(case: XbarCase) -> None:
    """Every packet reaches the correct slave with correct data."""
    scenario = XbarScenario(case)
    scenario.run()

    for slave_idx, expected_packets in case.expected_rx.items():
        received = scenario.received[slave_idx]
        assert len(received) == len(expected_packets), (
            f"slave_{slave_idx}: expected {len(expected_packets)} packets, "
            f"got {len(received)}"
        )
        for rx, ex in zip(received, expected_packets, strict=True):
            npt.assert_array_equal(rx, np.array(ex, dtype=np.uint32))

    for slave in scenario.slaves:
        assert slave.nrx.level == 0
        assert slave.ntx.level == 0


@pytest.mark.parametrize("case", XBAR_CASES)
def test_crossbar_timing(case: XbarCase) -> None:
    """Transfer completion timestamps match expected values."""
    scenario = XbarScenario(case)
    scenario.run()
    assert sorted(scenario.write_end_times) == list(case.expected_write_end)


# ---------------------------------------------------------------------------
# Runtime error tests
# ---------------------------------------------------------------------------

class TestCrossbarRuntimeErrors:
    def test_write_to_unbound_slave_raises(self) -> None:
        sim = Simulation()
        xbar = CrossbarIF(sim=sim, clk=Clock(freq=1), num_masters=1, num_slaves=2, bitwidth=32)
        master = CrossbarIFMaster(sim=sim, bitwidth=32)
        slave0 = CrossbarIFSlave(sim=sim, bitwidth=32)
        xbar.bind('master_0', master)
        xbar.bind('slave_0', slave0)
        # slave_1 is intentionally left unbound

        caught: list[Exception] = []

        def proc():
            words = np.array([1, 2], dtype=np.uint32)
            try:
                yield sim.env.process(xbar.write(words, dest=1))
            except RuntimeError as exc:
                caught.append(exc)

        sim.env.process(proc())
        sim.env.run()
        assert len(caught) == 1
        assert "not bound" in str(caught[0])

    def test_write_out_of_range_dest_raises(self) -> None:
        sim, xbar, masters, slaves = _make_xbar(num_masters=1, num_slaves=2)

        caught: list[Exception] = []

        def proc():
            words = np.array([1], dtype=np.uint32)
            try:
                yield sim.env.process(xbar.write(words, dest=5))
            except ValueError as exc:
                caught.append(exc)

        sim.env.process(proc())
        sim.env.run()
        assert len(caught) == 1
        assert "out of range" in str(caught[0])

    def test_master_write_without_interface_raises(self) -> None:
        sim = Simulation()
        master = CrossbarIFMaster(sim=sim, bitwidth=32)

        caught: list[Exception] = []

        def proc():
            words = np.array([1], dtype=np.uint32)
            try:
                yield sim.env.process(master.write(words, dest=0))
            except RuntimeError as exc:
                caught.append(exc)

        sim.env.process(proc())
        sim.env.run()
        assert len(caught) == 1
        assert "not bound to an interface" in str(caught[0])
