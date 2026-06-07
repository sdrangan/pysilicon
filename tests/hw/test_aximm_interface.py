"""
Unit tests for waveflow/hw/aximm.py.

Coverage
--------
AXIMMAddressRange         — contains, to_local
assign_address_ranges     — basic assignment, overlap detection, length mismatch
AXIMMCrossBarIF.bind      — type checking, range checking, bitwidth, double-bind
write (FULL)              — data delivery, latency model
write (LITE)              — per-word splitting, correct local addresses
read  (FULL)              — data return, latency model
read  (LITE)              — per-word sequential reads, correct return value
Address out of range      — RuntimeError on write and read
Concurrent masters        — serialization on same slave bus
Validation                — missing clock, missing binding
DirectMMIF                — write, read, latency
MMIFMaster schema helpers — read_schema, write_schema, read_array, write_array
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.testing as npt
import pytest

from waveflow.hw.aximm import (
    AXIMMAddressRange,
    AXIMMCrossBarIF,
    AXIMMProtocol,
    DirectMMIF,
    MMIFMaster,
    MMIFSlave,
    assign_address_ranges,
    Words,
)
from waveflow.hw.clock import Clock
from waveflow.simulation.simobj import ProcessGen
from waveflow.simulation.simulation import Simulation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sim() -> Simulation:
    return Simulation()


def _make_clk(freq: float = 1.0) -> Clock:
    return Clock(freq=freq)


def _make_xbar(
    sim: Simulation,
    clk: Clock,
    *,
    nports_master: int = 1,
    nports_slave: int = 1,
    bitwidth: int = 32,
    latency_init: float = 0.0,
    latency_read_return: float = 0.0,
) -> AXIMMCrossBarIF:
    return AXIMMCrossBarIF(
        sim=sim,
        clk=clk,
        nports_master=nports_master,
        nports_slave=nports_slave,
        bitwidth=bitwidth,
        latency_init=latency_init,
        latency_read_return=latency_read_return,
    )


# ---------------------------------------------------------------------------
# AXIMMAddressRange
# ---------------------------------------------------------------------------

class TestAXIMMAddressRange:
    def test_contains_in_range(self):
        ar = AXIMMAddressRange(base_addr=0x1000, size=0x100)
        assert ar.contains(0x1000)
        assert ar.contains(0x10FF)
        assert ar.contains(0x1050)

    def test_contains_out_of_range(self):
        ar = AXIMMAddressRange(base_addr=0x1000, size=0x100)
        assert not ar.contains(0x0FFF)
        assert not ar.contains(0x1100)

    def test_to_local(self):
        ar = AXIMMAddressRange(base_addr=0x1000, size=0x100)
        assert ar.to_local(0x1000) == 0
        assert ar.to_local(0x1040) == 0x40
        assert ar.to_local(0x10FF) == 0xFF


# ---------------------------------------------------------------------------
# assign_address_ranges
# ---------------------------------------------------------------------------

class TestAssignAddressRanges:
    def _make_slave(self, sim: Simulation) -> MMIFSlave:
        return MMIFSlave(sim=sim, bitwidth=32)

    def test_basic_assignment(self):
        sim = _make_sim()
        s0 = self._make_slave(sim)
        s1 = self._make_slave(sim)
        ranges = assign_address_ranges([s0, s1], [(0x0000, 0x1000), (0x1000, 0x1000)])
        assert s0.addr_range == AXIMMAddressRange(0x0000, 0x1000)
        assert s1.addr_range == AXIMMAddressRange(0x1000, 0x1000)
        assert len(ranges) == 2

    def test_length_mismatch(self):
        sim = _make_sim()
        s0 = self._make_slave(sim)
        with pytest.raises(ValueError, match="ranges"):
            assign_address_ranges([s0], [(0x0000, 0x100), (0x1000, 0x100)])

    def test_overlap_raises(self):
        sim = _make_sim()
        s0 = self._make_slave(sim)
        s1 = self._make_slave(sim)
        with pytest.raises(ValueError, match="overlap"):
            assign_address_ranges([s0, s1], [(0x0000, 0x200), (0x0100, 0x200)])

    def test_adjacent_ranges_ok(self):
        sim = _make_sim()
        s0 = self._make_slave(sim)
        s1 = self._make_slave(sim)
        # [0, 0x100) and [0x100, 0x200) are adjacent, not overlapping
        assign_address_ranges([s0, s1], [(0x0000, 0x100), (0x0100, 0x100)])
        assert s0.addr_range.contains(0x00FF)
        assert not s0.addr_range.contains(0x0100)
        assert s1.addr_range.contains(0x0100)


# ---------------------------------------------------------------------------
# AXIMMCrossBarIF.bind validation
# ---------------------------------------------------------------------------

class TestBind:
    def test_wrong_type_master_slot(self):
        sim = _make_sim()
        xbar = _make_xbar(sim, _make_clk(), nports_master=1, nports_slave=1)
        wrong = MMIFSlave(sim=sim, bitwidth=32)
        with pytest.raises(TypeError):
            xbar.bind("master_0", wrong)

    def test_wrong_type_slave_slot(self):
        sim = _make_sim()
        xbar = _make_xbar(sim, _make_clk(), nports_master=1, nports_slave=1)
        wrong = MMIFMaster(sim=sim, bitwidth=32)
        with pytest.raises(TypeError):
            xbar.bind("slave_0", wrong)

    def test_master_out_of_range(self):
        sim = _make_sim()
        xbar = _make_xbar(sim, _make_clk(), nports_master=1, nports_slave=1)
        ep = MMIFMaster(sim=sim, bitwidth=32)
        with pytest.raises(KeyError):
            xbar.bind("master_5", ep)

    def test_slave_out_of_range(self):
        sim = _make_sim()
        xbar = _make_xbar(sim, _make_clk(), nports_master=1, nports_slave=1)
        ep = MMIFSlave(sim=sim, bitwidth=32)
        with pytest.raises(KeyError):
            xbar.bind("slave_5", ep)

    def test_invalid_ep_name(self):
        sim = _make_sim()
        xbar = _make_xbar(sim, _make_clk())
        ep = MMIFMaster(sim=sim, bitwidth=32)
        with pytest.raises(KeyError):
            xbar.bind("port_0", ep)

    def test_bitwidth_mismatch(self):
        sim = _make_sim()
        xbar = _make_xbar(sim, _make_clk(), bitwidth=32)
        ep = MMIFMaster(sim=sim, bitwidth=64)
        with pytest.raises(ValueError, match="bitwidth"):
            xbar.bind("master_0", ep)

    def test_double_bind_raises(self):
        sim = _make_sim()
        xbar = _make_xbar(sim, _make_clk())
        ep1 = MMIFMaster(sim=sim, bitwidth=32)
        ep2 = MMIFMaster(sim=sim, bitwidth=32)
        xbar.bind("master_0", ep1)
        with pytest.raises(ValueError):
            xbar.bind("master_0", ep2)

    def test_missing_clock(self):
        sim = _make_sim()
        with pytest.raises(ValueError, match="clock"):
            AXIMMCrossBarIF(sim=sim, nports_master=1, nports_slave=1)

    def test_master_port_index_set_on_bind(self):
        sim = _make_sim()
        xbar = _make_xbar(sim, _make_clk(), nports_master=2, nports_slave=1)
        ep0 = MMIFMaster(sim=sim, bitwidth=32)
        ep1 = MMIFMaster(sim=sim, bitwidth=32)
        xbar.bind("master_0", ep0)
        xbar.bind("master_1", ep1)
        assert ep0.master_port == 0
        assert ep1.master_port == 1

    def test_protocol_stored_per_slave(self):
        """Protocol kwarg on bind() is stored per slave port."""
        sim = _make_sim()
        xbar = _make_xbar(sim, _make_clk(), nports_master=1, nports_slave=2)
        s0 = MMIFSlave(sim=sim, bitwidth=32)
        s1 = MMIFSlave(sim=sim, bitwidth=32)
        xbar.bind("slave_0", s0)                                   # default FULL
        xbar.bind("slave_1", s1, protocol=AXIMMProtocol.LITE)
        assert xbar._slave_protocols["slave_0"] == AXIMMProtocol.FULL
        assert xbar._slave_protocols["slave_1"] == AXIMMProtocol.LITE


# ---------------------------------------------------------------------------
# Simulation harness
# ---------------------------------------------------------------------------

class AXIMMHarness:
    """
    Minimal test harness: one master, one or two slaves, configurable latency.
    """

    def __init__(
        self,
        *,
        protocol: AXIMMProtocol = AXIMMProtocol.FULL,
        latency_init: float = 0.0,
        latency_read_return: float = 0.0,
        latency_per_word: float = 2.0,
        clk_freq: float = 1.0,
        nports_slave: int = 1,
        nports_master: int = 1,
    ) -> None:
        self.sim = Simulation()
        self.env = self.sim.env
        self.clk = Clock(freq=clk_freq)

        self.xbar = AXIMMCrossBarIF(
            sim=self.sim,
            clk=self.clk,
            nports_master=nports_master,
            nports_slave=nports_slave,
            bitwidth=32,
            latency_init=latency_init,
            latency_read_return=latency_read_return,
        )

        # Per-slave storage and per-word write-call log
        self._mem: list[dict[int, int]] = [{} for _ in range(nports_slave)]
        self.write_calls: list[list[tuple[np.ndarray, int]]] = [[] for _ in range(nports_slave)]
        self.read_calls:  list[list[tuple[int, int]]]       = [[] for _ in range(nports_slave)]

        self.slave_eps: list[MMIFSlave] = []
        for j in range(nports_slave):
            ep = MMIFSlave(
                sim=self.sim,
                bitwidth=32,
                rx_write_proc=self._make_write_proc(j),
                rx_read_proc=self._make_read_proc(j),
                latency_per_word=latency_per_word,
            )
            self.slave_eps.append(ep)
            self.xbar.bind(f"slave_{j}", ep, protocol=protocol)

        self.master_eps: list[MMIFMaster] = []
        for i in range(nports_master):
            ep = MMIFMaster(sim=self.sim, bitwidth=32)
            self.master_eps.append(ep)
            self.xbar.bind(f"master_{i}", ep)

    def _make_write_proc(self, slave_idx: int):
        def rx_write(words: Words, local_addr: int):
            self.write_calls[slave_idx].append((np.array(words, copy=True), local_addr))
            word_bytes = 4
            for i, w in enumerate(words):
                self._mem[slave_idx][local_addr + i * word_bytes] = int(w)
            yield self.env.timeout(0)
        return rx_write

    def _make_read_proc(self, slave_idx: int):
        def rx_read(nwords: int, local_addr: int):
            self.read_calls[slave_idx].append((nwords, local_addr))
            yield self.env.timeout(0)
            word_bytes = 4
            return np.array(
                [self._mem[slave_idx].get(local_addr + i * word_bytes, 0xDEAD)
                 for i in range(nwords)],
                dtype=np.uint32,
            )
        return rx_read

    def assign(self, ranges: list[tuple[int, int]]) -> None:
        assign_address_ranges(self.slave_eps, ranges)

    def run_proc(self, proc_fn, until: float | None = None):
        """Start proc_fn() as a simpy process and run until it completes."""
        done = self.env.event()

        def _wrap():
            yield from proc_fn()
            done.succeed()

        self.env.process(_wrap())
        target = done if until is None else self.env.timeout(until)
        self.env.run(until=target)


# ---------------------------------------------------------------------------
# FULL write tests
# ---------------------------------------------------------------------------

class TestFullWrite:
    def test_data_delivered(self):
        sc = AXIMMHarness(protocol=AXIMMProtocol.FULL)
        sc.assign([(0x0000, 0x1000)])

        words = np.array([10, 20, 30], dtype=np.uint32)

        def proc():
            yield sc.env.process(sc.master_eps[0].write(words, 0x0000))

        sc.run_proc(proc)
        assert len(sc.write_calls[0]) == 1
        npt.assert_array_equal(sc.write_calls[0][0][0], words)
        assert sc.write_calls[0][0][1] == 0   # local_addr == 0

    def test_local_addr_remapping(self):
        sc = AXIMMHarness(protocol=AXIMMProtocol.FULL)
        sc.assign([(0x2000, 0x1000)])

        words = np.array([1, 2], dtype=np.uint32)

        def proc():
            yield sc.env.process(sc.master_eps[0].write(words, 0x2010))

        sc.run_proc(proc)
        assert sc.write_calls[0][0][1] == 0x10

    def test_write_latency(self):
        """Transfer completes after (latency_init + nwords) / freq seconds."""
        sc = AXIMMHarness(protocol=AXIMMProtocol.FULL, latency_init=2.0, clk_freq=1.0)
        sc.assign([(0x0000, 0x1000)])

        words = np.array([1, 2, 3], dtype=np.uint32)  # nwords=3
        t_done: list[float] = []

        def proc():
            yield sc.env.process(sc.master_eps[0].write(words, 0x0000))
            t_done.append(sc.env.now)

        sc.run_proc(proc)
        # Expected: (2 + 3) / 1.0 = 5.0 seconds
        assert t_done[0] == pytest.approx(5.0)

    def test_out_of_range_raises(self):
        sc = AXIMMHarness(protocol=AXIMMProtocol.FULL)
        sc.assign([(0x0000, 0x100)])

        words = np.array([1], dtype=np.uint32)
        errors: list[Exception] = []

        def proc():
            try:
                yield sc.env.process(sc.master_eps[0].write(words, 0x9999))
            except RuntimeError as e:
                errors.append(e)

        sc.run_proc(proc)
        assert len(errors) == 1
        assert "0x00009999" in str(errors[0]).lower() or "9999" in str(errors[0])


# ---------------------------------------------------------------------------
# LITE write tests
# ---------------------------------------------------------------------------

class TestLiteWrite:
    def test_single_word_not_split(self):
        sc = AXIMMHarness(protocol=AXIMMProtocol.LITE, latency_per_word=2.0)
        sc.assign([(0x0000, 0x100)])

        words = np.array([0xFF], dtype=np.uint32)

        def proc():
            yield sc.env.process(sc.master_eps[0].write(words, 0x0000))

        sc.run_proc(proc)
        assert len(sc.write_calls[0]) == 1

    def test_multi_word_split(self):
        """n-word LITE write → n separate rx_write_proc calls, one word each."""
        sc = AXIMMHarness(protocol=AXIMMProtocol.LITE, latency_per_word=2.0)
        sc.assign([(0x0000, 0x100)])

        words = np.array([0xA, 0xB, 0xC], dtype=np.uint32)

        def proc():
            yield sc.env.process(sc.master_eps[0].write(words, 0x0000))

        sc.run_proc(proc)
        assert len(sc.write_calls[0]) == 3
        for i, (w, addr) in enumerate(sc.write_calls[0]):
            assert len(w) == 1
            assert int(w[0]) == [0xA, 0xB, 0xC][i]
            assert addr == i * 4   # word_bytes=4, auto-increment

    def test_lite_write_latency(self):
        """3 words × latency_per_word=3.0 / freq=1.0 → 9.0 s."""
        sc = AXIMMHarness(protocol=AXIMMProtocol.LITE, latency_per_word=3.0, clk_freq=1.0)
        sc.assign([(0x0000, 0x100)])

        words = np.array([1, 2, 3], dtype=np.uint32)
        t_done: list[float] = []

        def proc():
            yield sc.env.process(sc.master_eps[0].write(words, 0x0000))
            t_done.append(sc.env.now)

        sc.run_proc(proc)
        assert t_done[0] == pytest.approx(9.0)

    def test_lite_addr_increments(self):
        sc = AXIMMHarness(protocol=AXIMMProtocol.LITE, latency_per_word=2.0)
        sc.assign([(0x1000, 0x100)])

        words = np.array([1, 2, 3, 4], dtype=np.uint32)

        def proc():
            yield sc.env.process(sc.master_eps[0].write(words, 0x1008))

        sc.run_proc(proc)
        addrs = [addr for _, addr in sc.write_calls[0]]
        # local_addr starts at 0x1008 - 0x1000 = 0x08; increments by 4 each word
        assert addrs == [0x08, 0x0C, 0x10, 0x14]


# ---------------------------------------------------------------------------
# FULL read tests
# ---------------------------------------------------------------------------

class TestFullRead:
    def test_data_returned(self):
        sc = AXIMMHarness(protocol=AXIMMProtocol.FULL)
        sc.assign([(0x0000, 0x1000)])

        # Pre-populate memory
        sc._mem[0][0] = 42
        sc._mem[0][4] = 99

        result: list[np.ndarray] = []

        def proc():
            p = sc.env.process(sc.master_eps[0].read(2, 0x0000))
            yield p
            result.append(p.value)

        sc.run_proc(proc)
        assert len(result) == 1
        npt.assert_array_equal(result[0], [42, 99])

    def test_read_latency(self):
        """Total time = latency_init/freq + slave_access(0) + (latency_read_return + nwords)/freq."""
        sc = AXIMMHarness(
            protocol=AXIMMProtocol.FULL,
            latency_init=3.0,
            latency_read_return=2.0,
            clk_freq=1.0,
        )
        sc.assign([(0x0000, 0x1000)])
        t_done: list[float] = []

        def proc():
            p = sc.env.process(sc.master_eps[0].read(4, 0x0000))
            yield p
            t_done.append(sc.env.now)

        sc.run_proc(proc)
        # 3 (req wire) + 0 (slave, instant) + (2 + 4) (ret wire+burst) = 9.0 s
        assert t_done[0] == pytest.approx(9.0)

    def test_read_out_of_range_raises(self):
        sc = AXIMMHarness(protocol=AXIMMProtocol.FULL)
        sc.assign([(0x0000, 0x100)])
        errors: list[Exception] = []

        def proc():
            try:
                p = sc.env.process(sc.master_eps[0].read(1, 0x8000))
                yield p
            except RuntimeError as e:
                errors.append(e)

        sc.run_proc(proc)
        assert len(errors) == 1

    def test_read_local_addr_remapping(self):
        sc = AXIMMHarness(protocol=AXIMMProtocol.FULL)
        sc.assign([(0x4000, 0x100)])

        def proc():
            p = sc.env.process(sc.master_eps[0].read(1, 0x4020))
            yield p

        sc.run_proc(proc)
        assert sc.read_calls[0][0] == (1, 0x20)


# ---------------------------------------------------------------------------
# LITE read tests
# ---------------------------------------------------------------------------

class TestLiteRead:
    def test_data_returned(self):
        sc = AXIMMHarness(protocol=AXIMMProtocol.LITE, latency_per_word=2.0)
        sc.assign([(0x0000, 0x100)])

        sc._mem[0][0] = 7
        sc._mem[0][4] = 8
        sc._mem[0][8] = 9

        result: list[np.ndarray] = []

        def proc():
            p = sc.env.process(sc.master_eps[0].read(3, 0x0000))
            yield p
            result.append(p.value)

        sc.run_proc(proc)
        npt.assert_array_equal(result[0], [7, 8, 9])

    def test_lite_read_latency(self):
        """3 words × latency_per_word=4.0 / freq=1.0 → 12.0 s."""
        sc = AXIMMHarness(protocol=AXIMMProtocol.LITE, latency_per_word=4.0, clk_freq=1.0)
        sc.assign([(0x0000, 0x100)])
        t_done: list[float] = []

        def proc():
            p = sc.env.process(sc.master_eps[0].read(3, 0x0000))
            yield p
            t_done.append(sc.env.now)

        sc.run_proc(proc)
        assert t_done[0] == pytest.approx(12.0)

    def test_lite_read_addr_increments(self):
        sc = AXIMMHarness(protocol=AXIMMProtocol.LITE, latency_per_word=2.0)
        sc.assign([(0x2000, 0x100)])

        def proc():
            p = sc.env.process(sc.master_eps[0].read(3, 0x2008))
            yield p

        sc.run_proc(proc)
        addrs = [addr for _, addr in sc.read_calls[0]]
        assert addrs == [0x08, 0x0C, 0x10]

    def test_lite_per_word_read_calls(self):
        """n-word LITE read → n separate rx_read_proc calls."""
        sc = AXIMMHarness(protocol=AXIMMProtocol.LITE, latency_per_word=2.0)
        sc.assign([(0x0000, 0x100)])

        def proc():
            p = sc.env.process(sc.master_eps[0].read(4, 0x0000))
            yield p

        sc.run_proc(proc)
        assert len(sc.read_calls[0]) == 4
        for nwords, _ in sc.read_calls[0]:
            assert nwords == 1


# ---------------------------------------------------------------------------
# Multi-slave routing
# ---------------------------------------------------------------------------

class TestMultiSlaveRouting:
    def test_writes_reach_correct_slave(self):
        sc = AXIMMHarness(protocol=AXIMMProtocol.FULL, nports_slave=2)
        sc.assign([(0x0000, 0x1000), (0x1000, 0x1000)])

        w0 = np.array([0xAA], dtype=np.uint32)
        w1 = np.array([0xBB], dtype=np.uint32)

        def proc():
            yield sc.env.process(sc.master_eps[0].write(w0, 0x0000))
            yield sc.env.process(sc.master_eps[0].write(w1, 0x1000))

        sc.run_proc(proc)
        assert len(sc.write_calls[0]) == 1
        assert len(sc.write_calls[1]) == 1
        assert int(sc.write_calls[0][0][0][0]) == 0xAA
        assert int(sc.write_calls[1][0][0][0]) == 0xBB

    def test_reads_reach_correct_slave(self):
        sc = AXIMMHarness(protocol=AXIMMProtocol.FULL, nports_slave=2)
        sc.assign([(0x0000, 0x1000), (0x1000, 0x1000)])
        sc._mem[0][0] = 11
        sc._mem[1][0] = 22

        results: list[int] = []

        def proc():
            p0 = sc.env.process(sc.master_eps[0].read(1, 0x0000))
            yield p0
            results.append(int(p0.value[0]))
            p1 = sc.env.process(sc.master_eps[0].read(1, 0x1000))
            yield p1
            results.append(int(p1.value[0]))

        sc.run_proc(proc)
        assert results == [11, 22]


# ---------------------------------------------------------------------------
# Concurrent masters
# ---------------------------------------------------------------------------

class TestConcurrentMasters:
    def test_two_masters_serialize_on_same_slave(self):
        """Two masters writing to the same slave at the same time should both succeed."""
        sc = AXIMMHarness(
            protocol=AXIMMProtocol.FULL,
            nports_master=2,
            nports_slave=1,
            latency_init=0.0,
        )
        sc.assign([(0x0000, 0x2000)])

        w0 = np.array([0xAA, 0xBB], dtype=np.uint32)
        w1 = np.array([0xCC, 0xDD], dtype=np.uint32)
        done0 = sc.env.event()
        done1 = sc.env.event()

        def proc0():
            yield sc.env.process(sc.master_eps[0].write(w0, 0x0000))
            done0.succeed()

        def proc1():
            yield sc.env.process(sc.master_eps[1].write(w1, 0x0100))
            done1.succeed()

        sc.env.process(proc0())
        sc.env.process(proc1())
        sc.env.run(until=sc.env.all_of([done0, done1]))

        # Both writes must have completed
        assert len(sc.write_calls[0]) == 2

    def test_two_masters_independent_slaves_parallel(self):
        """Masters targeting different slaves can proceed independently."""
        sc = AXIMMHarness(
            protocol=AXIMMProtocol.FULL,
            nports_master=2,
            nports_slave=2,
            latency_init=5.0,
            clk_freq=1.0,
        )
        sc.assign([(0x0000, 0x1000), (0x1000, 0x1000)])

        words = np.array([1], dtype=np.uint32)
        t0_done: list[float] = []
        t1_done: list[float] = []

        def proc0():
            yield sc.env.process(sc.master_eps[0].write(words, 0x0000))
            t0_done.append(sc.env.now)

        def proc1():
            yield sc.env.process(sc.master_eps[1].write(words, 0x1000))
            t1_done.append(sc.env.now)

        sc.env.process(proc0())
        sc.env.process(proc1())
        sc.env.run(until=sc.env.timeout(20))

        # Both should finish at the same (latency_init+1)/freq=6.0
        assert t0_done[0] == pytest.approx(6.0)
        assert t1_done[0] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Unbound master
# ---------------------------------------------------------------------------

class TestUnbound:
    def test_write_unbound_raises(self):
        sim = _make_sim()
        ep = MMIFMaster(sim=sim, bitwidth=32)
        words = np.array([1], dtype=np.uint32)
        errors: list[Exception] = []

        def proc():
            try:
                yield from ep.write(words, 0x0)
            except RuntimeError as e:
                errors.append(e)

        sim.env.process(proc())
        sim.env.run()
        assert len(errors) == 1
        assert "not bound" in str(errors[0]).lower()

    def test_read_unbound_raises(self):
        sim = _make_sim()
        ep = MMIFMaster(sim=sim, bitwidth=32)
        errors: list[Exception] = []

        def proc():
            try:
                yield from ep.read(1, 0x0)
            except RuntimeError as e:
                errors.append(e)

        sim.env.process(proc())
        sim.env.run()
        assert len(errors) == 1


# ---------------------------------------------------------------------------
# DirectMMIF tests
# ---------------------------------------------------------------------------

class TestDirectMMIF:
    def _make_direct(self, latency_write=0., latency_read=0., latency_read_return=0.):
        sim = _make_sim()
        clk = _make_clk(freq=1.0)
        mem: dict[int, int] = {}
        write_calls: list[tuple] = []
        read_calls:  list[tuple] = []

        def rx_write(words: Words, addr: int):
            write_calls.append((np.array(words, copy=True), addr))
            for i, w in enumerate(words):
                mem[addr + i] = int(w)
            yield sim.env.timeout(0)

        def rx_read(nwords: int, addr: int):
            read_calls.append((nwords, addr))
            yield sim.env.timeout(0)
            return np.array([mem.get(addr + i, 0) for i in range(nwords)], dtype=np.uint32)

        master_ep = MMIFMaster(sim=sim, bitwidth=32)
        slave_ep  = MMIFSlave(sim=sim, bitwidth=32, rx_write_proc=rx_write, rx_read_proc=rx_read)

        direct = DirectMMIF(
            sim=sim, clk=clk,
            latency_write=latency_write,
            latency_read=latency_read,
            latency_read_return=latency_read_return,
        )
        direct.bind("master", master_ep)
        direct.bind("slave",  slave_ep)

        return sim, sim.env, master_ep, slave_ep, direct, mem, write_calls, read_calls

    def test_wrong_type_master(self):
        sim = _make_sim()
        direct = DirectMMIF(sim=sim, clk=_make_clk())
        with pytest.raises(TypeError):
            direct.bind("master", MMIFSlave(sim=sim, bitwidth=32))

    def test_wrong_type_slave(self):
        sim = _make_sim()
        direct = DirectMMIF(sim=sim, clk=_make_clk())
        with pytest.raises(TypeError):
            direct.bind("slave", MMIFMaster(sim=sim, bitwidth=32))

    def test_invalid_ep_name(self):
        sim = _make_sim()
        direct = DirectMMIF(sim=sim, clk=_make_clk())
        with pytest.raises(KeyError):
            direct.bind("port_0", MMIFMaster(sim=sim, bitwidth=32))

    def test_write_data_delivered(self):
        sim, env, master, _, _, mem, write_calls, _ = self._make_direct()
        words = np.array([10, 20, 30], dtype=np.uint32)

        def proc():
            yield env.process(master.write(words, 0))

        env.process(proc())
        env.run()
        assert len(write_calls) == 1
        npt.assert_array_equal(write_calls[0][0], words)
        assert write_calls[0][1] == 0

    def test_read_data_returned(self):
        sim, env, master, _, _, mem, _, read_calls = self._make_direct()
        mem[0] = 42
        mem[1] = 99
        results: list[np.ndarray] = []

        def proc():
            p = env.process(master.read(2, 0))
            yield p
            results.append(p.value)

        env.process(proc())
        env.run()
        npt.assert_array_equal(results[0], [42, 99])

    def test_write_latency(self):
        sim, env, master, _, _, _, _, _ = self._make_direct(latency_write=3.0)
        t_done: list[float] = []
        words = np.array([1], dtype=np.uint32)

        def proc():
            yield env.process(master.write(words, 0))
            t_done.append(env.now)

        env.process(proc())
        env.run()
        assert t_done[0] == pytest.approx(3.0)

    def test_read_latency(self):
        sim, env, master, _, _, _, _, _ = self._make_direct(
            latency_read=2.0, latency_read_return=1.0
        )
        t_done: list[float] = []

        def proc():
            p = env.process(master.read(3, 0))  # nwords=3 adds 3 to return leg
            yield p
            t_done.append(env.now)

        env.process(proc())
        env.run()
        # 2 (request) + 0 (slave instant) + (1+3) (return) = 6.0
        assert t_done[0] == pytest.approx(6.0)

    def test_address_passed_directly(self):
        """DirectMMIF passes the raw address to the slave callback."""
        sim, env, master, _, _, _, write_calls, _ = self._make_direct()
        words = np.array([7], dtype=np.uint32)

        def proc():
            yield env.process(master.write(words, 0x1234))

        env.process(proc())
        env.run()
        assert write_calls[0][1] == 0x1234


# ---------------------------------------------------------------------------
# MMIFMaster schema helpers
# ---------------------------------------------------------------------------

class TestMMIFMasterSchemaHelpers:
    def _make_harness(self):
        sim = _make_sim()
        clk = _make_clk(freq=1.0)
        mem: dict[int, int] = {}

        def rx_write(words: Words, addr: int):
            for i, w in enumerate(words):
                mem[addr + i] = int(w)
            yield sim.env.timeout(0)

        def rx_read(nwords: int, addr: int):
            yield sim.env.timeout(0)
            return np.array([mem.get(addr + i, 0) for i in range(nwords)], dtype=np.uint32)

        master_ep = MMIFMaster(sim=sim, bitwidth=32)
        slave_ep  = MMIFSlave(sim=sim, bitwidth=32, rx_write_proc=rx_write, rx_read_proc=rx_read)
        direct    = DirectMMIF(sim=sim, clk=clk)
        direct.bind("master", master_ep)
        direct.bind("slave",  slave_ep)
        return sim, sim.env, master_ep, mem

    def test_write_schema_read_schema(self):
        from waveflow.hw.dataschema import DataList, IntField
        Uint32 = IntField.specialize(bitwidth=32, signed=False)

        class TwoWords(DataList):
            elements = {"a": Uint32, "b": Uint32}

        sim, env, master, mem = self._make_harness()
        result = []

        def proc():
            obj = TwoWords(a=0xAA, b=0xBB)
            yield from master.write_schema(obj, addr=0)
            recovered = yield from master.read_schema(TwoWords, addr=0)
            result.append(recovered)

        env.process(proc())
        env.run()
        assert int(result[0].a) == 0xAA
        assert int(result[0].b) == 0xBB

    def test_write_array_read_array_float32(self):
        from waveflow.hw.dataschema import FloatField
        F32 = FloatField.specialize(bitwidth=32)

        sim, env, master, _ = self._make_harness()
        values = np.array([1.0, 2.5, -3.14], dtype=np.float32)
        result = []

        def proc():
            yield from master.write_array(values, F32, addr=0)
            arr = yield from master.read_array(F32, count=3, addr=0)
            result.append(arr)

        env.process(proc())
        env.run()
        assert isinstance(result[0], np.ndarray)
        assert result[0].dtype == np.float32
        assert np.allclose(result[0], values, atol=1e-6)
