"""
Unit tests for pysilicon/hw/aximm_queue.py.

Phase 1 coverage
----------------
AXIMMQueueLayout  — address math across mem_bw and elem_words, validation
MMMemory          — burst round-trip over a DirectMMIF
"""
from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from pysilicon.hw.aximm_queue import (
    AXIMMQueue,
    AXIMMQueueLayout,
    MMMemory,
    _split,
)
from pysilicon.hw.clock import Clock
from pysilicon.hw.memif import (
    AXIMMCrossBarIF,
    DirectMMIF,
    MMIFMaster,
    assign_address_ranges,
)
from pysilicon.simulation.simulation import Simulation


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_queue_direct(capacity, *, elem_words=1, mem_bw=32, base_addr=0x0):
    """One AXIMMQueue over a DirectMMIF + MMMemory; returns (sim, queue)."""
    sim = Simulation()
    clk = Clock(freq=1.0)
    mem = MMMemory(sim=sim, bitwidth=mem_bw)
    master = MMIFMaster(sim=sim, bitwidth=mem_bw)
    direct = DirectMMIF(sim=sim, clk=clk)
    direct.bind("master", master)
    direct.bind("slave", mem.slave_ep)
    layout = AXIMMQueueLayout(
        base_addr=base_addr, capacity=capacity, elem_words=elem_words, mem_bw=mem_bw
    )
    return sim, AXIMMQueue(master=master, layout=layout)


# ---------------------------------------------------------------------------
# AXIMMQueueLayout
# ---------------------------------------------------------------------------

class TestAXIMMQueueLayout:
    def test_basic_math_32bit(self):
        lay = AXIMMQueueLayout(base_addr=0x1000, capacity=8, elem_words=1, mem_bw=32)
        assert lay.word_bytes == 4
        assert lay.control_bytes == 16          # 4 control words * 4 bytes
        assert lay.head_addr == 0x1000
        assert lay.tail_addr == 0x1004
        assert lay.capacity_addr == 0x1008
        assert lay.data_base == 0x1010
        assert lay.slot_addr(0) == 0x1010
        assert lay.slot_addr(7) == 0x1010 + 7 * 4
        # 16 control bytes + 8 slots * 1 word * 4 bytes
        assert lay.total_bytes == 16 + 8 * 4

    def test_mem_bw_64_doubles_control_and_stride(self):
        """Regression: a hard-coded 0x04 / CONTROL_BYTES=16 would fail this."""
        lay32 = AXIMMQueueLayout(base_addr=0, capacity=8, mem_bw=32)
        lay64 = AXIMMQueueLayout(base_addr=0, capacity=8, mem_bw=64)
        assert lay64.word_bytes == 8
        assert lay64.control_bytes == 32        # double the 32-bit case
        assert lay64.control_bytes == 2 * lay32.control_bytes
        # tail is one word past head — stride doubles with mem_bw
        assert lay32.tail_addr == 4
        assert lay64.tail_addr == 8
        # data base sits after the (larger) control region
        assert lay64.data_base == 32
        assert lay64.slot_addr(1) - lay64.slot_addr(0) == 8

    def test_elem_words_scales_slot_stride(self):
        lay = AXIMMQueueLayout(base_addr=0, capacity=4, elem_words=4, mem_bw=32)
        assert lay.slot_addr(0) == lay.data_base
        # each slot is 4 words * 4 bytes = 16 bytes
        assert lay.slot_addr(1) - lay.slot_addr(0) == 16
        assert lay.slot_addr(3) == lay.data_base + 3 * 16
        assert lay.total_bytes == lay.control_bytes + 4 * 4 * 4

    def test_elem_words_4_mem_bw_64(self):
        lay = AXIMMQueueLayout(base_addr=0x40, capacity=4, elem_words=4, mem_bw=64)
        assert lay.word_bytes == 8
        assert lay.control_bytes == 32
        assert lay.data_base == 0x40 + 32
        # slot stride = elem_words(4) * word_bytes(8) = 32
        assert lay.slot_addr(1) - lay.slot_addr(0) == 32
        assert lay.total_bytes == 32 + 4 * 4 * 8

    def test_validation(self):
        with pytest.raises(ValueError, match="capacity"):
            AXIMMQueueLayout(base_addr=0, capacity=1)
        with pytest.raises(ValueError, match="elem_words"):
            AXIMMQueueLayout(base_addr=0, capacity=4, elem_words=0)
        with pytest.raises(ValueError, match="mem_bw"):
            AXIMMQueueLayout(base_addr=0, capacity=4, mem_bw=128)


# ---------------------------------------------------------------------------
# MMMemory
# ---------------------------------------------------------------------------

class TestMMMemory:
    def _make(self, bitwidth=32):
        sim = Simulation()
        clk = Clock(freq=1.0)
        mem = MMMemory(sim=sim, bitwidth=bitwidth)
        master = MMIFMaster(sim=sim, bitwidth=bitwidth)
        direct = DirectMMIF(sim=sim, clk=clk)
        direct.bind("master", master)
        direct.bind("slave", mem.slave_ep)
        return sim, master, mem

    def test_burst_round_trip(self):
        sim, master, mem = self._make()
        data = np.arange(6, dtype=np.uint32) + 100
        result = []

        def proc():
            yield from master.write(data, 0x0)
            got = yield from master.read(6, 0x0)
            result.append(got)

        sim.env.process(proc())
        sim.env.run()
        npt.assert_array_equal(result[0], data)

    def test_byte_stride_keys(self):
        """Words are stored at byte addresses spaced by bitwidth // 8."""
        sim, master, mem = self._make(bitwidth=32)

        def proc():
            yield from master.write(np.array([7, 8, 9], dtype=np.uint32), 0x10)

        sim.env.process(proc())
        sim.env.run()
        assert mem._mem[0x10] == 7
        assert mem._mem[0x14] == 8
        assert mem._mem[0x18] == 9

    def test_unwritten_reads_zero(self):
        sim, master, mem = self._make()
        result = []

        def proc():
            got = yield from master.read(3, 0x200)
            result.append(got)

        sim.env.process(proc())
        sim.env.run()
        npt.assert_array_equal(result[0], [0, 0, 0])

    def test_64bit_stride(self):
        sim, master, mem = self._make(bitwidth=64)

        def proc():
            yield from master.write(np.array([1, 2], dtype=np.uint64), 0x0)

        sim.env.process(proc())
        sim.env.run()
        assert mem._mem[0x0] == 1
        assert mem._mem[0x8] == 2   # stride 8 bytes for 64-bit


# ---------------------------------------------------------------------------
# _split wrap helper
# ---------------------------------------------------------------------------

class TestSplit:
    def test_no_wrap(self):
        assert _split(2, 3, 8) == [(2, 3)]

    def test_exact_end(self):
        assert _split(5, 3, 8) == [(5, 3)]

    def test_wrap(self):
        assert _split(6, 5, 8) == [(6, 2), (0, 3)]

    def test_full_wrap_from_zero(self):
        assert _split(0, 8, 8) == [(0, 8)]


# ---------------------------------------------------------------------------
# AXIMMQueue core (try_write / try_get), elem_words == 1
# ---------------------------------------------------------------------------

class TestAXIMMQueueCore:
    def test_mem_bw_mismatch_raises(self):
        sim = Simulation()
        master = MMIFMaster(sim=sim, bitwidth=32)
        layout = AXIMMQueueLayout(base_addr=0, capacity=8, mem_bw=64)
        with pytest.raises(ValueError, match="mem_bw"):
            AXIMMQueue(master=master, layout=layout)

    def test_fifo_order_many_cycles(self):
        sim, q = _make_queue_direct(capacity=8)
        out = []

        def proc():
            yield from q.reset()
            nxt = 0
            for _ in range(20):
                batch = np.arange(nxt, nxt + 3, dtype=np.uint32)
                ok = yield from q.try_write(batch)
                assert ok
                nxt += 3
                got = yield from q.try_get(3)
                out.append(np.array(got))

        sim.env.process(proc())
        sim.env.run()
        npt.assert_array_equal(np.concatenate(out), np.arange(60))

    def test_full_returns_false(self):
        sim, q = _make_queue_direct(capacity=8)   # usable depth 7
        result = {}

        def proc():
            yield from q.reset()
            ok = yield from q.try_write(np.arange(7, dtype=np.uint32))
            result["fill"] = ok
            result["space"] = yield from q.space()
            result["count"] = yield from q.count()
            # one more slot must not fit
            result["overflow"] = yield from q.try_write(np.array([99], dtype=np.uint32))

        sim.env.process(proc())
        sim.env.run()
        assert result["fill"] is True
        assert result["space"] == 0
        assert result["count"] == 7
        assert result["overflow"] is False

    def test_drain_to_empty(self):
        sim, q = _make_queue_direct(capacity=8)
        result = {}

        def proc():
            yield from q.reset()
            yield from q.try_write(np.arange(5, dtype=np.uint32))
            got = yield from q.try_get(5)
            result["drained"] = np.array(got)
            result["count"] = yield from q.count()
            # empty now: try_get returns empty
            empty = yield from q.try_get(3)
            result["empty_len"] = len(empty)

        sim.env.process(proc())
        sim.env.run()
        npt.assert_array_equal(result["drained"], np.arange(5))
        assert result["count"] == 0
        assert result["empty_len"] == 0

    def test_wrap_around_integrity(self):
        """capacity 8, repeatedly write 5 / get 5 forces head/tail past the end."""
        sim, q = _make_queue_direct(capacity=8)
        out = []

        def proc():
            yield from q.reset()
            nxt = 0
            for _ in range(10):
                ok = yield from q.try_write(np.arange(nxt, nxt + 5, dtype=np.uint32))
                assert ok
                got = yield from q.try_get(5)
                out.append(np.array(got))
                nxt += 5

        sim.env.process(proc())
        sim.env.run()
        npt.assert_array_equal(np.concatenate(out), np.arange(50))

    def test_partial_get_short(self):
        """try_get with max_slots > available returns only what's there."""
        sim, q = _make_queue_direct(capacity=8)
        result = {}

        def proc():
            yield from q.reset()
            yield from q.try_write(np.array([10, 11], dtype=np.uint32))
            got = yield from q.try_get(5)   # only 2 available
            result["got"] = np.array(got)

        sim.env.process(proc())
        sim.env.run()
        npt.assert_array_equal(result["got"], [10, 11])

    def test_space_count_track(self):
        sim, q = _make_queue_direct(capacity=8)
        trace = []

        def proc():
            yield from q.reset()
            for n in (1, 2, 3):
                yield from q.try_write(np.arange(n, dtype=np.uint32))
                c = yield from q.count()
                s = yield from q.space()
                trace.append((c, s))

        sim.env.process(proc())
        sim.env.run()
        # cumulative counts 1, 3, 6; space = 7 - count
        assert trace == [(1, 6), (3, 4), (6, 1)]

    def test_wrap_split_in_single_write(self):
        """A batch that itself straddles the wrap is stored correctly."""
        sim, q = _make_queue_direct(capacity=8)
        result = {}

        def proc():
            yield from q.reset()
            # advance tail to 6: write 6, get 6
            yield from q.try_write(np.arange(6, dtype=np.uint32))
            yield from q.try_get(6)
            # now head=tail=6; write 4 slots -> wraps [6,8)+[0,2)
            yield from q.try_write(np.array([100, 101, 102, 103], dtype=np.uint32))
            got = yield from q.try_get(4)
            result["got"] = np.array(got)

        sim.env.process(proc())
        sim.env.run()
        npt.assert_array_equal(result["got"], [100, 101, 102, 103])


# ---------------------------------------------------------------------------
# Blocking write/get + concurrent SPSC over a crossbar
# ---------------------------------------------------------------------------

class TestBlocking:
    def test_write_too_large_raises(self):
        sim, q = _make_queue_direct(capacity=8)   # usable depth 7

        def proc():
            yield from q.reset()
            yield from q.write(np.arange(8, dtype=np.uint32))   # 8 > 7

        sim.env.process(proc())
        with pytest.raises(ValueError, match="never fit"):
            sim.env.run()

    def test_blocking_write_get_single_process(self):
        sim, q = _make_queue_direct(capacity=4)   # usable depth 3
        result = {}

        def proc():
            yield from q.reset()
            yield from q.write(np.array([1, 2, 3], dtype=np.uint32))
            got = yield from q.get(3)
            result["got"] = np.array(got)

        sim.env.process(proc())
        sim.env.run()
        npt.assert_array_equal(result["got"], [1, 2, 3])


class TestConcurrentSPSC:
    def test_producer_consumer_crossbar(self):
        """Two masters, one shared region: producer blocks on full, consumer
        drains; data must arrive in order with no loss or duplication."""
        sim = Simulation()
        clk = Clock(freq=1.0)
        N = 100
        layout = AXIMMQueueLayout(base_addr=0x1000, capacity=8, mem_bw=32)

        mem = MMMemory(sim=sim, bitwidth=32)
        xbar = AXIMMCrossBarIF(
            sim=sim, clk=clk,
            nports_master=2, nports_slave=1, bitwidth=32,
            latency_init=1.0, latency_read_return=1.0,
        )
        prod_master = MMIFMaster(sim=sim, bitwidth=32)
        cons_master = MMIFMaster(sim=sim, bitwidth=32)
        xbar.bind("master_0", prod_master)
        xbar.bind("master_1", cons_master)
        xbar.bind("slave_0", mem.slave_ep)
        assign_address_ranges([mem.slave_ep], [(layout.base_addr, layout.total_bytes)])

        pq = AXIMMQueue(master=prod_master, layout=layout)
        cq = AXIMMQueue(master=cons_master, layout=layout)

        data = np.arange(N, dtype=np.uint32)
        result = {}

        def producer():
            yield from pq.reset()
            for i in range(0, N, 4):
                yield from pq.write(data[i:i + 4], poll_interval=1.0)

        def consumer():
            got = yield from cq.get(N, poll_interval=1.0)
            result["got"] = np.array(got)

        sim.env.process(producer())
        sim.env.process(consumer())
        sim.env.run()

        npt.assert_array_equal(result["got"], data)

    def test_consumer_starts_before_producer(self):
        """Consumer that begins polling on an empty queue still receives all
        data once the producer runs."""
        sim = Simulation()
        clk = Clock(freq=1.0)
        N = 30
        layout = AXIMMQueueLayout(base_addr=0x0, capacity=6, mem_bw=32)

        mem = MMMemory(sim=sim, bitwidth=32)
        xbar = AXIMMCrossBarIF(
            sim=sim, clk=clk, nports_master=2, nports_slave=1, bitwidth=32,
        )
        prod_master = MMIFMaster(sim=sim, bitwidth=32)
        cons_master = MMIFMaster(sim=sim, bitwidth=32)
        xbar.bind("master_0", prod_master)
        xbar.bind("master_1", cons_master)
        xbar.bind("slave_0", mem.slave_ep)
        assign_address_ranges([mem.slave_ep], [(layout.base_addr, layout.total_bytes)])

        pq = AXIMMQueue(master=prod_master, layout=layout)
        cq = AXIMMQueue(master=cons_master, layout=layout)
        data = np.arange(N, dtype=np.uint32) + 1000
        result = {}

        def producer():
            yield from pq.reset()
            yield pq.master.timeout(5.0)   # let the consumer poll an empty queue first
            for i in range(0, N, 5):
                yield from pq.write(data[i:i + 5], poll_interval=0.5)

        def consumer():
            got = yield from cq.get(N, poll_interval=0.5)
            result["got"] = np.array(got)

        sim.env.process(producer())
        sim.env.process(consumer())
        sim.env.run()
        npt.assert_array_equal(result["got"], data)
