"""
aximm_demo.py — AXI-MM crossbar demonstration.

Topology
--------
  CPU (master_0) ──┐
                   ├── AXIMMCrossBarIF ──┬── MemBank  (slave_0, FULL, 0x0000–0x0FFF)
  DMA (master_1) ──┘                    └── RegFile   (slave_1, LITE, 0x1000–0x100F)

Scenario
--------
  1. CPU writes a 4-word burst to MemBank, then reads it back.
  2. CPU writes 2 configuration words to RegFile (LITE, auto-split), reads back.
  3. DMA writes an 8-word burst to MemBank concurrently with the CPU read.

All writes and reads are checked for correctness, and transfer times are printed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pysilicon.hw.aximm import (
    AXIMMCrossBarIF,
    AXIMMCrossBarIFMaster,
    AXIMMCrossBarIFSlave,
    AXIMMProtocol,
    assign_address_ranges,
    Words,
)
from pysilicon.hw.clock import Clock
from pysilicon.simulation.simobj import ProcessGen, SimObj
from pysilicon.simulation.simulation import Simulation


# ---------------------------------------------------------------------------
# Slave: memory bank (FULL, burst)
# ---------------------------------------------------------------------------

@dataclass
class MemBank(SimObj):
    """Simple word-addressed RAM modeled as a dict."""

    access_latency: float = 4.0
    """Simulated memory access time in cycles."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._mem: dict[int, int] = {}
        self.slave_ep = AXIMMCrossBarIFSlave(
            sim=self.sim,
            protocol=AXIMMProtocol.FULL,
            bitwidth=32,
            rx_write_proc=self.rx_write,
            rx_read_proc=self.rx_read,
        )

    def rx_write(self, words: Words, local_addr: int) -> ProcessGen[None]:
        word_bytes = 4
        for i, w in enumerate(words):
            self._mem[local_addr + i * word_bytes] = int(w)
        yield self.timeout(0)   # no-op: write is non-blocking for caller

    def rx_read(self, nwords: int, local_addr: int) -> ProcessGen[Words]:
        yield self.timeout(self.access_latency / self.sim._clk_ref.freq
                           if hasattr(self.sim, '_clk_ref') else self.access_latency)
        word_bytes = 4
        result = np.array(
            [self._mem.get(local_addr + i * word_bytes, 0) for i in range(nwords)],
            dtype=np.uint32,
        )
        return result


# ---------------------------------------------------------------------------
# Slave: register file (LITE, single-word)
# ---------------------------------------------------------------------------

@dataclass
class RegFile(SimObj):
    """Configuration register file; each word is a separate register."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._regs: dict[int, int] = {}
        self.slave_ep = AXIMMCrossBarIFSlave(
            sim=self.sim,
            protocol=AXIMMProtocol.LITE,
            bitwidth=32,
            rx_write_proc=self.rx_write,
            rx_read_proc=self.rx_read,
            latency_per_word=3.0,   # address phase + data phase + ACK
        )

    def rx_write(self, words: Words, local_addr: int) -> ProcessGen[None]:
        self._regs[local_addr] = int(words[0])
        yield self.timeout(0)

    def rx_read(self, nwords: int, local_addr: int) -> ProcessGen[Words]:
        yield self.timeout(0)
        result = np.array([self._regs.get(local_addr, 0)], dtype=np.uint32)
        return result


# ---------------------------------------------------------------------------
# Master: CPU model
# ---------------------------------------------------------------------------

@dataclass
class CPU(SimObj):
    """Issues a scripted sequence of write/read transactions."""

    clk: Clock = field(default_factory=lambda: Clock(freq=1.0))

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.master_ep = AXIMMCrossBarIFMaster(sim=self.sim, bitwidth=32)
        self.read_results: dict[str, np.ndarray] = {}

    def run_proc(self) -> ProcessGen[None]:
        env = self.env

        # --- write 4 words to MemBank ---
        tx_words = np.array([0xA0, 0xA1, 0xA2, 0xA3], dtype=np.uint32)
        t0 = self.now
        yield self.process(self.master_ep.write(tx_words, 0x0000))
        print(f"[CPU] write(4, 0x0000): done at t={self.now:.4f}  "
              f"(dt={self.now-t0:.4f})")

        # --- read 4 words back from MemBank ---
        t0 = self.now
        proc = env.process(self.master_ep.read(4, 0x0000))
        yield proc
        rx_words = proc.value
        self.read_results['mem_read'] = rx_words
        print(f"[CPU] read(4,  0x0000): {rx_words} at t={self.now:.4f}  "
              f"(dt={self.now-t0:.4f})")

        # --- write 2 config words to RegFile (LITE, auto-split) ---
        cfg_words = np.array([0xCAFE, 0xBEEF], dtype=np.uint32)
        t0 = self.now
        yield self.process(self.master_ep.write(cfg_words, 0x1000))
        print(f"[CPU] write(2, 0x1000): done at t={self.now:.4f}  "
              f"(dt={self.now-t0:.4f})")

        # --- read 2 config words back from RegFile ---
        t0 = self.now
        proc = env.process(self.master_ep.read(2, 0x1000))
        yield proc
        cfg_rx = proc.value
        self.read_results['reg_read'] = cfg_rx
        print(f"[CPU] read(2,  0x1000): {cfg_rx} at t={self.now:.4f}  "
              f"(dt={self.now-t0:.4f})")


# ---------------------------------------------------------------------------
# Master: DMA engine
# ---------------------------------------------------------------------------

@dataclass
class DMA(SimObj):
    """Writes a large burst to MemBank after a short startup delay."""

    start_delay: float = 0.05

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.master_ep = AXIMMCrossBarIFMaster(sim=self.sim, bitwidth=32)
        self.write_done_time: float = 0.0

    def run_proc(self) -> ProcessGen[None]:
        yield self.timeout(self.start_delay)
        burst = np.arange(8, dtype=np.uint32) + 0x100
        t0 = self.now
        yield self.process(self.master_ep.write(burst, 0x0100))
        self.write_done_time = self.now
        print(f"[DMA] write(8, 0x0100): done at t={self.now:.4f}  "
              f"(dt={self.now-t0:.4f})")


# ---------------------------------------------------------------------------
# Demo harness
# ---------------------------------------------------------------------------

class AXIMMDemo:
    """Wires up the crossbar and runs the simulation."""

    MEM_BASE  = 0x0000
    MEM_SIZE  = 0x1000    # 4 KiB
    REG_BASE  = 0x1000
    REG_SIZE  = 0x0010    # 16 bytes (4 registers)

    def __init__(self) -> None:
        self.sim = Simulation()
        self.clk = Clock(freq=100.0)   # 100 Hz → 1 cycle = 0.01 s

        self.mem   = MemBank(sim=self.sim, access_latency=4.0)
        self.regs  = RegFile(sim=self.sim)
        self.cpu   = CPU(sim=self.sim, clk=self.clk)
        self.dma   = DMA(sim=self.sim, start_delay=0.05)

        self.xbar = AXIMMCrossBarIF(
            sim=self.sim,
            clk=self.clk,
            nports_master=2,
            nports_slave=2,
            bitwidth=32,
            latency_init=2.0,
            latency_read_return=2.0,
        )

        self.xbar.bind("master_0", self.cpu.master_ep)
        self.xbar.bind("master_1", self.dma.master_ep)
        self.xbar.bind("slave_0",  self.mem.slave_ep)
        self.xbar.bind("slave_1",  self.regs.slave_ep)

        assign_address_ranges(
            [self.mem.slave_ep, self.regs.slave_ep],
            [(self.MEM_BASE, self.MEM_SIZE), (self.REG_BASE, self.REG_SIZE)],
        )

    def run_and_check(self) -> None:
        print("=== AXI-MM crossbar demo ===")
        self.sim.run_sim()
        print()

        # MemBank round-trip
        expected_mem = np.array([0xA0, 0xA1, 0xA2, 0xA3], dtype=np.uint32)
        rx_mem = self.cpu.read_results['mem_read']
        assert np.array_equal(rx_mem, expected_mem), (
            f"MemBank read mismatch: expected {expected_mem}, got {rx_mem}"
        )

        # RegFile round-trip
        expected_cfg = np.array([0xCAFE, 0xBEEF], dtype=np.uint32)
        rx_cfg = self.cpu.read_results['reg_read']
        assert np.array_equal(rx_cfg, expected_cfg), (
            f"RegFile read mismatch: expected {expected_cfg}, got {rx_cfg}"
        )

        # DMA burst landed in MemBank
        word_bytes = 4
        for i in range(8):
            addr = 0x0100 + i * word_bytes
            local_addr = addr - self.MEM_BASE
            assert self.mem._mem.get(local_addr, None) == 0x100 + i, (
                f"DMA word {i} not found in MemBank"
            )

        print("All checks passed.")


if __name__ == "__main__":
    AXIMMDemo().run_and_check()
