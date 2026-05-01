"""
crossbar_demo.py — Demonstrates CrossbarIF with 2 masters and 3 slaves.

Two masters send packets simultaneously.  Each packet specifies a destination
slave port (0, 1, or 2).  The demo verifies that every packet arrives at the
correct slave and that no data is lost or corrupted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import simpy

from pysilicon.hw.clock import Clock
from pysilicon.hw.interface import CrossbarIF, CrossbarIFMaster, CrossbarIFSlave, Words
from pysilicon.simulation.simobj import ProcessGen, SimObj
from pysilicon.simulation.simulation import Simulation


@dataclass
class CrossbarMaster(SimObj):
    """Sends a list of (packet, dest) pairs over a CrossbarIF master port."""

    tx_items: list[tuple[Words, int]] = field(default_factory=list)
    """Packets to send, each paired with a destination slave index."""

    tx_gap: float = 1.0
    """Simulated time gap between successive sends."""

    bitwidth: int = 32

    def __init_subclass__(cls, **kwargs):
        return super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.xbar_ep = CrossbarIFMaster(sim=self.sim, bitwidth=self.bitwidth)
        self.tx_times: list[float] = []

    def run_proc(self) -> ProcessGen:
        for i, (packet, dest) in enumerate(self.tx_items):
            print(f"[{self.name}] sending packet {i} → slave_{dest}: {packet}  t={self.now:.2f}")
            yield self.process(self.xbar_ep.write(packet, dest))
            self.tx_times.append(self.now)
            yield self.timeout(self.tx_gap)


@dataclass
class CrossbarSlave(SimObj):
    """Receives packets from a CrossbarIF slave port."""

    rx_gap: float = 0.0
    """Optional processing delay after each received packet."""

    bitwidth: int = 32

    def __init_subclass__(cls, **kwargs):
        return super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rx_packets: list[Words] = []
        self.rx_times: list[float] = []
        self.xbar_ep = CrossbarIFSlave(
            sim=self.sim,
            bitwidth=self.bitwidth,
            rx_proc=self.rx_proc,
        )

    def rx_proc(self, words: Words) -> ProcessGen:
        print(f"[{self.name}] received: {words}  t={self.now:.2f}")
        self.rx_packets.append(np.array(words, copy=True))
        self.rx_times.append(self.now)
        yield self.timeout(self.rx_gap)


class CrossbarDemo:
    """
    2 masters × 3 slaves demo.

    master_0 sends to slave_0 and slave_2.
    master_1 sends to slave_1 and slave_2 (contention with master_0 on slave_2).
    """

    def __init__(self) -> None:
        self.sim = Simulation()
        self.clk = Clock(freq=1.0)  # 1 Hz → 1 cycle = 1 second

        # Masters
        self.m0 = CrossbarMaster(
            sim=self.sim,
            tx_items=[
                (np.array([10, 11, 12], dtype=np.uint32), 0),
                (np.array([20, 21], dtype=np.uint32), 2),
            ],
            tx_gap=0.5,
        )
        self.m1 = CrossbarMaster(
            sim=self.sim,
            tx_items=[
                (np.array([30], dtype=np.uint32), 1),
                (np.array([40, 41, 42, 43], dtype=np.uint32), 2),
            ],
            tx_gap=0.5,
        )

        # Slaves
        self.s0 = CrossbarSlave(sim=self.sim, rx_gap=0.0)
        self.s1 = CrossbarSlave(sim=self.sim, rx_gap=0.0)
        self.s2 = CrossbarSlave(sim=self.sim, rx_gap=0.0)

        # Crossbar interface
        self.xbar = CrossbarIF(
            sim=self.sim,
            clk=self.clk,
            num_masters=2,
            num_slaves=3,
            bitwidth=32,
        )
        self.xbar.bind('master_0', self.m0.xbar_ep)
        self.xbar.bind('master_1', self.m1.xbar_ep)
        self.xbar.bind('slave_0', self.s0.xbar_ep)
        self.xbar.bind('slave_1', self.s1.xbar_ep)
        self.xbar.bind('slave_2', self.s2.xbar_ep)

    def run_and_check(self) -> None:
        self.sim.run_sim()

        assert len(self.s0.rx_packets) == 1, f"slave_0: expected 1 packet, got {len(self.s0.rx_packets)}"
        assert len(self.s1.rx_packets) == 1, f"slave_1: expected 1 packet, got {len(self.s1.rx_packets)}"
        assert len(self.s2.rx_packets) == 2, f"slave_2: expected 2 packets, got {len(self.s2.rx_packets)}"

        np.testing.assert_array_equal(self.s0.rx_packets[0], np.array([10, 11, 12], dtype=np.uint32))
        np.testing.assert_array_equal(self.s1.rx_packets[0], np.array([30], dtype=np.uint32))

        # slave_2 receives from both masters; order depends on simulation timing
        combined = {tuple(p.tolist()) for p in self.s2.rx_packets}
        assert (20, 21) in combined, "slave_2 missing packet from master_0"
        assert (40, 41, 42, 43) in combined, "slave_2 missing packet from master_1"

        print("\nAll checks passed!")


if __name__ == "__main__":
    CrossbarDemo().run_and_check()
