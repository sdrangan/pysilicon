"""
crossbar_demo.py — demonstration of CrossBarIF with 2 inputs and 2 outputs.

Routing rule: if words[0] is even the burst goes to output port 0;
              if words[0] is odd  the burst goes to output port 1.

Two sources send bursts concurrently.  The demo prints each burst as it
arrives at its destination and asserts that every burst reached the
correct output.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pysilicon.hw.clock import Clock
from pysilicon.hw.interface import CrossBarIF, CrossBarIFInput, CrossBarIFOutput, Words
from pysilicon.simulation.simobj import ProcessGen, SimObj
from pysilicon.simulation.simulation import Simulation


def route_by_first_word(words: Words, port_in: int) -> int:
    """Route even first-word values to output 0, odd to output 1."""
    return int(words[0]) % 2


@dataclass
class Source(SimObj):
    """Sends a list of bursts through a CrossBarIFInput endpoint."""

    tx_packets: list[Words] = field(default_factory=list)
    """Bursts to send, in order."""

    tx_gap: float = 1.0
    """Simulation time to wait between consecutive bursts."""

    bitwidth: int = 32

    def __init_subclass__(cls, **kwargs):
        return super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.input_ep = CrossBarIFInput(sim=self.sim, bitwidth=self.bitwidth)
        self.tx_times: list[float] = []

    def run_proc(self) -> ProcessGen:
        for i, packet in enumerate(self.tx_packets):
            print(f"[{self.name}] sending burst {i}: {packet} at t={self.env.now:.1f}")
            yield self.process(self.input_ep.write(packet))
            self.tx_times.append(self.now)
            yield self.timeout(self.tx_gap)


@dataclass
class Sink(SimObj):
    """Receives bursts from a CrossBarIFOutput endpoint."""

    rx_gap: float = 0.0
    """Processing delay for each received burst."""

    bitwidth: int = 32
    queue_size: int | None = None

    def __init_subclass__(cls, **kwargs):
        return super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rx_packets: list[Words] = []
        self.rx_times: list[float] = []
        self.output_ep = CrossBarIFOutput(
            sim=self.sim,
            bitwidth=self.bitwidth,
            rx_proc=self.rx_proc,
            queue_size=self.queue_size,
        )

    def rx_proc(self, words: Words) -> ProcessGen:
        print(f"[{self.name}] received: {words} at t={self.env.now:.1f}")
        self.rx_packets.append(np.array(words, copy=True))
        self.rx_times.append(self.now)
        yield self.timeout(self.rx_gap)


class CrossBarDemo:
    """2-input × 2-output crossbar demonstration."""

    def __init__(self) -> None:
        self.sim = Simulation()
        self.clk = Clock(freq=10.0)

        # Bursts from source 0 — first words are all even → go to out_0
        packets_0: list[Words] = [
            np.array([2, 3, 4], dtype=np.uint32),
            np.array([10, 11], dtype=np.uint32),
        ]
        # Bursts from source 1 — first words are all odd → go to out_1
        packets_1: list[Words] = [
            np.array([1, 2, 3], dtype=np.uint32),
            np.array([7, 8], dtype=np.uint32),
        ]

        self.src0 = Source(sim=self.sim, tx_packets=packets_0, tx_gap=0.1)
        self.src1 = Source(sim=self.sim, tx_packets=packets_1, tx_gap=0.1)
        self.sink0 = Sink(sim=self.sim)
        self.sink1 = Sink(sim=self.sim)

        self.xbar = CrossBarIF(
            sim=self.sim,
            clk=self.clk,
            nports_in=2,
            nports_out=2,
            bitwidth=32,
            route_fn=route_by_first_word,
        )

        self.xbar.bind("in_0", self.src0.input_ep)
        self.xbar.bind("in_1", self.src1.input_ep)
        self.xbar.bind("out_0", self.sink0.output_ep)
        self.xbar.bind("out_1", self.sink1.output_ep)

    def run_and_check(self) -> None:
        self.sim.run_sim()

        # sink0 should receive all bursts from src0 (even first word)
        assert len(self.sink0.rx_packets) == 2, (
            f"sink0 expected 2 packets, got {len(self.sink0.rx_packets)}"
        )
        for tx, rx in zip(self.src0.tx_packets, self.sink0.rx_packets, strict=True):
            assert np.array_equal(tx, rx), f"sink0 data mismatch: {tx} != {rx}"

        # sink1 should receive all bursts from src1 (odd first word)
        assert len(self.sink1.rx_packets) == 2, (
            f"sink1 expected 2 packets, got {len(self.sink1.rx_packets)}"
        )
        for tx, rx in zip(self.src1.tx_packets, self.sink1.rx_packets, strict=True):
            assert np.array_equal(tx, rx), f"sink1 data mismatch: {tx} != {rx}"

        print("All routing checks passed.")


if __name__ == "__main__":
    CrossBarDemo().run_and_check()
