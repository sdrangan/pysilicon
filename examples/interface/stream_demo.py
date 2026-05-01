from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import simpy
from numpy.typing import NDArray

from pysilicon.hw.interface import StreamIF, StreamIFMaster, StreamIFSlave, Words
from pysilicon.simulation.simobj import ProcessGen, SimObj
from pysilicon.simulation.simulation import Simulation
from pysilicon.hw.clock import Clock

@dataclass  
class Master(SimObj):
    tx_packets: list[Words] = field(default_factory=list)
    """List of packets to send. Each packet is an array of words."""

    tx_gap: float = 1.0
    """Time gap between sending packets."""

    bitwidth: int = 32
    """Bitwidth of the stream interface."""

    def __init_subclass__(cls, **kwargs):
        return super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.stream_ep = StreamIFMaster(sim=self.sim, bitwidth=self.bitwidth)
        self.tx_times: list[float] = []

    def run_proc(self) -> ProcessGen:
        for i, packet in enumerate(self.tx_packets):
            print(f"Master sending {i}: {packet} at time {self.env.now}")
            yield self.process( self.stream_ep.write(packet) )
            self.tx_times.append(self.now)
            yield self.timeout(self.tx_gap)

@dataclass
class Slave(SimObj):

    rx_packets: list[Words] = field(default_factory=list)
    """List of RX packets. Each packet is an array of words."""

    rx_gap: float = 1.5
    """Time gap between sending packets."""

    bitwidth: int = 32
    """Bitwidth of the stream interface."""

    def __init_subclass__(cls, **kwargs):
        return super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.stream_ep = StreamIFSlave(sim=self.sim, bitwidth=self.bitwidth, rx_proc=self.rx_proc, queue_size=10)
        self.rx_times: list[float] = []

    def rx_proc(self, words: Words) -> ProcessGen:
        print(f"Slave received: {words} at time {self.env.now}")
        self.rx_packets.append(np.array(words, copy=True))
        self.rx_times.append(self.now)
        yield self.timeout(self.rx_gap)


class TestStream:
    def __init__(self) -> None:
        self.sim = Simulation()
        self.packets: list[Words] = [
            np.array([10, 11, 12], dtype=np.uint32),
            np.array([21, 22], dtype=np.uint32),
            np.array([30], dtype=np.uint32),
        ]
        self.clk = Clock(freq=10.0)
        self.master = Master(sim=self.sim, tx_packets=self.packets, tx_gap=0.01)
        self.slave = Slave(sim=self.sim, rx_gap=1.0, bitwidth=32)
        self.iface = StreamIF(sim=self.sim, clk=self.clk, bitwidth=32)  # Assuming StreamIF is imported and used correctly

        self.iface.bind(ep_name="master", endpoint=self.master.stream_ep)
        self.iface.bind(ep_name="slave", endpoint=self.slave.stream_ep)

    def run_and_check(self) -> None:
        self.sim.run_sim()

        if len(self.slave.rx_packets) != len(self.packets):
            raise AssertionError(
                f"Expected {len(self.packets)} packets at slave, got {len(self.slave.rx_packets)}"
            )

        for i, (tx_packet, rx_packet) in enumerate(zip(self.packets, self.slave.rx_packets, strict=True)):
            if not np.array_equal(tx_packet, rx_packet):
                raise AssertionError(f"Packet mismatch at index {i}: tx={tx_packet}, rx={rx_packet}")

        print("Transmission check passed")


if __name__ == "__main__":
    TestStream().run_and_check()
