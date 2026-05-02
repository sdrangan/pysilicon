"""
CrossBarIF-specific tests: routing behaviour and topology validation.

Timing / data-correctness tests shared with StreamIF are in test_interface.py
(parameterised over both StreamScenario and CrossBarScenario11).
"""
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
    Words,
)
from pysilicon.simulation.simulation import Simulation


# ---------------------------------------------------------------------------
# Multi-port routing harness
# ---------------------------------------------------------------------------

class CrossBarScenario:
    """
    Test harness for a CrossBarIF with configurable topology.

    Parameters
    ----------
    nports_in, nports_out : int
        Crossbar dimensions.
    queue_sizes : list[int | None]
        Per-output queue size (None = unbounded).
    route_fn : callable | None
        Routing function; None uses the default modulo routing.
    rx_processing_delay : float
        SimPy time each output spends processing one burst.
    clk_freq : float
        Clock frequency in Hz.
    """

    def __init__(
        self,
        nports_in: int = 2,
        nports_out: int = 2,
        queue_sizes: list[int | None] | None = None,
        route_fn=None,
        rx_processing_delay: float = 0.0,
        clk_freq: float = 1.0,
    ) -> None:
        self.sim = Simulation()
        self.env = self.sim.env
        self.clk = Clock(freq=clk_freq)

        if queue_sizes is None:
            queue_sizes = [None] * nports_out

        self.xbar = CrossBarIF(
            sim=self.sim,
            clk=self.clk,
            nports_in=nports_in,
            nports_out=nports_out,
            route_fn=route_fn,
        )
        self.inputs = [
            CrossBarIFInput(sim=self.sim, bitwidth=32) for _ in range(nports_in)
        ]
        self.outputs = [
            CrossBarIFOutput(
                sim=self.sim,
                bitwidth=32,
                queue_size=queue_sizes[j],
                rx_proc=self._make_rx_proc(j, rx_processing_delay),
            )
            for j in range(nports_out)
        ]

        for i, inp in enumerate(self.inputs):
            self.xbar.bind(f"in_{i}", inp)
        for j, out in enumerate(self.outputs):
            self.xbar.bind(f"out_{j}", out)

        self.received: list[list[np.ndarray]] = [[] for _ in range(nports_out)]
        self.rx_start_times: list[list[float]] = [[] for _ in range(nports_out)]
        self.rx_end_times: list[list[float]] = [[] for _ in range(nports_out)]

    def _make_rx_proc(self, port_out: int, delay: float):
        def rx_proc(words: Words):
            self.rx_start_times[port_out].append(self.env.now)
            self.received[port_out].append(np.array(words, copy=True))
            yield self.env.timeout(delay)
            self.rx_end_times[port_out].append(self.env.now)

        return rx_proc

    def send_proc(self, port_in: int, packets: list[np.ndarray], tx_gap: float = 0.0):
        def _proc():
            for packet in packets:
                yield self.env.process(self.inputs[port_in].write(packet))
                if tx_gap > 0:
                    yield self.env.timeout(tx_gap)

        return _proc

    def run(
        self,
        senders: dict[int, list[np.ndarray]],
        tx_gap: float = 0.0,
        until: float | None = None,
    ) -> None:
        all_done = []
        for port_in, packets in senders.items():
            done = self.env.event()
            all_done.append(done)

            def _proc(packets=packets, port_in=port_in, done=done):
                yield from self.send_proc(port_in, packets, tx_gap)()
                done.succeed()

            self.env.process(_proc())

        for out in self.outputs:
            self.env.process(out.run_proc())

        target = (
            self.env.all_of(all_done) if until is None
            else self.env.timeout(until)
        )
        self.env.run(until=target)


# ---------------------------------------------------------------------------
# Routing tests
# ---------------------------------------------------------------------------

def test_identity_routing_2x2():
    """Default modulo routing: in_0 -> out_0, in_1 -> out_1."""
    sc = CrossBarScenario(nports_in=2, nports_out=2)

    pkts0 = [np.array([1, 2, 3], dtype=np.uint32)]
    pkts1 = [np.array([4, 5], dtype=np.uint32)]

    sc.run({0: pkts0, 1: pkts1})

    assert len(sc.received[0]) == 1
    npt.assert_array_equal(sc.received[0][0], pkts0[0])
    assert len(sc.received[1]) == 1
    npt.assert_array_equal(sc.received[1][0], pkts1[0])


def test_custom_route_fn():
    """Custom route_fn sends all traffic from both inputs to out_0."""
    sc = CrossBarScenario(
        nports_in=2,
        nports_out=2,
        route_fn=lambda words, port_in: 0,
    )

    pkts0 = [np.array([10, 11], dtype=np.uint32)]
    pkts1 = [np.array([20, 21, 22], dtype=np.uint32)]

    sc.run({0: pkts0, 1: pkts1})

    assert len(sc.received[0]) == 2
    assert len(sc.received[1]) == 0


def test_route_fn_based_on_data():
    """Route even first-word bursts to out_0, odd to out_1."""

    def route(words: Words, port_in: int) -> int:
        return int(words[0]) % 2

    sc = CrossBarScenario(nports_in=1, nports_out=2, route_fn=route)

    even_pkt = np.array([2, 3], dtype=np.uint32)
    odd_pkt = np.array([7, 8, 9], dtype=np.uint32)

    sc.run({0: [even_pkt, odd_pkt]})

    assert len(sc.received[0]) == 1
    npt.assert_array_equal(sc.received[0][0], even_pkt)
    assert len(sc.received[1]) == 1
    npt.assert_array_equal(sc.received[1][0], odd_pkt)


def test_multiple_packets_single_path():
    """Multiple bursts on a single 1×1 path are all delivered in order."""
    sc = CrossBarScenario(nports_in=1, nports_out=1)

    packets = [np.array([i, i + 1], dtype=np.uint32) for i in range(0, 10, 2)]
    sc.run({0: packets})

    assert len(sc.received[0]) == len(packets)
    for rx, tx in zip(sc.received[0], packets, strict=True):
        npt.assert_array_equal(rx, tx)


# ---------------------------------------------------------------------------
# CrossBarIF-specific validation tests
# ---------------------------------------------------------------------------

def test_invalid_nports_in():
    sim = Simulation()
    with pytest.raises(ValueError, match="nports_in"):
        CrossBarIF(sim=sim, clk=Clock(freq=1), nports_in=0, nports_out=2)


def test_invalid_nports_out():
    sim = Simulation()
    with pytest.raises(ValueError, match="nports_out"):
        CrossBarIF(sim=sim, clk=Clock(freq=1), nports_in=2, nports_out=0)


def test_missing_clock():
    sim = Simulation()
    with pytest.raises(ValueError, match="clock"):
        CrossBarIF(sim=sim, nports_in=2, nports_out=2)


def test_bind_wrong_input_type():
    sim = Simulation()
    xbar = CrossBarIF(sim=sim, clk=Clock(freq=1), nports_in=1, nports_out=1)
    wrong = CrossBarIFOutput(sim=sim, bitwidth=32)
    with pytest.raises(TypeError):
        xbar.bind("in_0", wrong)


def test_bind_wrong_output_type():
    sim = Simulation()
    xbar = CrossBarIF(sim=sim, clk=Clock(freq=1), nports_in=1, nports_out=1)
    wrong = CrossBarIFInput(sim=sim, bitwidth=32)
    with pytest.raises(TypeError):
        xbar.bind("out_0", wrong)


def test_bind_invalid_ep_name():
    sim = Simulation()
    xbar = CrossBarIF(sim=sim, clk=Clock(freq=1), nports_in=1, nports_out=1)
    ep = CrossBarIFInput(sim=sim, bitwidth=32)
    with pytest.raises(KeyError):
        xbar.bind("master", ep)


def test_bind_out_of_range_input():
    sim = Simulation()
    xbar = CrossBarIF(sim=sim, clk=Clock(freq=1), nports_in=1, nports_out=1)
    ep = CrossBarIFInput(sim=sim, bitwidth=32)
    with pytest.raises(KeyError):
        xbar.bind("in_5", ep)


def test_bind_bitwidth_mismatch():
    sim = Simulation()
    xbar = CrossBarIF(sim=sim, clk=Clock(freq=1), nports_in=1, nports_out=1, bitwidth=32)
    ep = CrossBarIFInput(sim=sim, bitwidth=64)
    with pytest.raises(ValueError, match="bitwidth"):
        xbar.bind("in_0", ep)


def test_bind_already_bound():
    sim = Simulation()
    xbar = CrossBarIF(sim=sim, clk=Clock(freq=1), nports_in=1, nports_out=1)
    ep1 = CrossBarIFInput(sim=sim, bitwidth=32)
    ep2 = CrossBarIFInput(sim=sim, bitwidth=32)
    xbar.bind("in_0", ep1)
    with pytest.raises(ValueError):
        xbar.bind("in_0", ep2)


def test_write_without_binding():
    """Writing from an unbound input endpoint should raise RuntimeError."""
    sim = Simulation()
    ep = CrossBarIFInput(sim=sim, bitwidth=32)
    words = np.array([1, 2], dtype=np.uint32)

    result: list[Exception] = []

    def _proc():
        try:
            yield from ep.write(words)
        except RuntimeError as exc:
            result.append(exc)

    sim.env.process(_proc())
    sim.env.run()
    assert len(result) == 1
    assert "not bound" in str(result[0]).lower()
