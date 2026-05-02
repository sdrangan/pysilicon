---
title: Stream Interfaces
parent: Hardware Interfaces
nav_order: 2
has_children: false
---

# Stream Interfaces

Stream interfaces model **unidirectional data bursts** from one component to another. They correspond to AXI4-Stream and Vitis HLS stream bus protocols. Two interface classes are available: `StreamIF` for point-to-point connections and `CrossBarIF` for n-input × m-output switching fabrics.

## Point-to-point: StreamIF

`StreamIF` connects one master endpoint to one slave endpoint. The master calls `write(words)` to push a burst; the slave receives it via an `rx_proc` callback after the modelled latency.

### Classes

| Class | Role | Key parameters |
|---|---|---|
| `StreamIF` | Interface | `clk`, `bitwidth`, `latency_init`, `stream_type`, `notify_type` |
| `StreamIFMaster` | Master endpoint | `bitwidth`, `stream_type`, `notify_type` |
| `StreamIFSlave` | Slave endpoint | `bitwidth`, `stream_type`, `notify_type`, `rx_proc`, `queue_size` |

### Latency model

```
transfer_time = (latency_init + nwords) / clk.freq   [seconds]
```

- `latency_init` — fixed cycles for wire delay, arbitration, etc.
- `nwords` — one additional cycle per word in the burst (one beat per clock)

### Stream type

`StreamType` selects the bus protocol variant:

| Value | Meaning |
|---|---|
| `StreamType.axi4` | AXI4-Stream (default) |
| `StreamType.hls` | Vitis HLS stream |

The stream type must match between master and slave. The interface infers it from the first bound endpoint if not set explicitly.

### Notify type

`TransferNotifyType` controls when the slave's `rx_proc` is invoked:

| Value | Behaviour |
|---|---|
| `TransferNotifyType.end_only` | `rx_proc(words)` called once, after the full burst arrives (default) |
| `TransferNotifyType.begin_end` | `rx_proc(words)` called at burst start; `notify_end_proc()` called at burst end |
| `TransferNotifyType.per_word` | Not yet supported |

### Example: point-to-point stream

```python
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from pysilicon.hw.clock import Clock
from pysilicon.hw.interface import StreamIF, StreamIFMaster, StreamIFSlave, Words
from pysilicon.simulation.simobj import ProcessGen, SimObj
from pysilicon.simulation.simulation import Simulation


@dataclass
class Producer(SimObj):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.ep = StreamIFMaster(sim=self.sim, bitwidth=32)

    def run_proc(self) -> ProcessGen:
        for i in range(3):
            words = np.array([i * 10, i * 10 + 1], dtype=np.uint32)
            yield self.process(self.ep.write(words))


@dataclass
class Consumer(SimObj):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.received: list[np.ndarray] = []
        self.ep = StreamIFSlave(
            sim=self.sim,
            bitwidth=32,
            rx_proc=self.on_rx,
            queue_size=16,
        )

    def on_rx(self, words: Words) -> ProcessGen:
        self.received.append(words.copy())
        yield self.env.timeout(0)   # or model processing delay here

    def run_proc(self) -> ProcessGen:
        yield from self.ep.run_proc()


sim = Simulation()
clk = Clock(freq=100e6)

producer = Producer(sim=sim)
consumer = Consumer(sim=sim)

iface = StreamIF(sim=sim, clk=clk, bitwidth=32, latency_init=4.0)
iface.bind("master", producer.ep)
iface.bind("slave",  consumer.ep)

sim.run_sim()
```

The `Consumer.run_proc()` must delegate to `ep.run_proc()` so the slave's receive loop is active during simulation. `Simulation.run_sim()` calls each `SimObj.run_proc()` automatically, so this pattern wires together correctly.

---

## Crossbar: CrossBarIF

`CrossBarIF` routes bursts from `nports_in` input ports to `nports_out` output ports via a configurable routing function.

### Classes

| Class | Role | Key parameters |
|---|---|---|
| `CrossBarIF` | Interface | `clk`, `bitwidth`, `latency_init`, `nports_in`, `nports_out`, `route_fn` |
| `CrossBarIFInput` | Input (master) endpoint | `bitwidth` |
| `CrossBarIFOutput` | Output (slave) endpoint | `bitwidth`, `rx_proc`, `queue_size` |

Endpoint names follow the pattern `in_0`, `in_1`, …, `out_0`, `out_1`, …

### Routing function

The `route_fn(words, port_in) -> port_out` callable maps each burst to an output port. If not provided, the default is `port_out = port_in % nports_out`.

```python
def route_by_first_word(words: Words, port_in: int) -> int:
    return int(words[0]) % nports_out
```

### Example: 2×2 crossbar

```python
from pysilicon.hw.interface import CrossBarIF, CrossBarIFInput, CrossBarIFOutput

xbar = CrossBarIF(
    sim=sim,
    clk=clk,
    nports_in=2,
    nports_out=2,
    bitwidth=32,
    latency_init=2.0,
    route_fn=route_by_first_word,
)

xbar.bind("in_0",  src0.input_ep)
xbar.bind("in_1",  src1.input_ep)
xbar.bind("out_0", sink0.output_ep)
xbar.bind("out_1", sink1.output_ep)
```

The crossbar's `write(words, port_in)` is called internally by `CrossBarIFInput.write(words)` — callers only need the input endpoint's `write` method.

Each `CrossBarIFOutput` endpoint has the same `run_proc()` loop as `StreamIFSlave` and must be started before transfers are sent.

---

## Common patterns

### Checking data in a slave

```python
class Checker(SimObj):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.bursts: list[np.ndarray] = []
        self.ep = StreamIFSlave(sim=self.sim, bitwidth=32, rx_proc=self.rx_proc)

    def rx_proc(self, words: Words) -> ProcessGen:
        self.bursts.append(words.copy())
        yield self.env.timeout(0)

    def run_proc(self) -> ProcessGen:
        yield from self.ep.run_proc()

    def post_sim(self) -> None:
        assert len(self.bursts) == expected_count
```

### Modelling receiver processing delay

Set a non-zero delay in `rx_proc` to model the time the slave spends consuming each burst:

```python
def rx_proc(self, words: Words) -> ProcessGen:
    processing_cycles = len(words) * 2
    yield self.timeout(processing_cycles / self.clk.freq)
```

### Queue depth

`queue_size` on the slave endpoint bounds how many words can be in-flight. Setting `queue_size=None` (default) gives an unbounded queue. For backpressure modelling, set an explicit depth.

---

## Quick reference

```python
from pysilicon.hw.interface import (
    StreamIF, StreamIFMaster, StreamIFSlave,
    CrossBarIF, CrossBarIFInput, CrossBarIFOutput,
    StreamType, TransferNotifyType,
    Words,
)
from pysilicon.hw.clock import Clock
```

| Operation | Code |
|---|---|
| Create interface | `StreamIF(sim=sim, clk=clk, bitwidth=32, latency_init=4.0)` |
| Create master ep | `StreamIFMaster(sim=sim, bitwidth=32)` |
| Create slave ep | `StreamIFSlave(sim=sim, bitwidth=32, rx_proc=fn)` |
| Bind | `iface.bind("master", ep)` |
| Write (from run_proc) | `yield self.process(ep.write(words))` |
| Start slave loop | `yield from ep.run_proc()` |
