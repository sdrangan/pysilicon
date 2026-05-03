---
title: AXI-MM Interfaces
parent: Interfaces
nav_order: 3
has_children: false
---

# AXI-MM Interfaces

`AXIMMCrossBarIF` models an **AXI Memory-Mapped crossbar**: a switching fabric where multiple master ports can initiate both read and write transactions to multiple slave ports, with routing determined by global byte addresses rather than port indices.

The model operates at **burst-level, cycle-approximate** granularity. Sub-channel details (AW, W, AR, R, B) are not modelled separately. FULL (burst-capable) and LITE (register-style) slaves are supported on the same crossbar.

## Key concepts

### Address-based routing

There is no explicit "slave port number" in an AXI-MM transaction. Instead:

1. Each slave endpoint is assigned a byte-address range `[base_addr, base_addr + size)`.
2. When a master calls `write(words, global_addr)` or `read(nwords, global_addr)`, the crossbar decodes `global_addr` against every slave's range.
3. The matching slave receives the data at a **local address**: `local_addr = global_addr - slave.base_addr`.
4. If no slave covers the address, a `RuntimeError` is raised immediately.

Address ranges are assigned with `assign_address_ranges()` before the simulation starts:

```python
from pysilicon.hw.aximm import assign_address_ranges

assign_address_ranges(
    [mem_slave_ep, reg_slave_ep],
    [(0x0000_0000, 0x0001_0000),   # MemBank: 64 KiB at 0x0
     (0x0001_0000, 0x0000_0010)],  # RegFile: 16 bytes at 0x10000
)
```

### Protocol variants: FULL vs LITE

Each slave endpoint declares its protocol via `AXIMMProtocol`:

| Value | Transfer model | Typical use |
|---|---|---|
| `AXIMMProtocol.FULL` | One burst call for all `nwords` | DDR, block RAM, DMA buffers |
| `AXIMMProtocol.LITE` | One call per word, auto-incremented addresses | Configuration registers |

For LITE slaves, a multi-word `write(words, addr)` is **silently split** into `nwords` single-word transactions. Each word is delivered at `local_addr + i * word_bytes`. The caller does not need to know which protocol the slave implements.

### Read return path

Reads return data through the SimPy `Process.value` mechanism:

```python
proc = env.process(master_ep.read(nwords=4, global_addr=0x0000))
yield proc
data = proc.value   # numpy array, shape (nwords,)
```

Alternatively, inside a `run_proc` generator:

```python
def run_proc(self) -> ProcessGen:
    proc = self.env.process(self.master_ep.read(4, 0x0000))
    yield proc
    data = proc.value   # available immediately after yield
```

---

## Latency model

All cycle counts are divided by `clk.freq` to produce seconds.

### FULL write

```
time = (latency_init + nwords) / clk.freq
```

### FULL read

```
time = latency_init / clk.freq          (request wire)
     + slave rx_read_proc duration      (peripheral access)
     + (latency_read_return + nwords) / clk.freq   (return wire + burst)
```

### LITE write

```
time = nwords * latency_per_word / clk.freq
```

Each word is a full AXI-Lite transaction: `latency_per_word` must be ≥ 2 to represent the address and data phases. A typical register access is 3–5 cycles.

### LITE read

```
time = nwords * latency_per_word / clk.freq
```

Each word is a separate round-trip transaction at `latency_per_word` cycles.

---

## Classes

### AXIMMAddressRange

```python
from pysilicon.hw.aximm import AXIMMAddressRange

ar = AXIMMAddressRange(base_addr=0x1000, size=0x100)
ar.contains(0x1050)         # True
ar.contains(0x1100)         # False
ar.to_local(0x1040)         # 0x40
```

A frozen dataclass; instances are created by `assign_address_ranges()` and stored on each slave endpoint.

### AXIMMCrossBarIFSlave

```python
from pysilicon.hw.aximm import AXIMMCrossBarIFSlave, AXIMMProtocol

slave_ep = AXIMMCrossBarIFSlave(
    sim=sim,
    protocol=AXIMMProtocol.FULL,   # or AXIMMProtocol.LITE
    bitwidth=32,
    rx_write_proc=self.on_write,   # called on each write transaction
    rx_read_proc=self.on_read,     # called on each read transaction
    latency_per_word=3.0,          # cycles per word (LITE only)
)
```

**`rx_write_proc(words, local_addr) -> ProcessGen`**  
Called with the transferred words and the local byte address of the first word. For LITE slaves this is called once per word with a one-element array.

**`rx_read_proc(nwords, local_addr) -> ProcessGen`**  
Called to retrieve data. The generator's return value must be a numpy array of shape `(nwords,)`.

```python
def on_read(self, nwords: int, local_addr: int) -> ProcessGen:
    yield self.env.timeout(0)   # model peripheral latency here
    word_bytes = 4
    return np.array(
        [self._mem.get(local_addr + i * word_bytes, 0) for i in range(nwords)],
        dtype=np.uint32,
    )
```

### AXIMMCrossBarIFMaster

```python
from pysilicon.hw.aximm import AXIMMCrossBarIFMaster

master_ep = AXIMMCrossBarIFMaster(sim=sim, bitwidth=32)
```

Exposes two methods, both used as SimPy generators:

```python
yield self.process(master_ep.write(words, global_addr))

proc = self.env.process(master_ep.read(nwords, global_addr))
yield proc
data = proc.value
```

### AXIMMCrossBarIF

```python
from pysilicon.hw.aximm import AXIMMCrossBarIF

xbar = AXIMMCrossBarIF(
    sim=sim,
    clk=clk,
    nports_master=2,
    nports_slave=2,
    bitwidth=32,
    latency_init=2.0,           # wire cycles, forward direction
    latency_read_return=2.0,    # wire cycles, return direction (FULL reads)
)
```

Endpoint names: `master_0` … `master_{n-1}` and `slave_0` … `slave_{m-1}`.

---

## Full example

The following example wires a CPU and a DMA engine to a memory bank and a register file:

```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from pysilicon.hw.aximm import (
    AXIMMCrossBarIF, AXIMMCrossBarIFMaster, AXIMMCrossBarIFSlave,
    AXIMMProtocol, AXIMMAddressRange, assign_address_ranges,
)
from pysilicon.hw.clock import Clock
from pysilicon.simulation.simobj import ProcessGen, SimObj
from pysilicon.simulation.simulation import Simulation


@dataclass
class MemBank(SimObj):
    """Simple word-addressed SRAM (FULL, burst)."""
    def __post_init__(self) -> None:
        super().__post_init__()
        self._mem: dict[int, int] = {}
        self.slave_ep = AXIMMCrossBarIFSlave(
            sim=self.sim, protocol=AXIMMProtocol.FULL, bitwidth=32,
            rx_write_proc=self.on_write, rx_read_proc=self.on_read,
        )

    def on_write(self, words, local_addr: int) -> ProcessGen:
        for i, w in enumerate(words):
            self._mem[local_addr + i * 4] = int(w)
        yield self.env.timeout(0)

    def on_read(self, nwords: int, local_addr: int) -> ProcessGen:
        yield self.env.timeout(4 / self.sim._clk_ref.freq)  # 4-cycle access
        return np.array(
            [self._mem.get(local_addr + i * 4, 0) for i in range(nwords)],
            dtype=np.uint32,
        )


@dataclass
class RegFile(SimObj):
    """Configuration registers (LITE, one register per word)."""
    def __post_init__(self) -> None:
        super().__post_init__()
        self._regs: dict[int, int] = {}
        self.slave_ep = AXIMMCrossBarIFSlave(
            sim=self.sim, protocol=AXIMMProtocol.LITE, bitwidth=32,
            rx_write_proc=self.on_write, rx_read_proc=self.on_read,
            latency_per_word=3.0,
        )

    def on_write(self, words, local_addr: int) -> ProcessGen:
        self._regs[local_addr] = int(words[0])
        yield self.env.timeout(0)

    def on_read(self, nwords: int, local_addr: int) -> ProcessGen:
        yield self.env.timeout(0)
        return np.array([self._regs.get(local_addr, 0)], dtype=np.uint32)


@dataclass
class CPU(SimObj):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.master_ep = AXIMMCrossBarIFMaster(sim=self.sim, bitwidth=32)

    def run_proc(self) -> ProcessGen:
        env = self.env

        # Write 4 words to MemBank, then read back
        words = np.array([0xA0, 0xA1, 0xA2, 0xA3], dtype=np.uint32)
        yield self.process(self.master_ep.write(words, 0x0000))

        proc = env.process(self.master_ep.read(4, 0x0000))
        yield proc
        assert np.array_equal(proc.value, words)

        # Write 2 config words to RegFile (auto-split into 2 LITE transactions)
        cfg = np.array([0xCAFE, 0xBEEF], dtype=np.uint32)
        yield self.process(self.master_ep.write(cfg, 0x1000))

        proc = env.process(self.master_ep.read(2, 0x1000))
        yield proc
        assert np.array_equal(proc.value, cfg)


sim = Simulation()
clk = Clock(freq=100.0)

mem  = MemBank(sim=sim)
regs = RegFile(sim=sim)
cpu  = CPU(sim=sim)

xbar = AXIMMCrossBarIF(
    sim=sim, clk=clk,
    nports_master=1, nports_slave=2, bitwidth=32,
    latency_init=2.0, latency_read_return=2.0,
)
xbar.bind("master_0", cpu.master_ep)
xbar.bind("slave_0",  mem.slave_ep)
xbar.bind("slave_1",  regs.slave_ep)

assign_address_ranges(
    [mem.slave_ep, regs.slave_ep],
    [(0x0000, 0x1000), (0x1000, 0x0010)],
)

sim.run_sim()
```

---

## Building on top: register maps

The AXI-MM interface is intentionally word-addressed so that a higher-level register-map abstraction can be layered on top without changes to the interface model. A register-map layer would:

1. Name each register and its bit fields.
2. Translate field-level read-modify-write operations into single-word `write`/`read` calls on a LITE slave.
3. Handle bank switching or stride patterns as needed.

The AXI-MM interface handles all timing and routing; the register-map layer handles only semantics.

---

## Quick reference

```python
from pysilicon.hw.aximm import (
    AXIMMCrossBarIF,
    AXIMMCrossBarIFMaster,
    AXIMMCrossBarIFSlave,
    AXIMMProtocol,
    AXIMMAddressRange,
    assign_address_ranges,
)
```

| Operation | Code |
|---|---|
| Create crossbar | `AXIMMCrossBarIF(sim=sim, clk=clk, nports_master=M, nports_slave=N, ...)` |
| Create master ep | `AXIMMCrossBarIFMaster(sim=sim, bitwidth=32)` |
| Create FULL slave ep | `AXIMMCrossBarIFSlave(sim=sim, protocol=AXIMMProtocol.FULL, ...)` |
| Create LITE slave ep | `AXIMMCrossBarIFSlave(sim=sim, protocol=AXIMMProtocol.LITE, latency_per_word=3.0, ...)` |
| Assign address ranges | `assign_address_ranges([s0, s1], [(0x0000, 0x1000), (0x1000, 0x10)])` |
| Write | `yield self.process(master_ep.write(words, global_addr))` |
| Read | `proc = env.process(master_ep.read(nwords, global_addr)); yield proc; data = proc.value` |
