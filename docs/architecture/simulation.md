---
title: Simulation
parent: Architecture
nav_order: 2
---

# Unified High‑Speed Functional Simulation with Event‑Driven Semantics

The simulation subsystem models both hardware and non‑hardware entities in a unified discrete‑event environment. It is built on top of the `simpy` engine, which provides deterministic event scheduling, concurrency modeling, and precise control over simulated time.

## Core Simulation Classes

Two primary abstractions structure the simulation:

- **`Simulation`** — The runtime coordination object. It owns the `simpy` environment, maintains a registry of all `SimObj` instances, and drives the standard three-phase simulation lifecycle via `run_sim()`.
- **`SimObj`** — Base class for any entity that participates in the simulation. This includes hardware modules (`HwObj`), software processes, sensors, wireless channels, robots, or any physical or logical component that produces or consumes transactions. Each `SimObj` registers itself with a `Simulation` during construction.

## Simulation Lifecycle

Every `SimObj` participates in a three-phase lifecycle managed by `Simulation.run_sim()`:

1. **`pre_sim()`** — Called on all registered objects before the event loop starts. Use this for per-object setup, validation, or initial event scheduling. The default implementation is a no-op.
2. **`run_proc()`** — An optional SimPy generator process. If an object returns a non-`None` generator, it is scheduled via `env.process()`. Returning `None` (the default) marks the object as *passive*; it participates only through `pre_sim()` and `post_sim()`.
3. **`post_sim()`** — Called on all registered objects after the event loop ends. Use this to collect statistics, assert invariants, or emit reports. The default implementation is a no-op.

A minimal active object looks like:

```python
from pysilicon.simulation import Simulation, SimObj

class Producer(SimObj):
    def run_proc(self):
        for _ in range(3):
            yield self.timeout(1)
        self.done = True

sim = Simulation()
p = Producer(sim)
sim.run_sim()
assert p.done
```

## Modeling Hardware Objects

Every `HwObj` automatically derives from `SimObj`, allowing hardware modules to participate directly in the simulation. The simulation behavior of a hardware object is driven by its **transactional interfaces**:

- Each **slave port** corresponds to a process that waits for incoming transactions.
- When a transaction arrives, the associated **action method** is invoked.
- The action may update internal state, schedule future events, or emit transactions on master ports.
- Each action may include **timing estimates** (latency, pipeline depth, processing time) to approximate cycle‑accurate behavior.

This allows the simulation to reflect realistic hardware behavior while remaining lightweight and deterministic.

## Timing, Concurrency, and Race Detection

The simulation engine supports approximate cycle‑accurate modeling:

- Each action can specify a **processing delay** to represent hardware latency.
- Overlapping actions on the same module can be detected, allowing the simulator to identify **race conditions**, resource conflicts, or illegal concurrent behaviors.
- Interfaces enforce ready/valid or request/response semantics, ensuring that backpressure and flow control are faithfully modeled.

This approach provides a deterministic, event‑driven simulation that mirrors the transactional behavior of the synthesized hardware while remaining fast and scalable.

---
Go to [A Python‑Native Unit Testing Framework for Deterministic, Gated Builds](./unittest.md)
