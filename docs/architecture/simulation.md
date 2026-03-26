---
title: Simulation
parent: Architecture
nav_order: 2
---

# Unified High‑Speed Functional Simulation with Event‑Driven Semantics

The simulation subsystem models both hardware and non‑hardware entities in a unified discrete‑event environment. It is built on top of the `simpy` engine, which provides deterministic event scheduling, concurrency modeling, and precise control over simulated time.

## Core Simulation Classes

Two primary abstractions structure the simulation:

- **`SimObj`** — Represents any active entity in the simulation. This includes hardware modules (`HwObj`), software processes, sensors, wireless channels, robots, or any physical or logical component that produces or consumes transactions.
- **`Environment`** — A container that holds all `SimObj` instances and manages the global `simpy` event loop. It orchestrates time, scheduling, and interactions between objects.

Each `SimObj` registers one or more **processes** with the `simpy` environment. These processes model the concurrent behaviors of the object, such as reacting to incoming transactions, generating data, or interacting with the physical environment.

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
