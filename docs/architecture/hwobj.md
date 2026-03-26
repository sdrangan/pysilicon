---
title: Hardware Objects
parent: Architecture
nav_order: 1
---

# Python‑Native Hardware Architecture with Explicit Transactional Interface

PySilicon is built around two core Python abstractions that together define the structure, behavior, and connectivity of hardware systems.

## HardwareObject (`HwObj`)

An `HwObj` represents a hardware module. It may correspond to a pre‑built AMD IP block (such as AXI SmartConnect or an FFT engine) or a custom module synthesized by PySilicon. Each `HwObj` declares its own **ports**, which describe the module’s external interfaces:

- Ports are typed (FIFO, AXI‑Stream, AXI‑Lite, AXI‑MM, etc.).
- Each port has a **direction** (`master` or `slave`).
- Ports form the canonical definition used for synthesis, firmware generation, simulation, and documentation.

Each `HwObj` also defines its **functional behavior**, which is triggered by transactions arriving on its slave ports. This behavior can be expressed in two ways:

- As a **Python method** associated with a specific slave port (for example, `on_in_stream(self, data)`).
- As a **PyTorch-style `forward()` method** when the module derives from `nn.Module`, with incoming transaction data mapped to tensors and outputs mapped back to master‑port transactions.

This allows hardware designers to describe computation in a natural Pythonic or PyTorch-native style while still supporting transactional semantics.

## Interface

An `Interface` represents a **transactional connection** between two hardware objects. Each interface explicitly connects:

- a **master port** on one `HwObj`, and  
- a **slave port** on another `HwObj`.

Interfaces are first-class objects in the system graph and define:

- the protocol (FIFO, AXI‑Stream, AXI‑Lite, AXI‑MM, etc.)
- the width, depth, and timing properties
- the master/slave endpoints
- the transactional semantics used in simulation

Interfaces are created during system assembly and connect the ports declared inside each `HwObj`. This hybrid model ensures that:

- each module is self-contained and synthesizable, and  
- the system as a whole is a typed graph of explicit master/slave connections.

## Functional Description on Slave Ports

Each **slave port** is associated with an **action** that describes what the hardware object does when a transaction arrives. This action is the core of the functional model:

- For pure Python modules, the action is a method named after the port (for example, `on_ctrl_write`, `on_in_stream`).
- For PyTorch-based modules, the action may call into the module’s `forward()` method, mapping transactional data to tensors and mapping the output tensors back to transactions on master ports.

Example:

```python
class FFT(HwObj):
    in_stream = AxiStreamPort(direction="slave")
    out_stream = AxiStreamPort(direction="master")

    def on_in_stream(self, data):
        y = self.forward(data)
        self.out_stream.send(y)
```

This model ensures that simulation is event-driven and deterministic, synthesis can map actions to RTL processes, firmware protocols can be generated automatically, and runtime APIs reflect the control and data semantics of the module.

---
Go to [Unified High‑Speed Functional Simulation with Event‑Driven Semantics](./simulation.md)
