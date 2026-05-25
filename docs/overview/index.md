---
title: Overview
parent: PySilicon
nav_order: 1
has_children: true
---

# Overview

PySilicon is a Python-first hardware framework that keeps simulation, synthesis inputs, testbench generation, and verification artifacts in one source tree. Instead of maintaining disconnected models, you define schemas, interfaces, and component behavior once and reuse them throughout the flow.

The project thesis is that Python can be the operational source of truth for hardware-facing software and generated HLS code. Recent poly results make that concrete: Python simulation timing and RTL cosim timing agree within ±4 cycles on the reference kernel.

PySilicon is aimed at teams that want deterministic build steps, typed hardware interfaces, and traceable transitions from model code to generated artifacts.

## Start here

- [Motivation](./motivation.md)
- [Example](./example.md)
- [Poly tutorial](../examples/poly/)
