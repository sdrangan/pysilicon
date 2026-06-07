---
title: Overview
parent: Waveflow
nav_order: 1
has_children: true
---

# Overview

Waveflow is a **Python-native framework for algorithm, hardware, and software co-design**.
Instead of fragmenting a hardware system across algorithm notebooks, architecture spreadsheets,
simulation harnesses, HDL, software bindings, and build scripts, Waveflow describes it once as
**structured, executable Python** — so simulation, generated implementation, software, and
tooling all stay aligned around a single model.

This section covers the *why*, the *what*, and the *how*:

- **[Motivation](./motivation.md)** — the fragmentation problem, why Python, and who it's for.
- **[Key features](./keyfeatures.md)** — what makes Waveflow different, and why it's the right
  harness for AI.
- **[The Waveflow flow](./flow.md)** — the two-loop design methodology, end to end.

New to the project? The fastest way in is the
[basic vectorization example](../examples/basic_vec/) — one multiply-accumulate, bit-exact from
Python to Vitis.
