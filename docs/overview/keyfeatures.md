---
title: Key features
parent: Overview
nav_order: 2
---

# Key features

Waveflow is built around **executable architecture models** — not RTL-first development, and not
an AI orchestration layer. Its differentiators:

- **Single source of truth** — typed data schemas, transactional interfaces, and composable
  hardware objects describe the system once; simulation, codegen, software, and docs all derive
  from it.
- **Fast, bit-exact, vectorized simulation** — modeling at the **event level** with **vectorized
  NumPy** runs orders of magnitude faster than RTL for architecture exploration, while staying
  **bit-exact** in value.
- **Cycle- and resource-approximate models, calibrated** — fast timing/resource estimates that
  are calibrated against real toolchain runs, so the fast loop stays trustworthy.
- **Typed schemas and interfaces** — define hardware-visible structures and how components
  communicate once, eliminating ad-hoc packing and protocol drift.
- **HLS codegen** — generate synthesizable Vitis HLS (kernel + testbench) from the same component
  model.
- **Deterministic, reproducible builds** — a `BuildDag` gives explicit dependencies and
  repeatable `gen → simulate → synthesize → verify` outputs.
- **Bit-exact verification** — generated hardware is checked against the Python golden,
  bit-for-bit, on the real toolchain.

## The harness for AI

**Waveflow is the harness that makes AI effective for hardware design.** AI agents can generate
HLS and drive design-space exploration — but they're only as good as the substrate they work on,
and raw HDL/HLS gives them none of what they need. Waveflow gives them four:

- **Fast simulation** — vectorized, orders of magnitude faster than RTL, so an agent can try many
  designs *inside its loop* instead of waiting on one toolchain run.
- **Structured architecture** — typed schemas and well-defined interfaces make code generation
  **local**: an agent fills in one component against an explicit interface *contract*, not a
  monolithic kernel it has to get entirely right at once.
- **Deterministic, reproducible builds** — the build graph runs the same way every time, so a
  generated design can be rebuilt, compared, and trusted.
- **Built-in, bit-exact validation** — every result is checkable against the real toolchain, so
  the output is **verified**, not just plausible.

Waveflow is that substrate. AI is a **first-class downstream consumer** of it — a real strength —
but it is grounded by the representation, not the center of it. The codegen pipeline itself is
deterministic (structured `hwgen`, not an LLM), so the substrate stays trustworthy.

Next: [the Waveflow flow](./flow.md).
