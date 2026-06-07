---
title: Waveflow
nav_order: 1
has_children: true
---

# Waveflow

**From waveform to silicon — in Python.**

Waveflow is a Python-native framework for **algorithm–hardware co-design** —
describe your datapath once in Python, simulate it *fast* and *bit-exact*, and
generate its hardware implementation from the same source. Its flagship domain is
**wireless**, but it fits any project for efficiently realizing complex
signal-processing algorithms in hardware.

Its core idea is simple: instead of separating algorithm models, architecture descriptions, simulation code, interface definitions, synthesis scripts, software bindings, and documentation into disconnected layers, Waveflow aims to describe them from a **single structured Python representation**.

This makes Waveflow useful not only for hardware developers, but also for domain researchers — in areas such as wireless, DSP, and machine learning — who want to explore how algorithms map onto realistic hardware architectures without immediately dropping into RTL.

## Why Waveflow

Waveflow is built around a different point in the design flow than traditional RTL-first development.

It focuses on **executable architecture models**:
- **Typed data schemas** to define hardware-visible structures once
- **Transactional interfaces** to capture how components communicate
- **Composable hardware objects** that reflect system structure
- **Python-native simulation** for early architectural validation
- **Synthesis-oriented flows** that connect architectural models to implementation toolchains

A key advantage is simulation speed.  
Because Waveflow models systems at the **event level** rather than the raw signal-toggle level, and can apply **vectorized NumPy operations** over data, it can support **cycle-approximate simulations that run orders of magnitude faster than RTL simulation** for many architecture-exploration workloads.

That makes it well suited for:
- hardware/software partitioning studies
- accelerator architecture exploration
- DSP and wireless pipeline design
- interface and buffering studies
- memory, throughput, and scheduling tradeoffs
- early validation before RTL implementation

## What makes it different

Waveflow is **not primarily a text-to-RTL system** and **not just an AI orchestration layer**.

Its main goal is to provide a **structured and executable representation of a hardware system** that can serve as a durable foundation for:

- architecture exploration
- fast simulation
- generated implementation artifacts
- software-facing APIs
- documentation
- verification support
- grounded AI assistance

In this view, AI is important — but it is **downstream of the representation**, not the representation itself.  
The long-term opportunity is that a typed, executable system model is much easier for both people and AI tools to inspect, reason about, validate, and transform than a collection of disconnected notebooks, scripts, HDL fragments, and build glue.

## Who it is for

Waveflow is intended for:

- **Hardware architects** exploring system structure before RTL lock-in
- **Researchers in wireless, DSP, and ML** studying hardware/algorithm co-design
- **Teams building accelerators** who need a software-native architecture model
- **Developers integrating hardware and software** through explicit typed interfaces
- **Future AI-assisted toolchains** that need grounded, structured design context

## What exists today

Waveflow is still an early project, but it already explores several important building blocks:

- typed schema definitions for hardware-facing data
- explicit interface abstractions
- hardware component composition
- simulation infrastructure for executable system models
- early synthesis-oriented flows
- documentation and software-facing artifact generation ideas
- MCP / assistant-facing infrastructure for grounded tooling

These pieces are intended to support a workflow where one Python codebase can express architecture, behavior, interfaces, and downstream artifacts in a more coherent way than a fragmented hardware stack.

## Long-term vision

The long-term vision is to make **Python the operational source of truth** for hardware system design.

In that model, a single structured specification would drive:

- typed architectural definitions
- executable simulation
- cycle-approximate performance studies
- synthesis and implementation inputs
- software bindings and generated APIs
- documentation and examples
- verification scaffolding
- AI-assisted design and analysis workflows

This vision is aspirational and not yet fully built.  
Waveflow today should be understood as a research and prototyping environment for that broader direction.

## Getting Started

If you're new to Waveflow, begin with:

- [Overview](./overview/) — motivation, positioning, and examples
- [Installation](./guide/installation/) — how to install Waveflow from source
- [Guide](./guide/) — schemas, interfaces, components, simulation, synthesis, and build steps

## People

Waveflow is being developed by [Sundeep Rangan](https://wireless.engineering.nyu.edu/sundeep-rangan/), Professor of Electrical and Computer Engineering and Director of [NYU Wireless](https://wireless.engineering.nyu.edu/) at NYU.

## Feedback

Waveflow is an early-stage project and the ideas are still evolving.

Feedback is very welcome — especially from people working in:
- accelerator design
- wireless and DSP systems
- hardware/software co-design
- simulation and verification
- AI-assisted engineering tools
