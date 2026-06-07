---
title: Overview
parent: Waveflow
nav_order: 1
has_children: true
---

# Overview

Waveflow is a **Python-native framework for hardware, algorithm, and software co-design**.

It is motivated by a common problem in hardware research and development: the overall system description is usually fragmented across multiple disconnected forms:
- Python or MATLAB algorithm models
- architecture sketches and spreadsheets
- simulation harnesses
- ad hoc interface code
- HLS or RTL implementations
- software bindings
- test and verification infrastructure
- build scripts and documentation

As a result, each design change has to be translated manually across layers, and much of the system intent gets lost in the process.

Waveflow explores a different approach: use **structured Python code** to describe the key elements of a hardware system — data schemas, interfaces, components, behavior, and build relationships — so that simulation, downstream implementation artifacts, software integration, and tooling can all stay aligned around one executable model.

## The core thesis

The core thesis of Waveflow is that for many hardware systems, especially domain-specific accelerators, the most important problem is **not just generating RTL**.

The harder problem is maintaining a coherent and executable specification across:
- algorithms
- architecture
- interfaces
- simulation
- software integration
- implementation flows
- documentation and tooling

Waveflow is aimed at that broader problem.

## Why Python

Python is already the working language for a large fraction of algorithm development in:
- wireless systems
- DSP
- machine learning
- scientific computing
- architecture exploration

Waveflow builds on that reality.

Rather than treating Python as a thin scripting wrapper around external hardware tools, Waveflow uses Python as the place where system structure can be represented directly:
- types and schemas
- interfaces and transactions
- component hierarchies
- simulation behavior
- generated artifacts

This lowers the barrier for domain experts who are fluent in Python but may not want to commit immediately to a full RTL implementation when exploring a design.

## Simulation at the right abstraction level

A major design goal of Waveflow is to support **fast executable architecture models**.

Traditional RTL simulation is essential for implementation-level validation, but it is often too slow for early exploration of architectural alternatives. In many research and pre-implementation workflows, what matters first is not every signal transition, but questions such as:

- What throughput can this pipeline sustain?
- Where are the buffering bottlenecks?
- How do interface choices affect system behavior?
- What happens when we change vector widths, batching, or memory organization?
- How do algorithm and hardware choices interact?

Waveflow supports a higher-level simulation point:
- execution at the **event level**
- explicit component and interface behavior
- **vectorized NumPy operations** where appropriate
- **cycle-approximate** timing models rather than signal-exact RTL timing

For many workloads, this can enable simulations that are **orders of magnitude faster than RTL**, while still preserving enough architectural structure to study system tradeoffs meaningfully.

This is especially important for:
- accelerator architecture studies
- wireless and DSP pipelines
- hardware/software partitioning
- queueing, scheduling, and throughput analysis
- design-space exploration before RTL lock-in

## What Waveflow is not

Waveflow should not be understood primarily as:

- a generic text-to-RTL system
- an LLM wrapper for hardware code generation
- a lightweight orchestration layer over coding agents
- a replacement for detailed RTL validation

Those areas are either crowded, insufficiently differentiated, or miss the main technical point.

Instead, Waveflow is better viewed as a **structured executable substrate** for hardware design.  
That substrate can support many downstream activities — including synthesis and AI-assisted tooling — but its value does not depend on prompt engineering or multi-agent orchestration alone.

## Current focus

Today, Waveflow is focused on developing the foundations of this substrate, including:

- typed hardware-facing data schemas
- explicit interface abstractions
- component-level structural modeling
- simulation infrastructure
- early synthesis-oriented flows
- reproducible and traceable artifact generation
- documentation and assistant-friendly representations

These capabilities are still evolving, and not all parts of the end-to-end vision are complete.

## Long-term direction

The long-term direction is a workflow in which a single structured Python codebase can increasingly drive:

- architecture definition
- cycle-approximate simulation
- implementation inputs
- software-facing APIs
- documentation
- examples
- verification scaffolding
- grounded AI assistance

In that future, AI is not the center of the system by itself.  
Instead, AI becomes more useful because it operates on a representation that is typed, structured, executable, and connected to the actual design artifacts.

## Intended users

Waveflow is especially relevant for:

### Hardware architects
who want to explore systems and interfaces before committing to RTL details.

### Domain researchers
in wireless, DSP, signal processing, and ML who want a more realistic hardware model than a notebook, but a faster and more flexible workflow than an RTL-first approach.

### Accelerator developers
who need to keep architecture, simulation, software interfaces, and implementation flows aligned.

### Tool builders
who want a more structured and machine-readable hardware representation for automation and AI-assisted workflows.

## Start here

- [Motivation](./motivation.md)
- [Example](./example.md)
- [Poly tutorial](../examples/stream_inband/)
