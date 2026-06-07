---
title: Motivation
parent: Overview
nav_order: 1
---

# Motivation

## The fragmentation problem

In most hardware research and development, the description of a system is spread across many
disconnected forms:

- Python or MATLAB algorithm models
- architecture sketches and spreadsheets
- simulation harnesses
- ad-hoc interface and packing code
- HLS or RTL implementations
- software bindings
- test and verification infrastructure
- build scripts and documentation

Every design change then has to be re-translated by hand across these layers, and much of the
original system intent is lost along the way. AI-assisted code generation helps with local
fragments, but production work still breaks down when architecture, simulation, synthesis, and
host integration are maintained as separate artifacts.

## A single, executable source of truth

Waveflow explores a different approach: describe the key elements of a hardware system — **data
schemas, interfaces, components, behavior, and build relationships** — as structured Python, so
that simulation, downstream implementation artifacts, software integration, and tooling all stay
aligned around one executable model.

The core thesis is that for many systems — especially domain-specific accelerators — the hardest
problem is **not generating RTL**. It is keeping a coherent, executable specification across
algorithms, architecture, interfaces, simulation, software, implementation, and documentation.
Waveflow is aimed at that broader problem.

## Why Python

Python is already the working language for a large fraction of algorithm development in wireless,
DSP, machine learning, scientific computing, and architecture exploration. Rather than treating
Python as a thin wrapper around external hardware tools, Waveflow uses it as the place where
system structure lives directly: types and schemas, interfaces and transactions, component
hierarchies, simulation behavior, and generated artifacts.

This lowers the barrier for domain experts who are fluent in Python but don't want to commit to a
full RTL implementation just to explore a design.

## Who it is for

- **Hardware architects** exploring system structure and interfaces before RTL lock-in.
- **Researchers in wireless, DSP, and ML** studying algorithm–hardware co-design — a more
  realistic model than a notebook, a faster workflow than RTL-first.
- **Accelerator teams** who need architecture, simulation, software interfaces, and
  implementation flows to stay aligned.
- **Tool builders** who want a structured, machine-readable hardware representation for
  automation and AI-assisted workflows.

Next: [what makes Waveflow different](./keyfeatures.md).
