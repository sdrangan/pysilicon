---
title: Motivation
parent: Overview
nav_order: 1
---

# Motivation

AI-assisted hardware generation is useful for local code fragments, but production work still breaks down when architecture, simulation, synthesis, and host integration are managed as separate artifacts. Waveflow is motivated by closing that gap with one Python-defined workflow.

## Why this approach

- **Single-source modeling:** schemas, interfaces, components, and build steps are defined in Python and reused directly.
- **Deterministic builds:** BuildDag-based execution provides explicit dependencies and reproducible outputs.
- **Unified verification loop:** Python simulation, generated C++ testbench flow, and cosim timing checks are connected in one pipeline.
- **Typed control/data surfaces:** schema-based interfaces reduce ad hoc packing logic and protocol drift.

## End-to-end focus

The goal is practical consistency: the same model drives simulation behavior, generated kernel/testbench files, and timing-verdict artifacts. This reduces translation gaps across teams and makes regressions easier to detect.

For features that are not yet shipped, see [Future](../future/).
