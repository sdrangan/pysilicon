---
title: Vivado IPI Backend
parent: Future
---

# Vivado IPI Backend

> **Status:** Not implemented. This page describes intended future work.

## Concept

A Vivado IPI backend would extend Waveflow outputs beyond HLS kernel code into block-design integration flows. The intended capability is to map component interfaces and generated artifacts into reproducible IPI assembly steps so users can build larger systems without manual block-diagram wiring.

## Status

Current code generation and examples target HLS-centric flows. There is no implemented backend for IPI packaging, block automation scripts, or end-to-end export into Vivado block designs.

## See also

- [plans/vivado_plan.md](https://github.com/sdrangan/waveflow/blob/main/plans/vivado_plan.md)
- TBD
