---
title: Cycle Model Training
parent: Future
---

# Cycle Model Training

> **Status:** Not implemented. This page describes intended future work.

## Concept

Cycle-model training will fit timing parameters on `HwComponent` models (for example `proc_latency` and `proc_ii`) from measured RTL cosim results. The intended workflow is to run a parameterized workload corpus, collect timing artifacts, and update model parameters so Python simulation tracks hardware timing more closely across variants.

## Status

Ground-truth measurement infrastructure is available via `ExtractPyTimingStep`, `ExtractCosimTimingStep`, and `ValidateTimingStep`, including the measured `delta=4` cycle result on the poly example from PR #31. Automated training, parameter fitting, and feedback loops into component defaults have not started.

## See also

- [plans/project_cycle_model_training.md](https://github.com/sdrangan/waveflow/blob/main/plans/project_cycle_model_training.md)
- [PR #31](https://github.com/sdrangan/waveflow/pull/31)
