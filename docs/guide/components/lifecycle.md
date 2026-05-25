---
title: Lifecycle
parent: Hardware Components
nav_order: 5
---

# Lifecycle

## Concept

PySilicon simulation objects follow a standard lifecycle: `pre_sim`, `run_proc`, and `post_sim`. `HwComponent` participates in the same lifecycle through inheritance from `Component`/`SimObj`, while synthesis extraction targets selected methods (`run_proc` or `on_start`) depending on component structure.

For regmap-driven kernels, `on_start` is used as the invocation-style body triggered by host `ap_start`. Free-running simulation components usually implement `run_proc` as the long-running process body.

## API

- [`SimObj.pre_sim`](../../../pysilicon/simulation/simobj.py)
- [`SimObj.run_proc`](../../../pysilicon/simulation/simobj.py)
- [`SimObj.post_sim`](../../../pysilicon/simulation/simobj.py)
- [`extract_kernel`](../../../pysilicon/build/hwcodegen.py) method-selection policy
- [`@sim_only`](../../../pysilicon/hw/synth.py)

## Example

From [`examples/poly/poly.py`](../../../examples/poly/poly.py), `PolyAccelComponent` defines `on_start` for regmap-triggered kernel entry and a `@sim_only` helper (`_inc_job`) excluded from synthesis extraction.

## Quick reference

- `pre_sim`: setup/validation before event loop.
- `run_proc`: free-running or passive component process entry.
- `on_start`: regmap launch body for invocation-style kernels.
- `post_sim`: final checks/reporting.
- Mark non-synthesizable helpers with `@sim_only`.
