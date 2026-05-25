---
title: Hardware Components
parent: Guide
has_children: true
---

# Hardware Components

## Concept

`Component` is the base simulation object with named endpoints and SimPy lifecycle hooks. `HwComponent` extends it with synthesis-aware semantics: extractor-compatible methods, hardware endpoint declarations, and codegen metadata.

Within a component class, fields usually fall into three categories:

- `HwConst[T]` class-level constants for compile-time-style values.
- `HwParam[T]` instance parameters that participate in synthesis templating.
- Plain Python fields for simulation-only state and runtime configuration.

Endpoints are declared in `__post_init__` and attached with `add_endpoint(...)`, typically including stream interfaces (`StreamIFMaster` / `StreamIFSlave`) and AXI-Lite control through regmap-backed endpoints such as `VitisRegMapMMIFSlave`. AXI-MM style interfaces are documented in [Interfaces](../interface/aximm.md).

## API

- [`Component`](../../../pysilicon/hw/component.py)
- [`HwComponent`](../../../pysilicon/hw/hw_component.py)
- [`add_endpoint(endpoint)`](../../../pysilicon/hw/component.py)
- [`HwParam`](./hwparam.md)
- [`HwConst`](./hwconst.md)
- [`HwTestbench`](./hwtestbench.md)

## Example

From [`examples/poly/poly.py`](../../../examples/poly/poly.py), `PolyAccelComponent` declares stream + regmap endpoints in `__post_init__` and registers each endpoint through `add_endpoint(...)`.

## Quick reference

- Use `Component` for simulation-only behavior.
- Use `HwComponent` for synthesizable designs.
- Declare endpoints explicitly in `__post_init__`.
- Keep synthesis knobs in `HwParam` fields.
- Keep compile-time constants in `HwConst` fields.

## In this section

- [HwParam](./hwparam.md)
- [HwConst](./hwconst.md)
- [HwTestbench](./hwtestbench.md)
- [Lifecycle](./lifecycle.md)
