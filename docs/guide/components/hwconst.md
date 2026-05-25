---
title: HwConst
parent: Hardware Components
nav_order: 3
---

# HwConst

## Concept

`HwConst[T]` marks class-level constants intended to represent compile-time values attached to schema/component definitions. It communicates intent to readers and codegen paths that treat the value as fixed for the class.

Use `HwConst` for structural constants that do not vary across instances (for example static array extents). Use `HwParam` when a value should vary per instance and potentially per generated kernel variant.

## API

- [`HwConst`](../../../pysilicon/hw/hw_component.py)
- [`discover_hw_const(cls)`](../../../pysilicon/hw/hw_component.py)

## Example

Minimal pattern used by array-like schemas:

```python
class CoeffArray(DataArray):
    ncoeff: HwConst[int] = 4
    max_shape = (ncoeff,)
```

## Quick reference

- `HwConst` is class-level, not per-instance.
- Prefer `HwConst` for fixed structural constants.
- Prefer `HwParam` for configurable synthesis knobs.
- Current C++ `static constexpr` emission is deferred.
- In Python simulation, constants are still regular class attributes.
