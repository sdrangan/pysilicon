---
title: HwParam
parent: Hardware Components
nav_order: 2
---

# HwParam

## Concept

`HwParam[T]` marks component fields that should be treated as synthesis parameters. These fields behave like Python integers in simulation but preserve parameter identity for code generation.

During `HwComponent.__post_init__`, raw values for `HwParam` fields are wrapped as `HwParamValue`. This wrapper preserves which parameter name produced the value so emitters can substitute template-aware expressions where needed.

## API

- [`HwParam`](../../../pysilicon/hw/hw_component.py)
- [`HwParamValue`](../../../pysilicon/hw/hw_component.py)
- [`HwComponent.__post_init__`](../../../pysilicon/hw/hw_component.py)
- [`HwComponent.__setattr__`](../../../pysilicon/hw/hw_component.py)

## Example

From [`examples/poly/poly.py`](../../../examples/poly/poly.py):

```python
in_bw: HwParam[int] = 32
out_bw: HwParam[int] = 32
aximm_bw: HwParam[int] = 32
```

These parameters drive generated kernel signatures and stream/regmap bitwidths.

## Quick reference

- Declare synthesis parameters as `HwParam[...]`.
- Values are auto-wrapped to `HwParamValue` during construction.
- `HwParam` fields are immutable after construction.
- Use plain fields for mutable runtime state.
- See [Synthesis templating](../synthesis/templating.md) for codegen flow.
