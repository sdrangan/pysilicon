---
title: VitisRegMap Simple Function
parent: Interfaces
nav_order: 5
has_children: true
---

# `vitis_regmap_simp_fun`

`vitis_regmap_simp_fun` is a teaching-first AXI-Lite example that shows the smallest end-to-end `VitisRegMap` kernel flow that still feels realistic.

The kernel exposes four user registers:

- `x`
- `a`
- `b`
- `y`

plus the Vitis `ap_start` control register and a read-only `status` word.

The default scalar function is:

```text
y = relu(a*x + b)
```

The arithmetic is intentionally isolated in a swappable `compute` hook so future lessons can change the function without rewriting the build flow.

## What this example teaches

1. How host code writes AXI-Lite registers, asserts `ap_start`, and polls a status register.
2. How the same Python source drives Python simulation, HLS code generation, and a Vitis C-sim/C-synth flow.
3. How timing extraction and timing-diagram generation can be first-class build artifacts.

## File map

- `examples/interface/vitis_regmap_simp_fun/simp_fun.py`
- `examples/interface/vitis_regmap_simp_fun/simp_fun_build.py`
- `examples/interface/vitis_regmap_simp_fun/timing_diagram.py`
- `examples/interface/vitis_regmap_simp_fun/run.tcl`

## Register map

| Offset | Register | Access | Meaning |
|---|---|---|---|
| `0x00` | `ap_start` | W1S | Launch the kernel |
| `0x04` | `status` | R | `0=idle`, `1=busy`, `2=done` |
| `0x08` | `x` | RW | Signed input |
| `0x0C` | `a` | RW | Multiply coefficient |
| `0x10` | `b` | RW | Bias term |
| `0x14` | `y` | R | Output result |

## End-to-end stages

- Python golden model
- HLS code generation
- Vitis C-sim functional verification
- Vitis C-synth and RTL co-sim timing extraction
- Timing-diagram generation

## Next

- [Python simulation](./simulation.md)
- [Synthesis and timing](./synthesis.md)
