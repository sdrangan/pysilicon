---
title: Register Map (simple function)
parent: Examples
nav_order: 1
has_children: true
---

# Register Map (simple function)

This is the first end-to-end example in the interface guide. It walks through one PySilicon kernel from the Python specification all the way to RTL co-simulation, using the simplest of the AXI-* interfaces — an **AXI-Lite register map**.

A register map is a small set of named, memory-mapped scalar fields that a host CPU can read or write one at a time over AXI-Lite. For an FPGA kernel, the register map is the control plane: it's where the host passes scalar arguments to the kernel, signals it to start, polls for completion, and reads small scalar results back. Bulk data still flows over AXI-Stream or AXI memory-mapped buses; the register map is for the few control and status values that need an addressable home.

Vitis HLS automatically generates this AXI-Lite slave whenever a kernel function has `#pragma HLS interface s_axilite` on its scalar arguments and on `return`. The Vitis-generated slave includes:

- A user-defined region with one register per scalar argument (allocated by Vitis in declaration order).
- A reserved control region that Vitis adds at offsets `0x00–0x10`: `ap_start`, `ap_done`, `ap_idle`, `ap_ready`, and interrupt enables. The host writes `ap_start` to launch the kernel; the kernel writes `ap_done` when it returns.

PySilicon's [`VitisRegMap`](../../guide/interface/regmap.md) class mirrors this layout exactly: the user declares each register as a Python `RegField` with its data schema and access mode (`R`, `W`, `RW`, `W1C`, `W1S`), and PySilicon prepends the same Vitis control registers automatically. The same declaration drives the Python simulation, the generated HLS pragmas, and the host-side offset map.

In going through this example, you will learn to:

- Declare a `VitisRegMap` of typed registers and wire it into an `HwComponent`.
- Write the kernel's behavior as a Python method that reads the input registers, computes a result, and writes it back to the output register.
- Run a Python simulation in which a separate `HwTestbench` plays the role of the host: it writes `x`, `a`, `b`, asserts `ap_start`, polls `status` until done, and reads `y`.
- Generate the corresponding Vitis HLS C++ from the Python source — both the kernel and the testbench.
- Run the Vitis C-simulation and RTL co-simulation flow against the generated artifacts.
- Compare measured RTL cycle timing against the Python model's cycle estimate.

## Scalar function example

We illustrate the register map with a simple kernel that computes a clipped affine function:

```python
y = max(0, a * x + b)
```

for signed 32-bit integers `a`, `b`, and `x`. You would not normally build hardware for a function this small, but it isolates exactly one concept — the AXI-Lite register map — without any of the streaming or memory-mapped complexity that a real accelerator brings in.

The kernel has three input registers (`x`, `a`, `b`) and one output register (`y`), plus the standard Vitis control plane that wraps any AXI-Lite kernel. A host driver running on the CPU performs the following sequence to exercise it:

1. Write the three inputs to their register offsets.
2. Write `1` to `ap_start` to launch the kernel.
3. Poll `status` (or wait for an interrupt on `ap_done`) until the kernel signals it is finished.
4. Read `y` from its register offset.

The Python simulation in [`simulation.md`](./simulation.md) implements exactly this sequence in SimPy.

### User registers

| Register | Schema | Role |
|---|---|---|
| `x` | `S32` (signed 32-bit) | Argument — value to apply the affine map to |
| `a` | `S32` | Argument — multiplicative coefficient |
| `b` | `S32` | Argument — bias term |
| `y` | `S32` | Result — `max(0, a*x + b)` |

### Vitis-added control registers

Vitis HLS prepends a small fixed control region to every `s_axilite`-controlled kernel; PySilicon's `VitisRegMap` does the same so the Python regmap layout matches the generated AXI-Lite slave one-for-one:

| Register | Access | Role |
|---|---|---|
| `ap_start` | W1S | Host writes `1` to launch the kernel. Auto-cleared by Vitis after it is sampled. |
| `status` | R | Combined `idle / busy / done` indicator. Lets the host poll for completion without an interrupt. |

`ap_start` and the status bits together form Vitis's `ap_ctrl_hs` control protocol. PySilicon exposes them as ordinary register fields so the Python testbench can drive them the same way the host driver will at run time.

## File map

The Python source, build script, and Vitis driver all live in [`examples/regmap/`](../../../examples/regmap/):

- `simp_fun.py` — the `HwComponent` kernel, its `VitisRegMap`, and a SimPy host-side testbench.
- `simp_fun_build.py` — the build DAG: golden Python sim → HLS codegen → Vitis C-sim → C-synth → cosim timing.
- `simp_fun_compute_impl.cpp` — the sticky hand-written body of the kernel's compute hook (the rest of the C++ is generated).
- `timing_diagram.py` — generates a register-trace plot from the co-sim VCD for the synthesis page.
- `run.tcl` — Vitis HLS driver script invoked by the build DAG.

## Register map

| Offset | Register | Access | Meaning |
|---|---|---|---|
| `0x00` | `ap_start` | W1S | Launch the kernel |
| `0x04` | `status` | R | `0=idle`, `1=busy`, `2=done` |
| `0x08` | `x` | RW | Signed input |
| `0x0C` | `a` | RW | Multiply coefficient |
| `0x10` | `b` | RW | Bias term |
| `0x14` | `y` | R | Output result |

The Python `VitisRegMap` declaration in `simp_fun.py` lists exactly these fields in this order; the offsets above are what Vitis HLS allocates and what the host driver uses.

## End-to-end stages

The build DAG in `simp_fun_build.py` chains the standard five-stage pipeline introduced by the [poly example](../poly/):

- **Python golden model** — run the kernel in SimPy and record `y` plus a cycle-accurate timing log.
- **HLS code generation** — emit the Vitis HLS kernel C++ and testbench C++ from the Python source.
- **Vitis C-sim functional verification** — run the generated testbench against the generated kernel and compare its `y` against the Python golden.
- **Vitis C-synth and RTL co-sim timing extraction** — synthesize the kernel, run RTL co-sim, and pull the measured cycle latency out of the cosim report.
- **Timing-diagram generation** — turn the cosim VCD into a register-level trace plot that visually walks through the `ap_start` → `status=busy` → `status=done` handshake.

## Next

- [Python simulation](./simulation.md) — declaring the regmap, running it in SimPy, and the host-side testbench.
- [Synthesis and timing](./synthesis.md) — code generation, Vitis flows, and the RTL-vs-Python cycle comparison.
