---
title: Generating a VCD from Python
parent: Timing Analysis Tools
nav_order: 3
has_children: false
---

# Generating a VCD from Python

This page explains how to capture a VCD (Value Change Dump) file from the
`poly` Vitis HLS kernel using the Python-callable `run_xsim_vcd` API
introduced alongside the existing CLI.

> **Note:** This page is *additive* and complements the existing
> [Extracting VCD Files](vcd.md) guide, which describes the CLI
> (`xsim_vcd`) and the manual steps in detail.

## Pre-requisites

- Vivado / Vitis HLS installed on **Windows** (xsim is Windows-only)
- The RTL co-simulation for `poly` has already been run at least once with
  `cosim_design -trace_level all` so that the simulation directory and
  `.tcl` / `.bat` files exist

## Python API

The `run_xsim_vcd` function wraps the same logic as the CLI entry point
and can be called directly from Python:

```python
from pysilicon.scripts.xsim_vcd import run_xsim_vcd
from pathlib import Path

vcd_path = run_xsim_vcd(
    top="poly",
    comp="pysilicon_poly_proj",
    out="dump.vcd",
    workdir=Path("examples/poly"),
)
print(f"VCD written to: {vcd_path}")
```

### Parameters

| Parameter    | Default           | Description |
|--------------|-------------------|-------------|
| `top`        | *(required)*      | Top-level function name |
| `comp`       | `"hls_component"` | HLS component directory name |
| `out`        | `"dump.vcd"`      | Output filename (written to `vcd/<out>`) |
| `soln`       | `"solution1"`     | Solution subdirectory inside `comp` |
| `trace_level`| `"*"`             | VCD trace level: `"*"` for all signals, `"port"` for port signals only |
| `workdir`    | CWD               | Working directory containing the `comp` folder |

### Return value

Returns a `pathlib.Path` pointing to the written VCD file.

### Errors

- Raises `RuntimeError` on non-Windows platforms
- Raises `FileNotFoundError` if the simulation directory is not found
- Raises `RuntimeError` if xsim fails

## Convenience wrapper on `PolyTest`

`PolyTest` exposes `generate_vcd()` as a thin wrapper:

```python
from poly_demo import PolyTest

test = PolyTest()
vcd_path = test.generate_vcd(output_vcd="dump.vcd", trace_level="*")
```

This is equivalent to calling `run_xsim_vcd` with `top="poly"` and
`comp="pysilicon_poly_proj"` using the `PolyTest.example_dir` as the
working directory.

## CLI (unchanged)

The existing CLI entry point is unchanged:

```bash
xsim_vcd --top poly --comp pysilicon_poly_proj --out dump.vcd
```

See [Extracting VCD Files](vcd.md) for full CLI documentation.
