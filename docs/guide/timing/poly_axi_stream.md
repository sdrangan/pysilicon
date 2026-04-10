---
title: Poly AXI4-Stream Timing Analysis
parent: Timing Analysis Tools
nav_order: 4
has_children: false
---

# Poly AXI4-Stream Timing Analysis

This guide explains how to analyze the AXI4-Stream timing of the `poly`
Vitis HLS kernel from an existing VCD file — **without** rerunning RTL
co-simulation.

The source code for everything on this page lives in
`examples/poly/timing_analysis.py`.

## Overview

The `poly` kernel communicates over two AXI4-Stream interfaces:

- **Input stream**: command header followed by input sample data
- **Output stream**: response header, output sample data, and response footer

The timing analysis API:
1. loads a VCD file
2. discovers the clock and AXI4-Stream signals
3. extracts individual bursts from each stream
4. decodes the bursts into `PolyCmdHdr`, `PolyRespHdr`, `PolyRespFtr`, and
   NumPy sample arrays
5. returns everything in a `PolyTimingResult` instance

## The `PolyTimingResult` class

```python
class PolyTimingResult:
    clk_name: str        # full VCD name of the clock signal
    clk_period: float    # estimated clock period in nanoseconds
    in_signals: dict     # AXI4-Stream signal names for the input stream
    out_signals: dict    # AXI4-Stream signal names for the output stream
    bursts_in: list      # raw burst dicts from the input stream
    bursts_out: list     # raw burst dicts from the output stream
    cmd_hdr: PolyCmdHdr  # decoded command header
    x: np.ndarray        # input sample array
    resp_hdr: PolyRespHdr  # decoded response header
    y: np.ndarray          # output sample array
    resp_ftr: PolyRespFtr  # decoded response footer
```

Each `burst` dict has keys `data`, `beat_type`, `start_idx`, and `tstart`.

## Analyzing an existing VCD

```python
import sys
sys.path.insert(0, "examples/poly")   # make poly_demo importable
from timing_analysis import analyze_poly_vcd

result = analyze_poly_vcd("vcd/dump.vcd")

# Decoded command header
print(f"tx_id  = {result.cmd_hdr.val['tx_id']}")
print(f"nsamp  = {result.cmd_hdr.val['nsamp']}")
print(f"coeffs = {result.cmd_hdr.val['coeffs']}")

# Input / output sample arrays
print(f"x = {result.x}")
print(f"y = {result.y}")

# Response footer
print(f"nsamp_read = {result.resp_ftr.val['nsamp_read']}")
print(f"error      = {result.resp_ftr.val['error']}")

# Timing information
print(f"clk_period = {result.clk_period} ns")
print(f"Input bursts:  {len(result.bursts_in)}")
print(f"Output bursts: {len(result.bursts_out)}")
```

Expected output for a standard run with `nsamp=100`:

```
tx_id  = 42
nsamp  = 100
coeffs = [ 1. -2. -3.  4.]
x = [0.  0.010101 0.020202 ... 1.]
y = [ 1.  0.9697... ... 0.]
nsamp_read = 100
error      = 0
clk_period = 10.0 ns
Input bursts:  2
Output bursts: 3
```

## Burst structure

| Burst       | Stream | Contents |
|-------------|--------|----------|
| `bursts_in[0]`  | input  | `PolyCmdHdr` — tx_id, coefficients, nsamp |
| `bursts_in[1]`  | input  | `nsamp` input samples (float32) |
| `bursts_out[0]` | output | `PolyRespHdr` — tx_id echo |
| `bursts_out[1]` | output | `nsamp` output samples (float32) |
| `bursts_out[2]` | output | `PolyRespFtr` — nsamp_read, error code |

## Plotting the timing diagram

```python
import matplotlib
matplotlib.use("Agg")   # omit for interactive display

from timing_analysis import analyze_poly_vcd, plot_poly_timing

result = analyze_poly_vcd("vcd/dump.vcd")

# Full-range timing diagram with color-coded bursts
ax = plot_poly_timing(result, show=True)
```

The plot colors:

- 🟠 **Orange** — header bursts (cmd_hdr, resp_hdr)
- 🟢 **Green** — data bursts (input samples, output samples)
- 🔵 **Blue** — footer burst (resp_ftr)

To zoom into a specific time range, pass `trange=(t_start_ns, t_end_ns)`:

```python
ax = plot_poly_timing(result, trange=(0, 500), show=True)
```

## Using the `PolyTest` convenience wrapper

`PolyTest` exposes `analyze_timing()` as a thin wrapper:

```python
from poly_demo import PolyTest

test = PolyTest()
result = test.analyze_timing("vcd/dump.vcd")
```

## Stable test fixture

A minimal VCD fixture is committed to the repository for use in tests and
documentation examples:

```
tests/fixtures/poly/timing/poly_timing_fixture.vcd
```

This fixture contains a 3-sample transaction (`nsamp=3`) and is **never
overwritten** by regular demo runs.  To load it:

```python
from pathlib import Path
from timing_analysis import analyze_poly_vcd

fixture = Path("tests/fixtures/poly/timing/poly_timing_fixture.vcd")
result = analyze_poly_vcd(fixture)
```

## Generating a VCD

See [Generating a VCD from Python](generating_vcd.md) for how to capture
a fresh VCD using the Python-callable `run_xsim_vcd` API or the CLI.
