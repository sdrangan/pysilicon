# `vitis_regmap_simp_fun`

Teaching-first `VitisRegMap` example for a scalar AXI-Lite kernel.

The kernel computes:

```text
y = relu(a*x + b)
```

It is structured so the scalar math lives behind a swappable hook (`compute`) while the surrounding flow stays the same:

1. Python simulation and functional check
2. HLS code generation
3. Vitis C-sim
4. Vitis C-synth + RTL co-sim timing extraction
5. Timing-diagram generation as a normal DAG step

## Quick start

From this directory:

```bash
python simp_fun_build.py --through extract_py_timing
```

Full flow (requires Vitis HLS):

```bash
python simp_fun_build.py --through generate_timing_diagram
```

Useful discovery commands:

```bash
python simp_fun_build.py --list-steps
python simp_fun_build.py --list-artifacts
python simp_fun_build.py --status
```

## Default register map

- `ap_start` — Vitis auto control register at `0x00`
- `status`   — read-only status word (`0=idle`, `1=busy`, `2=done`)
- `x`        — signed 32-bit input
- `a`        — signed 32-bit coefficient
- `b`        — signed 32-bit bias
- `y`        — signed 32-bit output

## Key artifacts

- `data/x.bin`, `data/a.bin`, `data/b.bin`
- `results/sim/y.bin`
- `results/sim/regmap_status.json`
- `results/sim_summary.json`
- `results/py_timing.json`
- `results/verify_csim.json`
- `results/cosim_timing.json`
- `results/timing_verdict.json`
- `results/timing_diagram.svg`
- `results/timing_diagram.json`
