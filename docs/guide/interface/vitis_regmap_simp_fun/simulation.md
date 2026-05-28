---
title: Python Simulation
parent: VitisRegMap Simple Function
nav_order: 1
---

# Python simulation

Run the Python-only portion of the flow:

```bash
cd examples/interface/vitis_regmap_simp_fun
python simp_fun_build.py --through extract_py_timing
```

## Transaction narrative

For the default case:

1. The host writes `x`, `a`, and `b` over AXI-Lite.
2. The host writes `1` to `ap_start`.
3. The kernel sets `status=busy`.
4. After the configured latency, the kernel computes `relu(a*x + b)`.
5. The kernel writes `y` and sets `status=done`.
6. The host polls `status` until the transaction completes, then reads `y`.

## Primary artifacts

- `results/sim/y.bin`
- `results/sim/regmap_status.json`
- `results/sim_summary.json`
- `results/sim_log.csv`
- `results/py_timing.json`

`sim_summary.json` captures the input tuple, expected output, observed output, and pass/fail bit for the Python simulation.

`py_timing.json` captures the cycle count measured from `ap_start_host` to the host-observed completion event.
