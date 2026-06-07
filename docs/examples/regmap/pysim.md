---
title: Python Simulation
parent: Register Map (simple function)
nav_order: 3
has_children: false
---
# Python simulation

We now show how to simulate the Python model. Python simulation is fast and easy to debug, so we usually perform the simulation in Python before synthesizing and simulating the RTL.

The interesting part of this page is *not* the SimPy simulation itself — that happens inside `simulate_case()` in [`simp_fun.py`](../../../examples/regmap/simp_fun.py) and was covered on the [Python model](./python.md) page. What this page covers is how the simulation is wrapped into a **build DAG** so its inputs, outputs, and timing measurements become first-class artifacts that downstream stages (Vitis C-sim, C-synth, RTL cosim, timing validation) can consume.

## Creating a Build DAG

A **build DAG** is Waveflow's pipeline abstraction: a directed acyclic graph of typed steps where each step declares the named artifacts it **consumes** and **produces**, and the DAG handles execution order, incremental rebuilds, and partial-pipeline runs. The same DAG instance carries the design from input generation through Python sim, codegen, Vitis C-sim, C-synth, and RTL cosim — one pipeline, one entry point. See [Build System](../../guide/build/index.md) for the full reference.

For the Python-only portion of the simp_fun flow, the DAG has three steps:

```python
# examples/regmap/simp_fun_build.py
def build_simp_fun_dag() -> BuildDag:
    dag = BuildDag()
    dag.add(SourceStep(artifact="simp_fun_source",
                       path=_SOURCE_DIR / "simp_fun.py"))
    dag.add(BuildInputsStep(name="build_inputs"))
    dag.add(PySimStep(name="py_sim"))
    dag.add(ExtractPyTimingStep(name="extract_py_timing"))
    ...  # codegen + Vitis stages omitted; covered on the synthesis page
    return dag
```

Each of these steps does one thing:

- **`BuildInputsStep`** writes one test vector (`x`, `a`, `b`) to disk as `.bin` files. These are the inputs the kernel will consume — same files for the Python sim, the C-sim, and the cosim runs, so all three stages are exercising literally the same numbers.
- **`PySimStep`** reads those input files, instantiates a `SimpFunHost` and a `SimpFunComponent`, runs the SimPy simulation, and writes the result (`y.bin`, `regmap_status.json`, `sim_summary.json`) plus a CSV log of timed events.
- **`ExtractPyTimingStep`** parses the CSV log and emits a structured `py_timing.json` with the measured transaction latency in cycles — the artifact the eventual `ValidateTimingStep` will compare against the RTL cosim measurement.

At a high level, running this slice of the DAG performs one test transaction in the Python model end-to-end (host writes inputs → starts the kernel → polls `ap_done` → reads the result) and records both the functional output and the timing as comparable artifacts.

## Build artifacts

In a Waveflow build, an **artifact** is a named output produced by one step and consumed by zero or more downstream steps. Artifacts can be files on disk (most common — a `.bin` blob, a `.json` report, a generated `.cpp`) or in-memory Python objects (a `numpy` array, a parsed dataclass). The DAG tracks who produces what and wires the consumers automatically; you never write `dep_step.add_dependent(self)` by hand.

A step declares its artifact contract as class attributes:

```python
# examples/regmap/simp_fun_build.py
@dataclass(kw_only=True)
class BuildInputsStep(BuildStep):
    description = "Write scalar AXI-Lite inputs for the simp_fun example."
    consumes    = ["simp_fun_source"]
    produces    = {
        "x_in":     Path("data/x.bin"),
        "a_in":     Path("data/a.bin"),
        "b_in":     Path("data/b.bin"),
        "data_dir": Path("data"),
    }
    params = {"x": DEFAULT_VECTOR["x"],
              "a": DEFAULT_VECTOR["a"],
              "b": DEFAULT_VECTOR["b"]}
```

`PySimStep` then declares `consumes = ["simp_fun_source", "x_in", "a_in", "b_in"]`, so the DAG knows it must run `BuildInputsStep` first and inject those paths into `PySimStep.run()` as keyword arguments. The `params` dict holds tunable inputs (here the test vector); the same values can be overridden at the CLI via `--x 5 --a 3 --b -4` or in code via `BuildConfig(params={...})`.

## Primary artifacts

After running the Python-only slice of the DAG, the following artifacts exist on disk:

| Artifact                       | Path                               | Producer                | Role                                                       |
| ------------------------------ | ---------------------------------- | ----------------------- | ---------------------------------------------------------- |
| `x_in` / `a_in` / `b_in` | `data/{x,a,b}.bin`               | `BuildInputsStep`     | Test vector for one transaction                            |
| `sim_dir/y.bin`              | `results/sim/y.bin`              | `PySimStep`           | Kernel output, raw `Int32`                               |
| `sim_dir/regmap_status.json` | `results/sim/regmap_status.json` | `PySimStep`           | `{ap_done, y}` for cross-flow comparison                 |
| `sim_summary`                | `results/sim_summary.json`       | `PySimStep`           | Input tuple, expected `y`, observed `y`, pass/fail bit |
| `log`                        | `results/sim_log.csv`            | `PySimStep`           | Timestamped event log (host + kernel)                      |
| `py_timing`                  | `results/py_timing.json`         | `ExtractPyTimingStep` | Structured cycle-count measurement                         |

The two artifacts worth flagging:

- **`sim_summary.json`** is the human-readable record of one transaction: input tuple, expected output, observed output, pass/fail bit. Useful for inspection; not consumed by downstream automation.
- **`py_timing.json`** is the load-bearing one for the cycle-validation story. It captures the transaction latency in cycles measured from `ap_start_host` to host-observed completion. `ValidateTimingStep` (on the [C and RTL simulation page](./rtlsim.md)) compares it against the RTL cosim's measured cycles to prove the Python model is cycle-approximate.

## Logging and Timing

To get a meaningful cycle measurement out of the Python simulation, both the kernel and the host emit named events to a `Logger` at key moments. The logger writes a CSV row with `time` (in seconds) and `event` (a string label) for each call.

The relevant log points in [`simp_fun.py`](../../../examples/regmap/simp_fun.py):

```python
# Host side (SimpFunHost.run_proc)
self._log("ap_start_host", 1)         # just before writing ap_start
yield from rm.start()
...
self.ap_done = yield from rm.poll_end(...)
self._log("host_done", int(self.ap_done))   # right after poll_end returns

# Kernel side (SimpFunComponent.on_start)
self._log("kernel_busy", 1)           # at hook entry
...
self._log("kernel_done", 1)           # at hook exit
```

`_log` is a thin `@sim_only` wrapper around `self.logger.log(event=..., value=...)`. The decorator tells the codegen extractor "do not lower this method to C++" — these calls only exist in simulation. `Logger` itself adds the simulation time stamp automatically.

`ExtractPyTimingStep` reads back the CSV, picks out the events that bracket the transaction, and emits a structured JSON:

```python
# examples/regmap/simp_fun_build.py — ExtractPyTimingStep.run
t_start = events.get("ap_start_host")
t_end   = events.get("host_done", events.get("kernel_done"))
transaction_seconds = t_end - t_start
transaction_cycles  = int(round(transaction_seconds * clk_freq))
out_path.write_text(json.dumps({
    "transaction_cycles": transaction_cycles,
    "transaction_seconds": transaction_seconds,
    "clk_freq": float(clk_freq),
    "source": "py_sim",
    "events": {"ap_start_host": t_start, "host_done": t_end},
}, indent=2))
```

Cycles are derived from seconds via the configured `clk_freq` — same number Vitis uses in the cosim run, so the resulting cycle counts on both sides are directly comparable. The fallback from `host_done` to `kernel_done` handles the (uncommon) case where the host is hooked up without a poll loop — for the simp_fun flow `host_done` is always present.

## Running the Simulation

Run the Python-only portion of the flow:

```bash
cd examples/regmap
python simp_fun_build.py --through extract_py_timing
```

`--through STEP` is the BuildDag convention for "run everything required by `STEP` and stop." It will execute `build_inputs → py_sim → extract_py_timing` and skip the downstream Vitis stages — useful for fast iteration when you only want to validate the Python model. To run the full flow including code generation and Vitis, omit the flag (or pass `--through validate_timing`).

After the run, inspect the artifacts directly:

```bash
cat results/sim_summary.json
cat results/py_timing.json
```

The summary should show `"passed": true` and your observed `y` matching `expected_y`; the timing JSON should report `transaction_cycles` consistent with `latency_cycles + poll_interval_cycles * N` for whatever vector was simulated.

## Next

- [Vitis HLS Code Generation](./codegen.md) — generating the Vitis HLS kernel and testbench C++ from the same Python source.
- [C and RTL Simulation](./rtlsim.md) — running the Vitis flows and validating that the measured cosim cycles match the `py_timing.json` from this page.
