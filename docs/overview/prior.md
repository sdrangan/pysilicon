# Comparison to Existing Work

This section summarizes how PySilicon differs from and extends prior work across several research areas. Google Scholar results (as expected) surface PyMTL and related frameworks, but none provide the unified workflow proposed here.

## Python‑Based Hardware Frameworks (PyMTL, PyRTL, MyHDL, Amaranth)

- Provide Python frontends for hardware modeling or HDL generation.
- Do not generate firmware protocols or runtime APIs.
- Do not unify simulation, synthesis, and runtime tooling.
- Do not provide deterministic AI‑driven regeneration.
- Do not generate VS Code extensions for designers or consumers.

## AI‑Assisted HDL Generation

- Focus on producing Verilog/VHDL from natural language or examples.
- Lack determinism, reproducibility, and integration with simulation.
- Do not generate firmware, runtime APIs, or developer tooling.
- Do not use Python as the authoritative specification.

## Architecture Simulators (gem5, SST, PyMTL Simulation)

- Provide cycle‑accurate or approximate simulation environments.
- Do not synthesize hardware from the same specification.
- Do not generate firmware or runtime APIs.
- Do not unify design and consumption workflows.

## Driver/Firmware Synthesis Tools

- Automatically generate drivers or register interfaces for existing hardware.
- Assume hardware already exists and is manually designed.
- Do not unify with hardware design or simulation.
- Do not generate VS Code tooling or Python APIs.

## Domain‑Specific Accelerators (ML, Robotics, Wireless)

- Produce custom hardware with custom firmware interfaces.
- Require users to learn new DSLs, register maps, or command formats.
- Do not unify design, simulation, synthesis, and runtime use.
- Do not generate consumer‑facing tooling.

---
Go to [Python‑Native Hardware Architecture with Explicit Transactional Interface](./hwobj.md)