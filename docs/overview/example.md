# WaveFlow: A Reconfigurable Wireless Processing Chip

WaveFlow is a motivating example that illustrates why PySilicon is needed and what kinds of systems cannot be built reliably with current AI‑assisted hardware tools. WaveFlow represents a class of modern wireless architectures that combine heterogeneous accelerators, high‑rate antenna interfaces, and dynamic dataflow reconfiguration. These systems demand architectural exploration, concurrency modeling, and unified firmware generation—capabilities that existing tools do not provide.

## System Overview

WaveFlow is a tile‑based wireless processing chip designed for real‑time communication, sensing, and beamforming workloads. The architecture consists of:

- heterogeneous processing tiles such as FFT engines, systolic arrays, vector units, and filtering blocks  
- high‑rate antenna array interfaces that ingest wideband I/Q samples from multiple RF chains  
- a message‑passing fabric that routes data between tiles based on workload‑dependent flow graphs  
- flow‑dependent control logic that determines how each tile processes incoming messages and forwards results  
- concurrent dataflows that support communication, spectrum sensing, angle‑of‑arrival estimation, beam nulling, and interference mitigation simultaneously  

WaveFlow is not a fixed pipeline. It is a runtime‑programmable dataflow machine whose behavior changes dynamically based on the wireless environment and application demands.

## Why WaveFlow Demands PySilicon

WaveFlow exposes a set of architectural, concurrency, and firmware challenges that traditional hardware design flows—and current AI‑assisted tools—cannot handle. These challenges are precisely what motivate the design of PySilicon.

- **Dynamic reconfiguration** requires reasoning about multiple possible dataflow graphs rather than a single static pipeline. Existing tools cannot maintain architectural consistency across such variations.
- **Concurrency is fundamental**: multiple flows run simultaneously, sharing tiles, buffers, and fabric bandwidth. Modeling and verifying these interactions requires a unified simulation and synthesis framework.
- **Message‑passing semantics** demand precise handling of backpressure, buffering, and timing. Traditional HLS testbenches and AI‑generated RTL cannot express or validate these behaviors coherently.
- **Microcode DSLs** are typically needed to express tile behavior, but these DSLs are difficult to design, maintain, and expose to end users. WaveFlow’s flow‑dependent control logic highlights the need to eliminate DSLs entirely.
- **Heterogeneous accelerators**—FFTs, systolic arrays, vector units, filtering blocks—must be integrated into a coherent architecture with consistent interfaces. Current tools treat these as isolated modules rather than components of a unified system.
- **Architectural exploration** is essential: tile counts, interconnect topology, buffer sizes, and scheduling policies all affect performance and correctness. Existing tools cannot regenerate hardware, firmware, and simulation consistently across such explorations.

WaveFlow therefore serves as an ideal test case for PySilicon: it requires global architectural control, unified simulation and synthesis, deterministic regeneration, and automatic firmware generation—capabilities that no existing AI‑for‑hardware tool provides.

---

## Dynamic Reconfiguration and Dataflow Variation

WaveFlow must support multiple dataflow graphs that change based on workload demands—beamforming, spectrum sensing, interference mitigation, and communication tasks may all require different tile sequences and routing patterns. Traditional tools cannot maintain architectural consistency across these variations, and AI‑generated RTL cannot be regenerated deterministically when the dataflow graph changes.

PySilicon’s Python‑native hardware objects and deterministic build graph allow WaveFlow’s dataflow variations to be expressed, simulated, and synthesized from a single source of truth.

## Concurrency Across Multiple Wireless Workloads

WaveFlow executes multiple flows concurrently, sharing tiles, buffers, and interconnect bandwidth. Correctness depends on modeling:

- resource contention  
- flow scheduling  
- buffer occupancy  
- timing interactions  

PySilicon’s event‑driven simulation models these behaviors directly using the same hardware objects used for synthesis, enabling concurrency validation long before RTL exists.

## Message‑Passing Semantics and Timing

WaveFlow’s tiles communicate through a message‑passing fabric. Correct behavior depends on:

- message ordering  
- backpressure  
- buffer sizing  
- tile‑to‑tile timing  

Traditional HLS testbenches and AI‑generated RTL cannot express or validate these semantics coherently. PySilicon’s transactional interfaces and event‑driven simulation provide a unified model for both functional correctness and performance behavior.

## Eliminating Microcode DSLs

WaveFlow’s tiles traditionally require:

- custom microcode  
- bespoke command formats  
- device‑specific firmware  

These DSLs are difficult to design, maintain, and expose to end users. PySilicon replaces them with:

- Python control interfaces  
- auto‑generated firmware  
- auto‑generated documentation  
- a VS Code extension for hardware consumers  

This eliminates the DSL barrier entirely and ensures that WaveFlow’s control semantics remain consistent across hardware, firmware, and runtime.

## Integrating Heterogeneous Accelerators

WaveFlow mixes FFT engines, systolic arrays, vector units, and filtering blocks. Current tools treat these as isolated modules, requiring manual glue logic and inconsistent interfaces.

PySilicon treats each accelerator as a Python‑defined `HwObj` with consistent transactional semantics, enabling clean integration, deterministic regeneration, and unified simulation.

## Architectural Exploration Before RTL

WaveFlow requires exploring:

- tile configurations  
- dataflow graphs  
- concurrency patterns  
- bandwidth and latency constraints  
- control semantics  

PySilicon’s Python‑native simulation and build graph enable this exploration long before RTL is generated, ensuring that architectural decisions are validated before synthesis.

## Real‑Time Firmware and C/C++ Control

WaveFlow requires:

- low‑latency configuration  
- runtime flow switching  
- real‑time parameter updates  

PySilicon generates:

- Python APIs  
- optional C/C++ APIs  
- firmware protocols  
- documentation  
- VS Code tooling  

directly from the hardware specification, ensuring that WaveFlow’s runtime behavior is consistent with its architectural model.

---

## Why WaveFlow Cannot Be Built with Current Tools

Current AI‑assisted hardware tools can generate:

- FFT modules  
- FIR filters  
- systolic array kernels  

But they cannot generate:

- a coherent architecture  
- a message‑passing fabric  
- a dynamic dataflow scheduler  
- a unified firmware interface  
- a reproducible build flow  
- a simulation that matches synthesis  
- a system that remains maintainable over time  

WaveFlow requires global architectural control, deterministic regeneration, and unified simulation + synthesis + firmware—the core capabilities of PySilicon.

---

## How WaveFlow Will Be Used in This Documentation

WaveFlow serves as a running example throughout the PySilicon documentation:

- In the **Architecture** section, we show how tiles, ports, and flows map to `HwObj` definitions.  
- In the **Simulation** section, we demonstrate concurrent dataflows and message‑passing behavior.  
- In the **Unit Testing** section, we illustrate gating tests for WaveFlow workloads.  
- In the **Synthesis** section, we show deterministic generation of WaveFlow tiles.  
- In the **Firmware** section, we show how WaveFlow’s microcode DSL is replaced by Python APIs.  
- In the **C/C++** section, we show how real‑time control of WaveFlow is achieved.  
- In the **RTL Verification** section, we outline how WaveFlow’s concurrency can be tested at the RTL level.  

WaveFlow is the concrete system that demonstrates why PySilicon is necessary and what it enables that current tools cannot.

Go to [Prior Work](./prior.md)