---
title: Synthesis
parent: Architecture
nav_order: 5
---

# Guided and Reproducible AI‑Driven Synthesis 

The synthesis pipeline translates the unified Python specification into FPGA hardware, firmware, and integration artifacts. While the initial implementation targets Xilinx FPGAs, the abstractions are intentionally designed so that the same flow can later support ASIC backends. The pipeline is divided into two major tasks:

- synthesis of individual hardware objects that require custom RTL, and  
- synthesis of the overall FPGA design, including integration of AMD IP blocks and custom modules.

Both tasks are driven entirely by the `HwObj` definitions, their ports, their functional descriptions, and the system‑level interface graph.

## Synthesis of Custom Hardware Objects

Custom hardware modules are synthesized using Vitis HLS. For each `HwObj` that requires RTL generation, PySilicon constructs a **targeted, deterministic AI prompt** that produces synthesizable C/C++ suitable for HLS.

### Inputs to the HLS Prompt

The prompt is derived from:

- **Port descriptions** of the hardware object, including:
  - port names  
  - directions (master/slave)  
  - protocol type (AXI‑Stream, AXI‑Lite, FIFO, memory‑mapped)  
  - widths, depths, and timing annotations  

- **Functional descriptions** extracted from:
  - the Python action methods associated with each slave port, or  
  - the `forward()` method when the module derives from `torch.nn.Module`

  For PyTorch‑based modules, PySilicon extracts the full computation graph using **PyTorch FX**. FX tracing symbolically executes the `forward()` method and produces a complete, flattened dataflow graph that includes:

  - operations defined directly in the module,
  - operations inherited from parent classes,
  - calls to helper methods,
  - calls to submodules,
  - calls to external functions used inside `forward()`.

  This ensures that even if the module is composed hierarchically or relies on inherited behavior, the **entire functional dataflow** is available as a canonical IR for synthesis. The FX graph becomes the authoritative representation of the computation, enabling deterministic lowering into HLS code and later into RTL.

  - **Transactional semantics** are not specified separately; they are fully determined by the **port types** (AXI‑Stream, AXI‑Lite, FIFO, etc.) and **port directions** (master/slave). These semantics define ready/valid behavior, request/response patterns, and backpressure rules automatically.

- **Timing expectations**, such as:
  - pipeline depth  
  - latency constraints  
  - throughput goals  

### Output of the HLS Prompt

The AI‑generated HLS code includes:

- a synthesizable C/C++ implementation of the module  
- `pragma` annotations for pipelining, interfaces, and resource usage  
- a top‑level function matching the port structure  
- optional testbench scaffolding  
- a deterministic directory structure for Vitis HLS  

The pipeline then invokes Vitis HLS to generate:

- Verilog/VHDL RTL  
- interface adapters  
- latency and resource reports  
- co‑simulation stubs  

Because the prompt is derived from a canonicalized Python specification, the synthesis is **fully regenerable and deterministic**.

## Synthesis of the Overall FPGA

Once all custom modules have been synthesized, PySilicon generates a **Vivado TCL script** that builds the complete FPGA design. This script is derived from the system‑level interface graph and includes:

- instantiation of all hardware objects  
- integration of AMD IP blocks (e.g., AXI SmartConnect, DMA engines, FIFOs)  
- wiring of AXI‑Stream, AXI‑Lite, and memory‑mapped interfaces  
- clock and reset generation  
- address map generation for control interfaces  
- synthesis, implementation, and bitstream generation steps  
- optional export of hardware handoff files (XSA)  

The TCL script is deterministic and can be regenerated at any time from the Python specification, ensuring that:

- hardware, firmware, and runtime APIs remain consistent  
- collaborators can reproduce builds exactly  
- CI/CD pipelines can synthesize hardware automatically  

## Adaptability to ASIC Flows

Although the initial backend targets Xilinx FPGAs, the abstractions are backend‑agnostic:

- `HwObj` defines ports and behavior independent of FPGA‑specific constructs.  
- The interface graph is a generic hardware connectivity model.  
- Functional descriptions are expressed in Python or PyTorch, not HDL.  
- The AI prompt generator can be retargeted to produce:
  - SystemVerilog RTL  
  - Chisel/FIRRTL  
  - TL‑Verilog  
  - high‑level synthesis for ASIC flows  

The system‑level synthesis step can later emit:

- ASIC integration scripts  
- clock‑domain crossing logic  
- memory compiler instantiations  
- floorplanning constraints  

The FPGA flow is simply the first backend.

## Build Graph as the Core of Deterministic Hardware Development

A hardware project in PySilicon is defined by a **directed acyclic build graph (DAG)**, where each node represents a concrete, reproducible action:

- unit tests (hardware, environment, or co‑simulation)
- HLS synthesis of a custom hardware object
- Vivado TCL generation and top‑level synthesis
- documentation generation
- firmware/runtime API generation
- VS Code extension generation

Edges represent **dependencies**: a node can only run if all upstream nodes succeed. This structure provides:

- deterministic builds  
- incremental regeneration  
- early failure detection  
- reproducible CI/CD  
- a clean separation of concerns  

This DAG functions as PySilicon’s “makefile,” but expressed at a higher semantic level.

### Example (conceptual)

```
[Unit Tests for FFT] → [HLS Synthesis of FFT] → [Vivado Integration]
[Unit Tests for RadarEnv] → [System Co-Sim Tests] → [Vivado Integration]
```

If any test or synthesis step fails, the graph halts.  
No partial or inconsistent builds.  
No silent drift between hardware, firmware, and simulation.

---
Go to [Eliminating DSLs with Auto‑Generated Python Firmware and AI‑Grounded Tooling](./firmware.md)

