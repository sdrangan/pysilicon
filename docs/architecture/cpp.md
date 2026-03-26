---
title: Generated C/C++ APIs
parent: Architecture
nav_order: 8
---

# Optional High‑Performance C/C++ APIs Auto‑Generated from Python Semantics

## C/C++ Runtime APIs for Industry Deployment

While Python is the primary interface for hardware control, many industrial environments require C or C++ APIs for reasons of performance, integration with existing codebases, or deployment on embedded systems. PySilicon supports this need without compromising the unified Python‑first design philosophy.

### Python as the Canonical Firmware Specification

The functional description of the hardware interface is always expressed in Python. This includes:

- method signatures for control actions  
- parameter types and validation rules  
- data movement semantics for AXI‑Stream and memory‑mapped interfaces  
- documentation and examples  
- simulator‑compatible behavior  

Python remains the **single source of truth** for firmware semantics. All other representations are generated from this canonical specification.

### Auto‑Generated C/C++ Drivers

From the Python API description, PySilicon can automatically generate a C or C++ driver that mirrors the Python interface:

- identical method names and parameter structures  
- identical semantics for control actions  
- identical data movement behavior  
- identical error handling and validation rules  
- identical documentation  

The C/C++ driver plays the same role for deployment that the RTL plays for hardware: a deterministic, auto‑generated artifact derived from the unified Python specification.

This approach ensures:

- no divergence between Python and C/C++ APIs  
- no hand‑written microcode or register maps  
- no risk of mismatched semantics  
- no duplicated maintenance burden  

The C/C++ API becomes a **drop‑in replacement** for the Python API in performance‑critical or embedded contexts.

### VS Code Support for C/C++

Because the VS Code Extension includes structured metadata describing the API, the extension can provide:

- code completion for C/C++  
- inline documentation  
- example snippets  
- AI‑assisted usage guidance  

The hardware consumer receives the same level of support whether they use Python or C/C++.

### Option for Direct C/C++ Authoring

For teams that require absolute control or have strict certification requirements, PySilicon allows the hardware designer to write the API in C/C++ directly. This eliminates any risk of discrepancies but requires more manual effort.

This option is available but not recommended for most workflows, because:

- it breaks the single‑source‑of‑truth model  
- it increases maintenance burden  
- it complicates simulation and documentation generation  

### Co‑Simulation with Real C/C++ Code

The simulation engine supports co‑simulation with real C/C++ code:

- the Python simulator can call into compiled C/C++ drivers  
- C/C++ code can interact with simulated hardware objects  
- performance‑critical algorithms can be tested in the loop  
- embedded firmware can be validated before deployment  

This enables realistic system‑level testing where:

- Python models the environment  
- hardware is simulated via `HwObj`  
- C/C++ code drives the hardware through the generated API  

This closes the loop between simulation, firmware, and deployment.

### Summary

The C/C++ API layer provides:

- optional high‑performance deployment interfaces  
- deterministic auto‑generation from Python  
- full VS Code support  
- seamless co‑simulation with real C/C++ code  
- no DSLs, no microcode, no manual register maps  

Python remains the authoritative specification, while C/C++ becomes a reliable, generated artifact for industrial use.

---
Go to [Extensible RTL‑Level Verification for Future Regression and Co‑Simulatio](./rtlverification.md)