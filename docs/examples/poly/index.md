---
title: Polynomial Accelerator
parent: Examples
nav_order: 4
has_children: true
---

# Polynomial Accelerator

This example demonstrates an end-to-end PySilicon workflow for a small polynomial accelerator.  
The key idea is that the interface is defined once in Python using PySilicon `DataSchema` classes, and that definition is then reused across software modeling, test-vector generation, and Vitis HLS integration.

With this flow, PySilicon helps reduce drift between Python reference code and hardware-facing C++ interfaces, while making it easier to validate the accelerator against a golden model.

In this example, we:

- Define the request and response schemas in Python using PySilicon `DataSchema` classes
- Auto-generate matching C++ headers for Vitis HLS, including serialization and deserialization support for the data schemas
- Build a Python golden model and generate binary test vectors
- Create a Vitis IP with AXI4-Streaming interfaces using the generated schema serialization logic
- Create a Vitis testbench that reads and writes test vectors produced by the Python golden model
- Run the Vitis testbench from Python using PySilicon `toolchain` utilities
- Compare the Vitis outputs against the Python reference

The full example files are in [examples/poly](https://github.com/sdrangan/pysilicon/tree/main/examples/poly).

The main runnable script is [examples/poly/poly_demo.py](https://github.com/sdrangan/pysilicon/blob/main/examples/poly/poly_demo.py).

## Why this example matters

This example shows how PySilicon can be used to:

- Keep interface definitions consistent across Python and HLS code
- Reuse the same schema description for both modeling and hardware integration
- Generate reproducible test vectors from a Python golden model
- Validate hardware-oriented code against a software reference
- Move toward a more automated hardware/software co-design flow

## Current limitations

This example demonstrates only part of the intended PySilicon workflow. Future versions will add:

- A Python golden model represented as a PySilicon `HwObj` class that defines AXI interfaces and interface actions
- Auto-generation of the Vitis IP and Vitis testbench code from the `HwObj` description

## Polynomial Accelerator Protocol

This example implements a polynomial accelerator with the following protocol:

- The client sends a `PolyCmdHeader` containing polynomial coefficients and execution metadata
- The client then streams `SampleDataIn` input values (`x`)
- The accelerator may return `PolyRespHeader` early, before all inputs are consumed, so the host can correlate timing and transaction state
- The accelerator streams `SampleDataOut` result values (`y`) as they become available
- The transaction ends with `PolyRespFooter`, which contains final counters and error status
## Files used in the example

- [examples/poly/poly_demo.py](https://github.com/sdrangan/pysilicon/blob/main/examples/poly/poly_demo.py) drives the Python flow.
- [examples/poly/poly.hpp](https://github.com/sdrangan/pysilicon/blob/main/examples/poly/poly.hpp) declares the HLS interface and includes the generated schemas.
- [examples/poly/poly.cpp](https://github.com/sdrangan/pysilicon/blob/main/examples/poly/poly.cpp) implements the polynomial kernel.
- [examples/poly/poly_tb.cpp](https://github.com/sdrangan/pysilicon/blob/main/examples/poly/poly_tb.cpp) reads vectors, calls the kernel, and writes outputs.
- [examples/poly/run.tcl](https://github.com/sdrangan/pysilicon/blob/main/examples/poly/run.tcl) runs the Vitis C-simulation flow.

---

Go to [Python flow](./python-flow.md)

