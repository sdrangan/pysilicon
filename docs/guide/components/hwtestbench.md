---
title: HwTestbench
parent: Hardware Components
nav_order: 4
---

# HwTestbench

## Concept

`HwTestbench` is a `HwComponent` subclass for codegen-source testbenches. Its `main(self)` method is extracted and lowered into C++ testbench code by `HlsCodegenStep` testbench mode.

The v1 model is sequential: blocking file I/O, stream push/pop operations, and `dut.run()` are supported in `main()`. Concurrent SimPy-style stimulus/capture (`env.process(...)`) is not currently supported in this pathway.

## API

- [`HwTestbench`](../../../pysilicon/hw/hw_testbench.py)
- [`main(self)`](../../../pysilicon/hw/hw_testbench.py)
- [`HlsCodegenStep(is_testbench=True)`](../../../pysilicon/build/hwcodegen_steps.py)

## Example

From [`examples/stream_inband/poly.py`](../../../examples/stream_inband/poly.py), `PolyTBHls.main()`:

```python
dut = PolyAccelComponent()
dut.s_in.push(data_hdr)
dut.s_in.push_array(samp_in, count=data_hdr.nsamp)
dut.run()
dut.m_out.pop(resp_hdr)
```

This is the reference sequential pattern for generated C++ testbench mains.

## Quick reference

- Subclass `HwTestbench` for codegen-source TBs.
- Put the test sequence in `main()`.
- Use stream push/pop and regmap helpers directly.
- Keep flow sequential in v1.
- See [Synthesis testbench](../synthesis/testbench.md) for emitter behavior.
