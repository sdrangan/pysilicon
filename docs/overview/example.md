---
title: Example
parent: Overview
nav_order: 2
---

# Poly example at a glance

The polynomial accelerator is the reference end-to-end example in this repository. It shows how one Python model feeds simulation, generated HLS code, and timing validation.

## 5-stage pipeline

1. **Define schemas + component in Python**

```python
class PolyAccelComponent(HwComponent):
    cpp_kernel_name = "poly"
```

2. **Extract and generate kernel files**

```python
inner_dag.add(HlsCodegenStep(comp_class=PolyAccelComponent, source_artifact="source_dir"))
```

3. **Run C simulation and synthesis steps**

```python
outer_dag.add(CSimStep(...))
outer_dag.add(CSynthStep(...))
```

4. **Run cosim and parse timing**

```python
outer_dag.add(ExtractCosimTimingStep(top=top_name, report_dir_artifact="report_dir"))
```

5. **Validate Python vs RTL timing**

```python
outer_dag.add(ValidateTimingStep(tolerance_cycles=timing_tol_cycles))
```

For the complete walkthrough and artifacts, see [Examples / Poly](../examples/stream_inband/).
