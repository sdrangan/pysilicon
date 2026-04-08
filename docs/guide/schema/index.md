---
title: Data Schemas
parent: Guide
nav_order: 1
has_children: true
---

# PySilicon Data Schemas

PySilicon’s **data schemas** provide a general method for describing how data is stored in and communicated between different hardware modules. Since PySilicon is Python-based, each data structure’s schema is represented by a `DataSchema` class. The `DataSchema` abstraction provides:

- A detailed specification of data types and fields, including bitwidths—fully compatible with the general precision types supported by Vitis HLS.
- Methods for representing and manipulating data structures in Python. When possible, these types are mapped to NumPy arrays to enable efficient, vectorized processing.
- Methods for automatically generating Vitis HLS-compatible C++ header files. The generated headers define C++ structs matching the schema fields, along with templated serialization/deserialization routines for arbitrary bit widths. Supported interfaces include general arrays, HLS streams, and AXI4 streams.

In this way, data schemas offer a consistent and reliable mapping between Python models and Vitis HLS implementations. The translation is automatic, eliminating manual boilerplate and error-prone hand-written packing/unpacking. As we’ll see, Data Schemas are also central for specifying strongly-typed transactional interfaces between modules.

