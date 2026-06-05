---
title: Fields and Lists
parent: Data Schemas
nav_order: 1
has_children: true
---

# Data Fields and Data Lists

## Example

A representative schema appears in the [polynomial example](../../examples/stream_inband/):

```python
include_dir = "include"

class CoeffArray(DataArray):
    element_type = Float32
    static = True
    ncoeff = 4
    max_shape = (ncoeff,)
    cpp_storage = "raw"
    include_dir = include_dir

class PolyCmdHdr(DataList):
    elements = {
        "tx_id": {
            "schema": IntField.specialize(bitwidth=16, signed=False),
            "description": "Transaction ID",
        },
        "coeffs": {
            "schema": CoeffArray,
            "description": "Polynomial coefficients",
        },
        "nsamp": {
            "schema": IntField.specialize(bitwidth=16, signed=False),
            "description": "Number of samples",
        },
    }
    include_dir = include_dir
```

Key points:

- `DataList` models structured records (field dictionaries with types/descriptions).
- `DataArray` models typed arrays and can be nested inside `DataList` fields.
- Bitwidth and schema typing are explicit and shared across Python and generated C++.

## Creating instances in Python

```python
coeffs = CoeffArray()
coeffs.val = np.array([1.0, -2.0, -3.0, 4.0], dtype=np.float32)

cmd_hdr = PolyCmdHdr()
cmd_hdr.tx_id = 42
cmd_hdr.coeffs = coeffs.val
cmd_hdr.nsamp = 100
```

These schema objects can be serialized directly for simulation interfaces, test vectors, and generated testbench flows.
