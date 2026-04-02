---
title: Fields and Lists
parent: Data Schemas
nav_order: 1
has_children: true
---

# Data Fields and Data Lists

## Example

It is easiest to illustrate data schemas via an example. In the [polynomial evaluation example](../../examples/poly/), a command header is sent to the accelerator to describe the polynomial that should be evaluated on a set of real-valued data samples. This command is defined by the `PolyCmdHdr` data schema in Python:

```python
include_dir = 'include'  # Directory to generate headers

# Coefficients for a cubic polynomial
class CoeffArray(DataArray):
    element_type = Float32
    static = True
    ncoeffs = 4
    max_shape = (ncoeffs,)
    include_dir = include_dir

# Command header schema
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

Key features illustrated here:

- Fields can use standard types (float, arbitrary-precision integers, etc.). See other examples for enums.
- Data schemas can represent *arrays* (like `CoeffArray`) or *lists/structs* (like `PolyCmdHdr`).
- The schema defines exact bitwidth, type, and (optionally) a description for each field.

---

## Creating Instances in Python

Once you have defined a data schema class, you can create and assign values like regular Python objects. For instance, from `poly_demo.py`:

```python
# Example coefficients
coeffs = CoeffArray()
coeffs.val = np.array([1.0, -2.0, -3.0, 4.0], dtype=np.float32)

nsamp = 100  # Number of samples to process 
cmd_hdr = PolyCmdHdr()
cmd_hdr.tx_id = 42
cmd_hdr.coeffs = coeffs.val
cmd_hdr.nsamp = nsamp
```

You can now use these schema instances in your Python model of the accelerator, or for reading/writing structured test data to and from your hardware module.
