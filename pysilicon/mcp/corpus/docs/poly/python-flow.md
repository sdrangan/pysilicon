---
title: Python Flow
parent: Polynomial Accelerator
nav_order: 1
---
# Python Flow

The Python side of the example lives in [examples/poly/poly_demo.py](https://github.com/sdrangan/pysilicon/blob/main/examples/poly/poly_demo.py). It follows the same structure as the regression in [tests/hw/test_dataschema_poly.py](https://github.com/sdrangan/pysilicon/blob/main/tests/hw/test_dataschema_poly.py): define schemas, build test inputs, run a golden model, emit generated headers, and write binary vectors.

## Step 1: Define the schemas

The first task is to define the data structures that represent the accelerator inputs and outputs. In PySilicon, these structures are specified using `DataSchema` classes. The script [examples/poly/poly_demo.py](https://github.com/sdrangan/pysilicon/blob/main/examples/poly/poly_demo.py) defines the following schema classes:

- `PolyCmdHdr` for the command header
- `PolyRespHdr` for the early response header
- `PolyRespFtr` for the closing status footer
- `SampDataIn` and `SampDataOut` for the variable-length sample payloads
- `CoeffArray` and `PolyErrorField` as reusable building blocks

For example, the command header schema is defined in Python as:

```python
class CoeffArray(DataArray):
    element_type = F32
    static = True
    max_shape = (4,)
    include_dir = INCLUDE_DIR

class PolyCmdHdr(DataList):
    elements = {
        "tx_id": {
            "schema": U16,
            "description": "Transaction ID",
        },
        "coeffs": {
            "schema": CoeffArray,
            "description": "Polynomial coefficients",
        },
        "nsamp": {
            "schema": U16,
            "description": "Number of samples",
        },
    }
    include_dir = INCLUDE_DIR
```

This example illustrates how interface data structures can be described in compact, declarative Python syntax. Each field has a well-defined type and bit width, and that definition is preserved across both the Python model and the generated Vitis HLS implementation.

## Step 2: Auto-generate the include files

PySilicon can auto-generate an include file such as `poly_cmd_hdr.h` for each `DataSchema` class. Each generated header defines a Vitis HLS C++ data structure together with serialization and deserialization support.

In this example, header generation is driven by a list of schema classes and a call to the `gen_include()` method for each one:

```python
SCHEMA_CLASSES = [
    PolyErrorField,
    CoeffArray,
    PolyCmdHdr,
    PolyRespHdr,
    PolyRespFtr,
    SampDataIn,
    SampDataOut,
]

def generate_headers(example_dir: Path) -> None:
    cfg = CodeGenConfig(root_dir=example_dir, util_dir=INCLUDE_DIR)
    for schema_class in SCHEMA_CLASSES:
        out_path = schema_class.gen_include(cfg=cfg, word_bw_supported=WORD_BW_SUPPORTED)
        print(f"generated {out_path}")
    copy_streamutils(cfg)
```

This step turns the Python schema definitions into hardware-facing C++ interface code without manually rewriting the same structures in a second language.

Later, this flow can be integrated into a more explicit incremental build process, but the example already demonstrates the core idea: define the interface once in Python and reuse it in the generated HLS headers.

## Step 3: Build the golden-model inputs

The helper `build_demo_inputs()` creates:

- a coefficient vector `[1, -2, -3, 4]`
- a command header with `tx_id = 42`
- `nsamp` input samples spanning `[0, 1]`

## Step 4: Run the Python golden model

The helper `polynomial_eval()` computes the expected outputs using the same protocol as the hardware example:

- echo `tx_id` into `PolyRespHdr`
- evaluate the polynomial for each input sample
- set `PolyRespFtr.nsamp_read`
- set `PolyRespFtr.error`

## Step 5: Write the binary vectors

The script writes the command and sample input vectors to `examples/poly/data/` using `write_uint32_file()`.

It also writes the Python golden-model outputs there so you can inspect the exact files used by the example:

- `cmd_hdr_data.bin`
- `samp_in_data.bin`
- `resp_hdr_data.bin`
- `samp_out_data.bin`
- `resp_ftr_data.bin`

## Running the script

Activate the `pysilicon` virtual environment.
Navigate to the `examples/poly` directory.  Then run

```powershell
python examples/poly/poly_demo.py --skip-vitis
```

To include the Vitis C-simulation step:

```powershell
python examples/poly/poly_demo.py
```

To display the golden-model plot:

```powershell
python examples/poly/poly_demo.py --plot
```

---

Go to [implementing the Vitis HLS Kernel](./vitis-kernel.md)
