---
title: Code Generation
parent: Data Schemas
nav_order: 2
has_children: true
---

# Auto-generating Vitis HLS Files

## Generating the Include Files

A key feature of PySilicon's data schemas is that you can **auto-generate the Vitis HLS include files** that correspond to your schema definitions. For example, in the polynomial example introduced in the [Data Lists section](./datalists.md), we generate the include files as follows:

```python
from pysilicon.codegen.build import CodeGenConfig

# Define a code generation configuration
cfg = CodeGenConfig(root_dir=root_dir, util_dir=include_dir)

# Supported bit widths for serialization
word_bw_supported = [32, 64]

# Generate the include files
PolyCmdHdr.gen_include(cfg=cfg, word_bw_supported=word_bw_supported)
```

Running this command generates two include files:

- `poly_cmd_hdr.h`:  The main header for use in Vitis synthesizable code. This file defines the C++ data structure as well as templated serialization and deserialization routines.
- `poly_cmd_hdr_tb.h`:  A companion header for non-synthesizable Vitis code such as testbenches. This file adds routines for reading/writing files and JSON dumps.

For example, the first header, `poly_cmd_hdr.h`, contains:

```cpp
struct PolyCmdHdr {
    ap_uint<16> tx_id;   // Transaction ID
    CoeffArray coeffs;   // Polynomial coefficients
    ap_uint<16> nsamp;   // Number of samples

    ...
};
```

Thus, there is a C++ struct that exactly mirrors the Python data schema.

---

## Serialization and Deserialization Methods

In addition to the structure definition, the auto-generated header includes several methods for serializing and deserializing data using various bit widths. All serialization is to unsigned integers of configurable width, represented in Vitis HLS as:

```cpp
ap_uint<word_bw>
```
where `word_bw` is the bit width (typically 32, 64, 128, etc.).

Commonly generated methods include:

- `write_array(ap_uint<word_bw> x[]) const` – Serializes the data to an array of packed words.
- `write_stream(hls::stream<ap_uint<word_bw>>& s) const` – Serializes the data to an HLS stream for inter-module communication.
- `write_axi4_stream(hls::stream<axis_word_t>& s, bool last = true) const` – Serializes data to an AXI4-Stream using `hls::axis`.

Corresponding read/unpack methods are also generated.

---

## Using the Serialization Methods in Vitis HLS

Once the header files are generated, you can use these serialization methods directly in your Vitis HLS code. For example, in the polynomial example:

```cpp
void poly(hls::stream<axis_word_t>& in_stream, hls::stream<axis_word_t>& out_stream) {
#pragma HLS INTERFACE axis port=in_stream
#pragma HLS INTERFACE axis port=out_stream
#pragma HLS INTERFACE ap_ctrl_none port=return

    // Read the command header from the input stream
    PolyCmdHdr cmd_hdr;
    cmd_hdr.read_axi4_stream<WORD_BW>(in_stream);

    // (Computation...)

    // Write the response header to the output stream
    PolyRespHdr resp_hdr;
    resp_hdr.tx_id = cmd_hdr.tx_id;
    resp_hdr.write_axi4_stream<WORD_BW>(out_stream, false);

    ...
}
```

The data schemas make serialization just one or two lines of code. If the stream's bit width changes, the code does not: simply set the new `WORD_BW`. No manual bit-packing is needed. 

The code is carefully written to maximize throughput and hardware efficiency—in many cases, `#pragma` hints and pipelining are included out-of-the-box by the code generator.

