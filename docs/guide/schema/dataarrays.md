---
title: Data arrays
parent: Data Schemas
nav_order: 3
has_children: true
---

# Data Arrays

PySilicon supports not only structured headers and control messages, but also efficient transfer of raw data arrays between Python models and Vitis HLS implementations. This section does **not** discuss the `DataArray` schema class itself. Instead, it focuses on the generated **array utilities**, which are used for moving arrays of primitive element types such as `float32` between Python and hardware.

A good example appears in the polynomial evaluation example, where the input samples `samp_in` and output samples `samp_out` are arrays of `float32` values.

In Python, these arrays are typically mapped directly to **NumPy arrays**, which gives two major benefits:

- Access to the full NumPy library for numerical operations
- Fast vectorized execution for golden models and test generation


## Generating Array Utilities

For primitive array element types, PySilicon can generate specialized C++ helper functions for reading and writing packed array data in Vitis HLS. These helpers are generated separately from the schema headers.

For example, to generate array utilities for `float32`, use:

```python
from pysilicon.hw.arrayutils import gen_array_utils

# Element type
Float32 = FloatField.specialize(bitwidth=32)

# Word bitwidths to support for serialization / deserialization methods
word_bw_supported = [32, 64]

# Define a code generation configuration
cfg = CodeGenConfig(root_dir=root_dir, util_dir=include_dir)

# Generate the include file
gen_array_utils(Float32, WORD_BWword_bw_supported, cfg=cfg)
```

This generates a header file containing specialized utility functions for the requested element type and supported interface bit widths.  Specifically, similar to the case of the [code generation for the data lists](./codegen.md), two header files are generated:

- `float32_array_utils.h`:  The main header for use in Vitis synthesizable code. This file contains templated serialization and deserialization routines.
- `float32_array_utils_tb.h`:  A companion header for non-synthesizable Vitis code such as testbenches. This file adds routines for reading/writing files and JSON dumps.

## Using Array Utilities in Vitis HLS

In Vitis HLS, the generated array utilities support two main usage patterns.

### 1. Copying an Entire Array into Local Storage

The first pattern is to read a full array from an interface into a local buffer before processing. This is useful when the algorithm needs random access to the full array, or when the entire dataset must be available before computation can begin.

```cpp

void my_kernel(hls::stream<axis_word_t>& in_stream, hls::stream<axis_word_t>& out_stream) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return
    
    static const int nin = 128, nout= 128;
    float input_array[nin], output_array[nout];

    // Read data from the input stream
    float32_array_utils::read_stream(in_stream, input_array, nin);

    // Computation ...
    // output_array[i] = ...

    // Write  data from the input stream
    float32_array_utils::write_stream(output_array, out_stream, nout);

}
```

This pattern is appropriate when:
- the full dataset is needed before processing starts,
- the algorithm revisits elements multiple times, or
- buffering the array is acceptable.

You can see from the example, again, that all the boilerplate for serializing is removed.

### 2. Processing Data as It Arrives

The second pattern is to read and write data incrementally, one interface word at a time, while processing continues in a streaming manner. This enables:
- lower latency,
- reduced on-chip storage,
- and a more naturally pipelined design.

This approach works when the algorithm can process the data continuously as it arrives, without needing the full array in advance.

In the polynomial example, this is exactly how `samp_in` and `samp_out` are handled in `poly.cpp`. The kernel reads input samples from the AXI4 stream, evaluates the polynomial immediately, and writes output samples back to the stream without storing the full input or output array.

Because one interface word may contain multiple array elements, the generated utilities expose a **packing factor**. In `poly.cpp`, this appears as:

```cpp
static const int pf = float32_array_utils::pf<WORD_BW>();
```

The packing factor `pf` is the number of array elements that fit in one interface word. The code then reads up to `pf` input samples at a time, processes them in parallel, and writes up to `pf` output samples back to the interface.

For example, in the polynomial example:

```cpp
void poly(hls::stream<axis_word_t>& in_stream, hls::stream<axis_word_t>& out_stream) {
#pragma HLS INTERFACE axis port=in_stream
#pragma HLS INTERFACE axis port=out_stream
#pragma HLS INTERFACE ap_ctrl_none port=return

    ...

    // Compute the packing factor 
    static const int pf = float32_array_utils::pf<WORD_BW>();

    // Create a "lane" representing a set of elements that can be read in one word read.
    // The pragmas ensure they are fully accessible all in the same time -- e.g., implemented in flip-flops
    float x_lane[pf];
    float y_lane[pf];
#pragma HLS ARRAY_PARTITION variable=x_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=y_lane complete dim=1

    // Loop over the data
    int nrem = cmd_hdr.nsamp;
    for (int i = 0; i < cmd_hdr.nsamp; i += pf) {

        // Read one "lane"
        float32_array_utils::read_axi4_stream_elem<WORD_BW>(in_stream, x_lane, nrem);

        // Process the elements in the "lane"
        // The loop is unrolled to ensure they are processed in parallel, maximizing throughput
        for (int k = 0; k < pf; ++k) {
#pragma HLS UNROLL
            if (k < nrem) {
                y_lane[k] = eval_poly_horner(cmd_hdr.coeffs.data, x_lane[k]);
            }
        }

        // Write one word to the output
        bool tlast = (nrem <= pf);
        float32_array_utils::write_axi4_stream_elem<WORD_BW>(out_stream, y_lane, tlast, nrem);

        nrem -= pf;
    }
    ...
}

```

This style is especially useful for high-throughput streaming accelerators, since it naturally matches the packed interface width while minimizing buffering.

## Summary

The generated array utilities provide a convenient bridge between:
- **NumPy arrays in Python**, used for modeling and test generation, and
- **packed streaming or memory interfaces in Vitis HLS**, used in synthesized hardware.

They support both:
- full-array transfer into local buffers, and
- word-by-word streaming transfer for low-latency pipelined processing.

This allows PySilicon to handle array data efficiently on both the Python and hardware sides, while preserving a consistent packing convention across the entire flow.