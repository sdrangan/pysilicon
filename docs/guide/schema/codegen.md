---
title: Code Generation
parent: Data Schemas
nav_order: 7
---

# Auto-generating Vitis HLS Files

A key feature of Waveflow data schemas is automatic generation of Vitis-compatible C++ headers from Python schema definitions. For a full build walkthrough, see the [Build System guide](../build/).

## What gets generated

For each schema class, two files are produced:

- `<schema_name>.h` — synthesizable header used in kernel code.
- `<schema_name>_tb.h` — testbench companion header with file I/O/JSON helpers.

For array element helpers, `ArrayUtilsStep` generates:

- `<elem>_array_utils.h`
- `<elem>_array_utils_tb.h`

## Current build flow (`DataSchemaStep` + `ArrayUtilsStep`)

Schema headers and array helpers are generated through `BuildDag` steps:

```python
from waveflow.build.build import BuildConfig, BuildDag
from waveflow.build.streamutils import StreamUtilsStep
from waveflow.hw.dataschema import DataSchemaStep
from waveflow.hw.arrayutils import ArrayUtilsStep

cfg = BuildConfig(root_dir=example_dir)
dag = BuildDag()
dag.add(StreamUtilsStep(output_dir="include"))
dag.add(DataSchemaStep(PolyCmdHdr, word_bw_supported=[32, 64], include_dir="include"))
dag.add(ArrayUtilsStep(Float32, [32, 64]))
dag.run(cfg)
```

`BuildConfig(root_dir=...)` is the build-root entry point. `ArrayUtilsStep` and `DataSchemaStep` both integrate with DAG dependency resolution.

## Using generated headers in kernel code

```cpp
#include "include/poly_cmd_hdr.h"
#include "include/streamutils_hls.h"

static const int WORD_BW = 32;
using axis_word_t = streamutils::axi4s_word<WORD_BW>;

void poly(hls::stream<axis_word_t>& in_stream,
          hls::stream<axis_word_t>& out_stream) {
#pragma HLS INTERFACE axis port=in_stream
#pragma HLS INTERFACE axis port=out_stream
#pragma HLS INTERFACE ap_ctrl_none port=return

    PolyCmdHdr cmd_hdr;
    streamutils::tlast_status cmd_hdr_tlast;
    cmd_hdr.read_axi4_stream<WORD_BW>(in_stream, cmd_hdr_tlast);
}
```

Serialization helpers are templated on `word_bw`, so switching bus width typically changes only one constant.
