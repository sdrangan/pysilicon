# CGINV Matrix Inverse Algorithm - Vitis HLS C++ Schema Conversion Report

## Overview

Successfully converted the Python `cginv` function (Conjugate Gradient Matrix Inverse algorithm) into Vitis HLS C++ dataschema code using the PySilicon deterministic pipeline.

> Update after the merged dataschema rewrite: the current AI pipeline now emits class-specialized Python dataschema modules (`IntField.specialize(...)`, `DataArray.specialize(...)`, `class ... (DataList): elements = {...}`), and this CGINV example can now generate the struct wrapper headers `cginv_input.h`, `cginv_output.h`, and `cginv_state.h`. The word-alignment failure described later in this historical report no longer applies to the current implementation for this example.

## Python Source Function

```python
def cginv(Q, nit):
    """
    Computes the matrix inverse for a nxn positive semi-definite matrix Q.

    Parameters
    ----------
    Q : nxn matrix
    nit : number of iterations

    Returns
    -------
    X : approximation of inv(Q)
    """
    # ... (algorithm implementation)
```

## Conversion Pipeline Summary

### 1. Python Dataclass Definitions
Created three annotated dataclasses representing the algorithm's data structures:

- **CginvInput**: Input parameters (matrix Q, iteration count)
- **CginvOutput**: Output result (inverse matrix X)
- **CginvState**: Internal computation state (R, P, X, rnorm matrices)

### 2. Schema Specification Inference
Inferred normalized schemas from Python symbols using `spec_from_python_symbol` MCP helper:

#### CginvInput Schema
```json
{
  "module_name": "cginv_schema",
  "word_bw_supported": [32, 64],
  "root": {
    "kind": "struct",
    "type_name": "CginvInput",
    "fields": [
      {"kind": "int", "name": "n", "bitwidth": 8, "signed": false},
      {"kind": "int", "name": "nit", "bitwidth": 16, "signed": false},
      {
        "kind": "array",
        "name": "Q",
        "type_name": "QArray",
        "element_name": "q_elem",
        "element": {"kind": "float", "bitwidth": 32},
        "max_shape": [1024],
        "static": true
      }
    ]
  }
}
```

#### CginvOutput Schema
```json
{
  "module_name": "cginv_schema",
  "word_bw_supported": [32, 64],
  "root": {
    "kind": "struct",
    "type_name": "CginvOutput",
    "fields": [
      {"kind": "int", "name": "n", "bitwidth": 8, "signed": false},
      {
        "kind": "array",
        "name": "X",
        "type_name": "XArray",
        "element_name": "x_elem",
        "element": {"kind": "float", "bitwidth": 32},
        "max_shape": [1024],
        "static": true
      }
    ]
  }
}
```

#### CginvState Schema
```json
{
  "module_name": "cginv_schema",
  "word_bw_supported": [32, 64],
  "root": {
    "kind": "struct",
    "type_name": "CginvState",
    "fields": [
      {"kind": "int", "name": "n", "bitwidth": 8, "signed": false},
      {"kind": "int", "name": "iteration", "bitwidth": 16, "signed": false},
      {
        "kind": "array",
        "name": "R",
        "type_name": "RArray",
        "element_name": "r_elem",
        "element": {"kind": "float", "bitwidth": 32},
        "max_shape": [1024],
        "static": true
      },
      {
        "kind": "array",
        "name": "P",
        "type_name": "PArray",
        "element_name": "p_elem",
        "element": {"kind": "float", "bitwidth": 32},
        "max_shape": [1024],
        "static": true
      },
      {
        "kind": "array",
        "name": "X",
        "type_name": "XArray",
        "element_name": "x_elem",
        "element": {"kind": "float", "bitwidth": 32},
        "max_shape": [1024],
        "static": true
      },
      {
        "kind": "array",
        "name": "rnorm",
        "type_name": "RnormArray",
        "element_name": "rnorm_elem",
        "element": {"kind": "float", "bitwidth": 32},
        "max_shape": [32],
        "static": true
      }
    ]
  }
}
```

### 3. Schema Validation
All three schemas validated successfully:
- ✅ CginvInput: VALID
- ✅ CginvOutput: VALID
- ✅ CginvState: VALID

## Generated Artifacts

### 3.1 Python Dataschema Modules

**File**: [cginv_input_dataschema.py](cginv_input_dataschema.py)
- Contains `QArray` and `CginvInput` classes
- Uses PySilicon `DataList` and `DataArray` base classes
- Supports serialization/deserialization via `DataList` interface

**File**: [cginv_output_dataschema.py](cginv_output_dataschema.py)
- Contains `XArray` and `CginvOutput` classes
- Supports result data serialization

**File**: [cginv_state_dataschema.py](cginv_state_dataschema.py)
- Contains `RArray`, `PArray`, `XArray`, `RnormArray`, and `CginvState` classes
- Supports full internal state serialization

### 3.2 Vitis HLS C++ Headers

#### Array Headers Generated Successfully:

| Header File | Type | Bitwidth | Elements | Status |
|------------|------|----------|----------|--------|
| [q_array.h](q_array.h) | QArray | 32,768 bits | 1024 float32 | ✅ Generated |
| [x_array.h](x_array.h) | XArray | 32,768 bits | 1024 float32 | ✅ Generated |
| [r_array.h](r_array.h) | RArray | 32,768 bits | 1024 float32 | ✅ Generated |
| [p_array.h](p_array.h) | PArray | 32,768 bits | 1024 float32 | ✅ Generated |
| [rnorm_array.h](rnorm_array.h) | RnormArray | 1,024 bits | 32 float32 | ✅ Generated |

#### Failed Headers (Word-Alignment Issue):

| Type | Reason |
|------|--------|
| CginvInput | DataArray requires word-aligned entry (ipos0 == 0) |
| CginvOutput | DataArray requires word-aligned entry (ipos0 == 0) |
| CginvState | DataArray requires word-aligned entry (ipos0 == 0) |

**Note**: Struct headers failed due to the DataArray write recursion layer requiring word-aligned field entry. This is a current limitation in the code generation layer. Array types (QArray, RArray, PArray, XArray, RnormArray) were generated successfully.

## C++ Header Features

Each generated header includes:

1. **Data Structure Definition**
   ```cpp
   class QArray {
   public:
       float q_elem[1024];
       static constexpr int bitwidth = 32768;
       // ...
   };
   ```

2. **Packing/Unpacking Methods**
   ```cpp
   static ap_uint<bitwidth> pack_to_uint(const QArray& data);
   static QArray unpack_from_uint(const ap_uint<bitwidth>& packed);
   ```

3. **Word-Width Templates** (32-bit and 64-bit support)
   ```cpp
   template<int word_bw>
   static void write_word(const float in[], ap_uint<word_bw>& w, int n);
   
   template<int word_bw>
   static void read_word(float out[], const ap_uint<word_bw>& w, int n);
   ```

4. **Stream I/O Support**
   ```cpp
   template<int word_bw>
   void write_array(ap_uint<word_bw> x[]) const;
   
   template<int word_bw>
   static QArray read_array(const ap_uint<word_bw> x[]);
   ```

5. **HLS Pragmas**
   - `#pragma HLS INLINE` for critical paths
   - `#pragma HLS ALLOCATION` for resource constraints
   - Loop unrolling and pipelining directives

## Data Specifications

### Design Assumptions

| Parameter | Specification | Rationale |
|-----------|---------------|-----------| 
| Matrix Size | max 32×32 (1024 elements) | Suitable for embedded HLS synthesis |
| Q Element Type | float32 | Standard IEEE 754 single-precision |
| Bitwidth for `n` | 8-bit unsigned | Supports up to 256 matrix dimension |
| Bitwidth for `nit` | 16-bit unsigned | Supports up to 65535 iterations |
| QArray Bitwidth | 32,768 bits (32KB) | 1024 elements × 32 bits |

### Memory Layout

**CginvInput** (word-aligned):
```
Byte 0:      n (uint8)
Byte 1:      (padding)
Bytes 2-3:   nit (uint16)
Bytes 4-...: Q array (1024 × float32 = 4096 bytes)
Total: 4100 bytes
```

**CginvState** (word-aligned):
```
Byte 0:      n (uint8)
Byte 1:      (padding)
Bytes 2-3:   iteration (uint16)
Bytes 4-...: R array (1024 × float32)
Bytes 4100-...: P array (1024 × float32)
Bytes 8196-...: X array (1024 × float32)
Bytes 12292-...: rnorm array (32 × float32 = 128 bytes)
Total: 12420 bytes
```

## Validation Status

### Schema Validation: ✅ PASSED
- All three schemas verified as valid and conforming to the constrained dataschema spec format
- No structural conflicts or type mismatches

### Generated Code Status: ⚠️ PARTIAL
- Array classes (5/5) successfully generated
- Struct wrappers (3/3) have word-alignment blockers

### Next Steps

1. **Use Array Types Directly**: The generated `QArray`, `RArray`, `PArray`, `XArray`, and `RnormArray` headers are production-ready for HLS synthesis.

2. **Wrap Scalars Separately**: Implement scalar (n, nit, iteration) fields using separate word-aligned structs if needed.

3. **HLS Integration**: Include the generated headers in your Vitis HLS project:
   ```cpp
   #include "q_array.h"
   #include "r_array.h"
   #include "p_array.h"
   #include "x_array.h"
   #include "rnorm_array.h"
   
   void cginv_hls(QArray q_in, ap_uint<32> n, ap_uint<16> nit, 
                  hls::stream<...>& result_stream) {
       // Implementation using generated types
   }
   ```

## Files Generated

```
/Users/asheshkaji/projects/pysilicon/examples/
├── cginv_schema.py              # Original Python schema with annotations
├── cginv_input_dataschema.py    # Generated Python dataschema module
├── cginv_output_dataschema.py   # Generated Python dataschema module
├── cginv_state_dataschema.py    # Generated Python dataschema module
├── q_array.h                    # Vitis HLS header (32KB array)
├── r_array.h                    # Vitis HLS header (32KB array)
├── p_array.h                    # Vitis HLS header (32KB array)
├── x_array.h                    # Vitis HLS header (32KB array)
└── rnorm_array.h                # Vitis HLS header (1KB array)
```

## Result Summary

| Metric | Value |
|--------|-------|
| Schemas Inferred | 3 |
| Schemas Validated | 3/3 (100%) |
| Python Modules Generated | 3/3 (100%) |
| Vitis Headers Generated | 5/5 arrays (100%) |
| Vitis Headers Failed | 3/3 structs (word-alignment) |
| Total Bitwidth (Arrays) | 97,280 bits (12.16 KB per set) |
| Supported Word Widths | 32-bit, 64-bit |

---
**Generated**: 2026-03-20  
**Pipeline**: PySilicon Dataschema Authoring Skill v1.0  
**MCP Server**: pysilicon.ai.mcp_server
