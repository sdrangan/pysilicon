---
title: Vitis Kernel Implementation
parent: Polynomial Accelerator
nav_order: 2
---

# Vitis Kernel implementation

The HLS kernel is implemented in [`examples/poly/poly.cpp`](https://github.com/sdrangan/pysilicon/blob/main/examples/poly/poly.cpp).

```cpp
#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>

#include "poly.hpp"

static float eval_poly_horner(const float coeff[4], float x) {
#pragma HLS INLINE
    float y = coeff[3];
    y = y * x + coeff[2];
    y = y * x + coeff[1];
    y = y * x + coeff[0];
    return y;
}

void poly(hls::stream<axis_word_t>& in_stream, hls::stream<axis_word_t>& out_stream) {
#pragma HLS INTERFACE axis port=in_stream
#pragma HLS INTERFACE axis port=out_stream
#pragma HLS INTERFACE ap_ctrl_none port=return

    PolyCmdHdr cmd_hdr;
    cmd_hdr.read_axi4_stream<WORD_BW>(in_stream);

    PolyRespHdr resp_hdr;
    resp_hdr.tx_id = cmd_hdr.tx_id;
    resp_hdr.write_axi4_stream<WORD_BW>(out_stream, false);

    static const int pf = SampDataIn::pf<WORD_BW>();
    float x_lane[pf];
    float y_lane[pf];
#pragma HLS ARRAY_PARTITION variable=x_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=y_lane complete dim=1

    int nrem = cmd_hdr.nsamp;
    for (int i = 0; i < cmd_hdr.nsamp; i += pf) {
        SampDataIn::read_axi4_stream_elem<WORD_BW>(in_stream, x_lane, nrem);

        for (int k = 0; k < pf; ++k) {
#pragma HLS UNROLL
            if (k < nrem) {
                y_lane[k] = eval_poly_horner(cmd_hdr.coeffs.data, x_lane[k]);
            }
        }

        bool tlast = (nrem <= pf);
        SampDataOut::write_axi4_stream_elem<WORD_BW>(out_stream, y_lane, tlast, nrem);

        nrem -= pf;
    }

    PolyRespFtr resp_ftr;
    resp_ftr.nsamp_read = cmd_hdr.nsamp;
    resp_ftr.error = PolyError::NO_ERROR;
    resp_ftr.write_axi4_stream<WORD_BW>(out_stream);
}
```

The generated schema classes remove most of the stream-handling boilerplate:

- `cmd_hdr.read_axi4_stream<WORD_BW>(in_stream)` reads the typed command header directly from the AXI4-Stream input
- `resp_hdr.write_axi4_stream<WORD_BW>(out_stream, false)` writes the response header in the correct stream format
- `SampDataIn::read_axi4_stream_elem<WORD_BW>(...)` reads packed sample payload elements into a local array
- `SampDataOut::write_axi4_stream_elem<WORD_BW>(...)` writes result payload elements back to the stream
- `resp_ftr.write_axi4_stream<WORD_BW>(out_stream)` emits the closing footer record

The kernel itself is therefore mostly standard HLS algorithm code plus a small amount of protocol sequencing.

---

Go to [implementing the Vitis HLS testbench](./vitis-tb.md)