# Source: examples/poly/poly.hpp

Original extension: `.hpp`

```cpp
#ifndef POLY_HPP
#define POLY_HPP

#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>

#include "include/poly_error.h"
#include "include/coeff_array.h"
#include "include/poly_cmd_hdr.h"
#include "include/poly_resp_ftr.h"
#include "include/poly_resp_hdr.h"
#include "include/float32_array_utils.h"
#include "include/streamutils_hls.h"


static const int WORD_BW = 32;
// To build a 64-bit variant, set WORD_BW to 64.
static_assert(WORD_BW == 32 || WORD_BW == 64, "WORD_BW must be 32 or 64");

using axis_word_t = hls::axis<ap_uint<WORD_BW>, 0, 0, 0>;

static const int MAX_NSAMP = 128;

static float eval_poly_horner(const float coeff[4], float x);
void poly(hls::stream<axis_word_t>& in_stream, hls::stream<axis_word_t>& out_stream);

#endif
```
