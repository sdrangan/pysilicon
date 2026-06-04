// incr_transform_impl.cpp
// Hand-written body for the non-templated `incr_impl::transform` hook.
// Forward-declared in gen/incr.hpp; compiled as its own translation unit
// (added via add_files in run.tcl).  The increment transform: each of the
// first n words gets +1, in place (the kernel writes the same buffer back).
#include "incr.hpp"

namespace incr_impl {

void transform(ap_uint<32> buf[1024], ap_uint<32> n) {
    for (int i = 0; i < (int)n; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1024
#pragma HLS PIPELINE II=1
        buf[i] = buf[i] + 1;
    }
}

}  // namespace incr_impl
