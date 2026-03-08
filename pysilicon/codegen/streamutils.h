#ifndef STREAMUTILS_H
#define STREAMUTILS_H

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_axi_stream.h>

namespace streamutils {

    /**
     * Reinterprets the 32 bits of a float as an unsigned integer
     * without performing any type truncation or rounding.
     */
    inline uint32_t float_to_uint(float f) {
        union {
            float f_val;
            uint32_t u_val;
        } converter;
        converter.f_val = f;
        return converter.u_val;
    }

    /**
     * Reinterprets a 32-bit unsigned integer as a float.
     * Critical for restoring floating point data from a bitstream.
     */
    inline float uint_to_float(uint32_t u) {
        union {
            uint32_t u_val;
            float f_val;
        } converter;
        converter.u_val = u;
        return converter.f_val;
    }

    /**
     * Helper to write a word to an AXI4-Stream with TLAST support.
     * Sets TKEEP and TSTRB to all-ones by default.
     */
    template<int W>
    void write_axi4_word(hls::stream<hls::axis<ap_uint<W>, 0, 0, 0>> &s, ap_uint<W> data, bool tlast) {
        hls::axis<ap_uint<W>, 0, 0, 0> pkt;
        pkt.data = data;
        pkt.last = tlast;
        pkt.keep = -1; // -1 in ap_uint sets all bits to 1
        pkt.strb = -1;
        s.write(pkt);
    }

} // namespace streamutils

#endif // STREAMUTILS_H