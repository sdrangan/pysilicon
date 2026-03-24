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

    // First read the command header
    PolyCmdHeader cmd_header;
    cmd_header.read_axi4_stream<WORD_BW>(in_stream);

    // Create and write the response header
    PolyRespHeader resp_header;
    resp_header.txn_id = cmd_header.txn_id;
    resp_header.write_axi4_stream<WORD_BW>(out_stream, false);


    // Read the input samples where we read pf 
    static const int pf = SampleDataIn::pf<WORD_BW>();
    float x_lane[pf];
    float y_lane[pf];
#pragma HLS ARRAY_PARTITION variable=x_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=y_lane complete dim=1

    int nrem = cmd_header.nsamp;
    for (int i = 0; i < cmd_header.nsamp; i += pf) {

        // Read the pf input samples for the input stream
        SampleDataIn::read_axi4_stream_elem<WORD_BW>(in_stream, x_lane, nrem);

        for (int k = 0; k < pf; ++k) {
#pragma HLS UNROLL
            if (k < nrem) {
                y_lane[k] = eval_poly_horner(
                    cmd_header.coeff, x_lane[k]);
            }
        }

        // Write the output samples, setting tlast on the last beat
        bool tlast = (nrem <= pf);
        SampleDataOut::write_axi4_stream_elem<WORD_BW>(
            out_stream, y_lane, tlast, nrem);

        // Decrement the remaining samples
        nrem -= pf;
    }

    // Write the response footer
    // For now, we do not model any errors
    PolyRespFooter resp_footer;
    resp_footer.ndata_read = cmd_header.nsamp;
    resp_footer.err_code = PolyError::NO_ERROR;
    resp_footer.write_axi4_stream<WORD_BW>(out_stream);
}
