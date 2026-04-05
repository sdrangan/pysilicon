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


static void flush_input_to_tlast(hls::stream<axis_word_t>& in_stream) {
    bool done = false;
    while (!done) {
#pragma HLS PIPELINE II=1
        axis_word_t axis_word = in_stream.read();
        done = axis_word.last;
    }
}

void poly(hls::stream<axis_word_t>& in_stream, hls::stream<axis_word_t>& out_stream) {
#pragma HLS INTERFACE axis port=in_stream
#pragma HLS INTERFACE axis port=out_stream
#pragma HLS INTERFACE ap_ctrl_none port=return

    // Read the command header from the input stream and track whether TLAST arrives
    // at the expected boundary for the header payload.
    PolyCmdHdr cmd_hdr;
    streamutils::tlast_status cmd_hdr_tlast = streamutils::tlast_status::no_tlast;
    cmd_hdr.read_axi4_stream<WORD_BW>(in_stream, cmd_hdr_tlast);

    // Return a response header immediately so the test bench can correlate the output
    // stream with the transaction ID from the command.
    PolyRespHdr resp_hdr;
    resp_hdr.tx_id = cmd_hdr.tx_id;
    resp_hdr.write_axi4_stream<WORD_BW>(out_stream, true);

    // Allocate one packed lane of sample storage.  The packing factor depends on the
    // AXI stream word width, so a 64-bit stream processes two float32 samples per word.
    static const int pf = float32_array_utils::pf<WORD_BW>();
    float x_lane[pf];
    float y_lane[pf];
#pragma HLS ARRAY_PARTITION variable=x_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=y_lane complete dim=1

    // Read the sample-data burst, evaluate the polynomial lane-by-lane, and stream the
    // results back out.  The command header and sample payload are now separate TLAST-
    // terminated bursts on the input stream.
    int nsamp_read = 0;
    streamutils::tlast_status samp_in_tlast = streamutils::tlast_status::no_tlast;
    bool read_samples = (cmd_hdr_tlast == streamutils::tlast_status::tlast_at_end) && (cmd_hdr.nsamp > 0);
    for (int i = 0; i < cmd_hdr.nsamp && read_samples; i += pf) {
        const int nrem = cmd_hdr.nsamp - i;
        const int lane_count = (nrem < pf) ? nrem : pf;
        streamutils::tlast_status lane_tlast = streamutils::tlast_status::no_tlast;
        float32_array_utils::read_axi4_stream_elem<WORD_BW>(in_stream, x_lane, lane_tlast, nrem);

        for (int k = 0; k < pf; ++k) {
#pragma HLS UNROLL
            if (k < lane_count) {
                y_lane[k] = eval_poly_horner(cmd_hdr.coeffs.data, x_lane[k]);
            }
        }

        const bool out_tlast = (nrem <= pf);
        float32_array_utils::write_axi4_stream_elem<WORD_BW>(out_stream, y_lane, out_tlast, nrem);

        nsamp_read += lane_count;
        if (lane_tlast == streamutils::tlast_status::tlast_at_end) {
            samp_in_tlast = out_tlast ? streamutils::tlast_status::tlast_at_end : streamutils::tlast_status::tlast_early;
            read_samples = false;
        }
    }

    // Summarize how many samples were consumed and classify any stream-boundary errors
    // after the pipelined loop has completed.
    PolyRespFtr resp_ftr;
    resp_ftr.nsamp_read = nsamp_read;
    resp_ftr.error = PolyError::NO_ERROR;
    bool need_flush = false;
    if (cmd_hdr_tlast == streamutils::tlast_status::tlast_early) {
        resp_ftr.error = PolyError::TLAST_EARLY_CMD_HDR;
    } else if (cmd_hdr_tlast == streamutils::tlast_status::no_tlast) {
        resp_ftr.error = PolyError::NO_TLAST_CMD_HDR;
        need_flush = true;
    } else if (cmd_hdr.nsamp == 0) {
        resp_ftr.nsamp_read = 0;
    } else if (samp_in_tlast == streamutils::tlast_status::tlast_early) {
        resp_ftr.error = PolyError::TLAST_EARLY_SAMP_IN;
    } else if (samp_in_tlast == streamutils::tlast_status::no_tlast) {
        resp_ftr.error = PolyError::NO_TLAST_SAMP_IN;
        need_flush = true;
    } else if (nsamp_read != cmd_hdr.nsamp) {
        resp_ftr.error = PolyError::WRONG_NSAMP;
    }

    // If TLAST has not yet been seen for a malformed input message, drain words until the
    // next TLAST boundary so the following transaction starts aligned on the input stream.
    if (need_flush) {
        flush_input_to_tlast(in_stream);
    }

    // Terminate the response footer with TLAST so the test bench can detect the end of
    // the final response message independently of the payload stream.
    resp_ftr.write_axi4_stream<WORD_BW>(out_stream, true);
}
