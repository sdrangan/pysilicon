#include <cstdint>
#include <string>
#include <stdexcept>

#include "poly.hpp"
#include "include/float32_array_utils_tb.h"
#include "include/streamutils_tb.h"

int main(int argc, char** argv) {
    const std::string data_dir = (argc > 1) ? argv[1] : "data";

    PolyCmdHdr cmd_hdr;
    streamutils::read_uint32_file(cmd_hdr, (data_dir + "/cmd_hdr_data.bin").c_str());

    const int nsamp = cmd_hdr.nsamp;
    float samp_in[MAX_NSAMP] = {};
    float samp_out[MAX_NSAMP] = {};
    float32_array_utils::read_uint32_file_array(samp_in, (data_dir + "/samp_in_data.bin").c_str(), nsamp);

    hls::stream<axis_word_t> in_stream;
    hls::stream<axis_word_t> out_stream;
    static const int pf = float32_array_utils::pf<WORD_BW>();

    cmd_hdr.write_axi4_stream<WORD_BW>(in_stream, false);
    for (int i = 0; i < nsamp; i += pf) {
        const int nrem = nsamp - i;
        const bool tlast = (nrem <= pf);
        float32_array_utils::write_axi4_stream_elem<WORD_BW>(in_stream, samp_in + i, tlast, nrem);
    }

    poly(in_stream, out_stream);

    PolyRespHdr resp_hdr;
    resp_hdr.read_axi4_stream<WORD_BW>(out_stream);

    for (int i = 0; i < nsamp; i += pf) {
        const int nrem = nsamp - i;
        float32_array_utils::read_axi4_stream_elem<WORD_BW>(out_stream, samp_out + i, nrem);
    }

    PolyRespFtr resp_ftr;
    resp_ftr.read_axi4_stream<WORD_BW>(out_stream);

    streamutils::write_uint32_file(resp_hdr, (data_dir + "/resp_hdr_data.bin").c_str());
    float32_array_utils::write_uint32_file_array(samp_out, (data_dir + "/samp_out_data.bin").c_str(), nsamp);
    streamutils::write_uint32_file(resp_ftr, (data_dir + "/resp_ftr_data.bin").c_str());

    return 0;
}
