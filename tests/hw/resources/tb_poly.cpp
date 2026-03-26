#include <cstdint>
#include <string>
#include <stdexcept>

#include "poly.hpp"

int main(int argc, char** argv) {
    const std::string data_dir = (argc > 1) ? argv[1] : "data";

    PolyCmdHdr cmd_hdr;
    streamutils::read_uint32_file(cmd_hdr, (data_dir + "/cmd_hdr_data.bin").c_str());

    const int nsamp = cmd_hdr.nsamp;
    SampDataIn samp_in;
    streamutils::read_uint32_file_len(samp_in, (data_dir + "/samp_in_data.bin").c_str(), nsamp);

    hls::stream<axis_word_t> in_stream;
    hls::stream<axis_word_t> out_stream;

    cmd_hdr.write_axi4_stream<WORD_BW>(in_stream, false);
    samp_in.write_axi4_stream<WORD_BW>(in_stream, true, nsamp);

    poly(in_stream, out_stream);

    PolyRespHdr resp_hdr;
    resp_hdr.read_axi4_stream<WORD_BW>(out_stream);

    SampDataOut samp_out;
    samp_out.read_axi4_stream<WORD_BW>(out_stream, nsamp);

    PolyRespFtr resp_ftr;
    resp_ftr.read_axi4_stream<WORD_BW>(out_stream);

    streamutils::write_uint32_file(resp_hdr, (data_dir + "/resp_hdr_data.bin").c_str());
    streamutils::write_uint32_file_len(samp_out, (data_dir + "/samp_out_data.bin").c_str(), nsamp);
    streamutils::write_uint32_file(resp_ftr, (data_dir + "/resp_ftr_data.bin").c_str());

    return 0;
}