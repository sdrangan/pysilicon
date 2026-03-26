#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "poly.hpp"



int main(int argc, char* argv[])  {
 
    // Read the command header from a unit32 file
    PolyCmdHdr cmd_hdr;
    streamutils::read_uint32_file(cmd_hdr, "data/cmd_hdr_data.bin");

    // Read the sample input
    int nsamp = cmd_hdr.nsamp;
    SampDataIn samp_in;
    streamutils::read_uint32_file_len(samp_in, "data/samp_in_data.bin", nsamp);


    //  Instantiate the DUT and stream interfaces
    hls::stream<axis_word_t> in_stream;
    hls::stream<axis_word_t> out_stream;


    // Write input data to the streams
    cmd_hdr.write_axi4_stream<WORD_BW>(in_stream, false);
    samp_in.write_axi4_stream<WORD_BW>(in_stream, true, nsamp);

    // "Call" the DUT which will run like a function until the stream is drained.
    poly(in_stream, out_stream);

    // Read the output stream staring with the RespHeader, then the  output samples, 
    // and finally the RespFooter
    PolyRespHeader resp_hdr;
    resp_hdr.read_axi4_stream<WORD_BW>(out_stream);

    SampDataOut samp_out;
    samp_out.read_axi4_stream<WORD_BW>(out_stream, nsamp);

    PolyRespFtr resp_ftr;
    resp_ftr.read_axi4_stream<WORD_BW>(out_stream);

    // TODO:  Write the response header, output samples, and response footer to files for comparison with python testbenches
    streamutils::write_uint32_file(resp_hdr, "data/resp_hdr_data.bin"); 
    streamutils::write_uint32_file_len(samp_out, "data/samp_out_data.bin", nsamp); 
    streamutils::write_uint32_file(resp_ftr, "data/resp_ftr_data.bin"); 

    
    return 0;


}
