#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>


#include "poly_error.h"
#include "coeffs.h"
#include "poly_cmd_header.h"
#include "poly_resp_footer.h"
#include "poly_resp_header.h"
#include "sample_data_in.h"
#include "sample_data_out.h"
#include "streamutils.h"
#include "poly.hpp"


/**
 * @brief Read up to `nwords` 32-bit words from a binary file.
 *
 * Reads raw `uint32_t` words from `filename` into `data`. If EOF is reached
 * before `nwords` words are available, this returns the number of words that
 * were actually read.
 *
 * @param[in]  filename Path to the input binary file.
 * @param[out] data     Destination buffer with capacity for at least `nwords`.
 * @param[in]  nwords   Maximum number of 32-bit words to read.
 * @param[in]  require_exact_words
 *                       If true, throws when words read does not equal `nwords`.
 * @return Number of words actually read.
 *
 * @throws std::runtime_error If the file cannot be opened or a strict-length
 *                            read is requested and not satisfied.
 */
std::size_t read_uint32_file(
    const std::string& filename,
    uint32_t data[],
    std::size_t nwords,
    bool require_exact_words = false
)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    const std::streamsize bytes_to_read =
        static_cast<std::streamsize>(nwords * sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(data), bytes_to_read);

    const std::streamsize bytes_read = file.gcount();
    if (bytes_read < 0) {
        throw std::runtime_error("Error reading file: " + filename);
    }

    const std::size_t words_read = static_cast<std::size_t>(bytes_read) / sizeof(uint32_t);

    if (require_exact_words && words_read != nwords) {
        throw std::runtime_error(
            "Expected " + std::to_string(nwords) +
            " words but read " + std::to_string(words_read) +
            " from file: " + filename
        );
    }

    // If we did not hit EOF but still failed to read requested bytes, treat as error.
    if (!file.eof() && file.fail() && words_read < nwords) {
        throw std::runtime_error("I/O failure while reading file: " + filename);
    }

    return words_read;
}

/**
 * @brief Write 32-bit words to a binary file.
 *
 * Writes all words from `data` to `filename` as raw `uint32_t` values.
 *
 * @param[in] filename Path to the output binary file.
 * @param[in] data     Source buffer of 32-bit words to write.
 * @param[in] nwords   Number of 32-bit words to write.
 * @throws std::runtime_error If the file cannot be opened or a write fails.
 */
void write_uint32_file(
    const std::string& filename,
    const uint32_t data[],
    std::size_t nwords
)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Unable to open file: " + filename);
    }
    file.write(reinterpret_cast<const char*>(data), nwords * sizeof(uint32_t));
    if (!file) {
        throw std::runtime_error("Error writing file: " + filename);
    }
}

int main(int argc, char* argv[])  {
 
    // Read the command header from a unit32 file
    PolyCmdHeader cmd_hdr;
    streamutils::read_uint32_file(cmd_hdr, "cmd_hdr_data.bin");

    // Read the sample input
    int nsamp = cmd_hdr.nsamp;
    SampleDataIn samp_in;
    streamutils::read_uint32_file_len(samp_in, "samp_in_data.bin", nsamp);


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

    SampleDataOut samp_out;
    samp_out.read_axi4_stream<WORD_BW>(out_stream, nsamp);

    PolyRespFooter resp_ftr;
    resp_ftr.read_axi4_stream<WORD_BW>(out_stream);

    // TODO:  Write the response header, output samples, and response footer to files for comparison with python testbenches
    streamutils::write_uint32_file(resp_hdr, "resp_hdr_data.bin"); 
    streamutils::write_uint32_file_len(samp_out, "samp_out_data.bin", nsamp); 
    streamutils::write_uint32_file(resp_ftr, "resp_ftr_data.bin"); 

    
    return 0;


}
