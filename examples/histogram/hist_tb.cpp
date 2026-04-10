#include <fstream>
#include <cstdint>
#include <string>
#include <stdexcept>

#include "hist.hpp"
#include "include/streamutils_tb.h"
#include "include/float32_array_utils_tb.h"
#include "include/uint32_array_utils_tb.h"

// Total memory words: enough for max_ndata floats + (max_nbins-1) edge floats + max_nbins count words.
static const int MEM_SIZE = max_ndata + max_nbins * 2;

int main(int argc, char** argv) {
    const std::string data_dir = (argc > 1) ? argv[1] : "data";

    // Read the command descriptor written by the Python testbench.
    HistCmd cmd;
    streamutils::read_uint32_file(cmd, (data_dir + "/cmd_data.bin").c_str());

    const int ndata = static_cast<int>(cmd.ndata);
    const int nbins = static_cast<int>(cmd.nbins);

    // Read input data and bin edges from files written by the Python testbench.
    static float data_buf[max_ndata] = {};
    static float edge_buf[max_nbins] = {};  // holds max_nbins-1 edges; sized to max_nbins to avoid zero-length VLA
    float32_array_utils::read_uint32_file_array(data_buf, (data_dir + "/data_array.bin").c_str(), ndata);
    if (nbins > 1) {
        float32_array_utils::read_uint32_file_array(edge_buf, (data_dir + "/edges_array.bin").c_str(), nbins - 1);
    }

    // Build the flat memory image the kernel will access via its AXI4 memory-mapped interface.
    // Addresses in cmd are byte addresses; with mem_dwidth=32 (4 bytes/word), word_idx = byte_addr / 4.
    static mem_word_t mem[MEM_SIZE] = {};
    const int data_word_idx  = static_cast<int>(cmd.data_addr)      / (mem_dwidth / 8);
    const int edge_word_idx  = static_cast<int>(cmd.bin_edges_addr)  / (mem_dwidth / 8);
    const int count_word_idx = static_cast<int>(cmd.cnt_addr)        / (mem_dwidth / 8);

    // Populate the flat memory array at the addresses specified by the command descriptor.
    float32_array_utils::write_array<mem_dwidth>(data_buf, mem + data_word_idx, ndata);
    if (nbins > 1) {
        float32_array_utils::write_array<mem_dwidth>(edge_buf, mem + edge_word_idx, nbins - 1);
    }

    // Push the command descriptor into the input stream and run the kernel.
    hls::stream<axis_word_t> in_stream;
    hls::stream<axis_word_t> out_stream;
    cmd.write_axi4_stream<stream_dwidth>(in_stream, true);
    hist(in_stream, out_stream, mem);

    // Read the response back from the output stream.
    HistResp resp;
    streamutils::tlast_status resp_tlast = streamutils::tlast_status::no_tlast;
    resp.read_axi4_stream<stream_dwidth>(out_stream, resp_tlast);

    // Read the histogram counts back from memory at the address specified by cmd.
    static ap_uint<32> count_buf[max_nbins] = {};
    uint32_array_utils::read_array<mem_dwidth>(mem + count_word_idx, count_buf, nbins);

    // Write outputs for the Python comparison step.
    streamutils::write_uint32_file(resp, (data_dir + "/resp_data.bin").c_str());
    uint32_array_utils::write_uint32_file_array(count_buf, (data_dir + "/counts_array.bin").c_str(), nbins);

    std::ofstream sync_ofs(data_dir + "/sync_status.json");
    if (!sync_ofs) {
        throw std::runtime_error("Failed to open sync_status.json for writing.");
    }
    sync_ofs
        << "{\n"
        << "  \"resp_tlast\": \"" << streamutils::to_string(resp_tlast) << "\"\n"
        << "}\n";

    return 0;
}
