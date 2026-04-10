#include <fstream>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <sstream>
#include <iterator>

#include "hist.hpp"
#include "include/streamutils_tb.h"
#include "include/memmgr_tb.hpp"
#include "include/float32_array_utils_tb.h"
#include "include/uint32_array_utils_tb.h"

// Total memory words: enough for max_ndata floats + (max_nbins-1) edge floats + max_nbins count words.
static const int MEM_SIZE = max_ndata + max_nbins * 2;

int main(int argc, char** argv) {
    const std::string data_dir = (argc > 1) ? argv[1] : "data";

    // Read scalar parameters from the JSON file written by the Python testbench.
    int tx_id = 0, ndata = 0, nbins = 0;
    {
        std::ifstream params_ifs(data_dir + "/params.json");
        if (!params_ifs) {
            throw std::runtime_error("Failed to open params.json for reading.");
        }
        const std::string params_json(
            (std::istreambuf_iterator<char>(params_ifs)),
            std::istreambuf_iterator<char>()
        );

        size_t pos = 0;
        streamutils::json_expect_char(params_json, pos, '{');
        while (true) {
            streamutils::json_skip_ws(params_json, pos);
            if (pos < params_json.size() && params_json[pos] == '}') break;
            const std::string key = streamutils::json_parse_string(params_json, pos);
            streamutils::json_expect_char(params_json, pos, ':');
            const int val = static_cast<int>(streamutils::json_parse_number(params_json, pos));
            if (key == "tx_id")   tx_id = val;
            else if (key == "ndata") ndata = val;
            else if (key == "nbins") nbins = val;
            streamutils::json_skip_ws(params_json, pos);
            if (pos < params_json.size() && params_json[pos] == ',') ++pos;
        }
    }

    // Read input data and bin edges from files written by the Python testbench.
    static float data_buf[max_ndata] = {};
    static float edge_buf[max_nbins] = {};  // holds max_nbins-1 edges; sized to max_nbins to avoid zero-length VLA
    float32_array_utils::read_uint32_file_array(data_buf, (data_dir + "/data_array.bin").c_str(), ndata);
    if (nbins > 1) {
        float32_array_utils::read_uint32_file_array(edge_buf, (data_dir + "/edges_array.bin").c_str(), nbins - 1);
    }

    // Allocate regions in the flat memory array using MemMgr, mirroring the Python Memory.alloc flow.
    static mem_word_t mem[MEM_SIZE] = {};
    pysilicon::memmgr::MemMgr<mem_dwidth> mgr(mem, MEM_SIZE);

    // Word counts for each region (float32 and uint32 are each 1 word at mem_dwidth=32).
    const int nwords_data  = (ndata * 32 + mem_dwidth - 1) / mem_dwidth;
    const int nwords_edges = (nbins > 1) ? ((nbins - 1) * 32 + mem_dwidth - 1) / mem_dwidth : 1;
    const int nwords_count = (nbins  * 32 + mem_dwidth - 1) / mem_dwidth;

    const int data_word_idx  = mgr.alloc(nwords_data);
    const int edge_word_idx  = mgr.alloc(nwords_edges);
    const int count_word_idx = mgr.alloc(nwords_count);

    // Byte addresses used in the HistCmd (AXI4 byte-addressed interface).
    const int bytes_per_word   = mem_dwidth / 8;
    const ap_uint<mem_awidth> data_byte_addr  = data_word_idx  * bytes_per_word;
    const ap_uint<mem_awidth> edge_byte_addr  = edge_word_idx  * bytes_per_word;
    const ap_uint<mem_awidth> count_byte_addr = count_word_idx * bytes_per_word;

    // Populate memory from the input arrays.
    float32_array_utils::write_array<mem_dwidth>(data_buf, mem + data_word_idx, ndata);
    if (nbins > 1) {
        float32_array_utils::write_array<mem_dwidth>(edge_buf, mem + edge_word_idx, nbins - 1);
    }

    // Build the HistCmd in the testbench using the allocated addresses.
    HistCmd cmd;
    cmd.tx_id         = tx_id;
    cmd.data_addr     = data_byte_addr;
    cmd.bin_edges_addr = edge_byte_addr;
    cmd.ndata         = ndata;
    cmd.nbins         = nbins;
    cmd.cnt_addr      = count_byte_addr;

    // Push the command descriptor into the input stream and run the kernel.
    hls::stream<axis_word_t> in_stream;
    hls::stream<axis_word_t> out_stream;
    cmd.write_axi4_stream<stream_dwidth>(in_stream, true);
    hist(in_stream, out_stream, mem);

    // Read the response back from the output stream.
    HistResp resp;
    streamutils::tlast_status resp_tlast = streamutils::tlast_status::no_tlast;
    resp.read_axi4_stream<stream_dwidth>(out_stream, resp_tlast);

    // Read the histogram counts back from memory at the address allocated for counts.
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
