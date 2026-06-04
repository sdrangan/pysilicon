// incr_tb.cpp — hand-written testbench for the generated increment kernel.
//
// A trimmed examples/histogram/hist_tb.cpp: allocate one buffer in a flat
// memory array via MemMgr (preserving the Python Memory.alloc order — the
// address is NOT baked in Python, decision 8), populate it from in.bin, run
// the kernel, then read the kernel-produced buffer back and write out.bin for
// the functional-verify step to compare against the Python model (in + 1).
#include <fstream>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <sstream>
#include <iterator>

#include "incr.hpp"
#include "include/streamutils_tb.h"
#include "include/memmgr_tb.hpp"
#include "include/uint32_array_utils_tb.h"

// Flat backing memory size in words.  Compile-time max from the queue/buffer
// bound (MemComponent.nwords_tot == IncrAccel.max_n); static arrays need it.
static const int MEM_SIZE = 1024;
static const int MEM_DWIDTH = 32;

int main(int argc, char** argv) {
    const std::string data_dir = (argc > 1) ? argv[1] : "data";

    // --- Read the element count n from params.json. ---
    int n = 0;
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
            if (key == "n") n = val;
            streamutils::json_skip_ws(params_json, pos);
            if (pos < params_json.size() && params_json[pos] == ',') ++pos;
        }
    }

    // --- Read the input buffer written by the Python testbench. ---
    static ap_uint<32> in_buf[MEM_SIZE] = {};
    uint32_array_utils::read_uint32_file_array(in_buf, (data_dir + "/in.bin").c_str(), n);

    // --- Allocate a region in the flat memory, mirroring Python Memory.alloc. ---
    static ap_uint<32> mem[MEM_SIZE] = {};
    pysilicon::memmgr::MemMgr<MEM_DWIDTH> mgr(mem, MEM_SIZE);

    const int nwords = uint32_array_utils::get_nwords<MEM_DWIDTH>(n);
    const int word_idx = mgr.alloc(nwords);
    const int bytes_per_word = MEM_DWIDTH / 8;
    const ap_uint<64> byte_addr = word_idx * bytes_per_word;

    // Populate memory from the input array.
    uint32_array_utils::write_array<MEM_DWIDTH>(in_buf, mem + word_idx, n);

    // --- Build the command with the allocated address. ---
    IncrCmd cmd;
    cmd.addr = byte_addr;
    cmd.n    = n;

    // --- Push the command and run the kernel. ---
    hls::stream<streamutils::axi4s_word<32>> in_stream;
    hls::stream<streamutils::axi4s_word<32>> out_stream;
    cmd.write_axi4_stream<32>(in_stream, true);
    incr(in_stream, out_stream, mem);

    // --- Read the response back. ---
    IncrResp resp;
    streamutils::tlast_status resp_tlast = streamutils::tlast_status::no_tlast;
    resp.read_axi4_stream<32>(out_stream, resp_tlast);

    // --- Read the kernel-produced buffer back from memory. ---
    static ap_uint<32> out_buf[MEM_SIZE] = {};
    uint32_array_utils::read_array<MEM_DWIDTH>(mem + word_idx, out_buf, n);

    // --- Write outputs for the Python comparison step. ---
    streamutils::write_uint32_file(resp, (data_dir + "/resp_data.bin").c_str());
    uint32_array_utils::write_uint32_file_array(out_buf, (data_dir + "/out_data.bin").c_str(), n);

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
