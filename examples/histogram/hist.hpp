#ifndef HIST_HPP
#define HIST_HPP

#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>

#include "include/hist_cmd.h"
#include "include/hist_resp.h"
#include "include/float32_array_utils.h"
#include "include/memmgr.hpp"
#include "include/uint32_array_utils.h"

static const int max_nbins = 32;
static const int max_ndata = 1024;
static const int max_mem_words = max_ndata + max_nbins * 2;

// Stream width in bits for the input and output AXI4-Stream interfaces
static const int stream_dwidth = 32;  

// Memory data and address widths in bits for the AXI4 memory-mapped interfaces
static const int mem_dwidth = 32;
static const int mem_awidth = 64;

static_assert(stream_dwidth == 32 || stream_dwidth == 64, "stream_dwidth must be 32 or 64");
static_assert(mem_dwidth == 32 || mem_dwidth == 64, "mem_dwidth must be 32 or 64");

using axis_word_t = hls::axis<ap_uint<stream_dwidth>, 0, 0, 0>;
using mem_word_t = ap_uint<mem_dwidth>;

void hist(hls::stream<axis_word_t>& in_stream, hls::stream<axis_word_t>& out_stream, mem_word_t* mem);

#endif // HIST_HPP