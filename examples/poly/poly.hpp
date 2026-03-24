#ifndef POLY_HPP
#define POLY_HPP

#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>

#include "poly_error.h"
#include "coeffs.h"
#include "poly_cmd_header.h"
#include "poly_resp_footer.h"
#include "poly_resp_header.h"
#include "sample_data_in.h"
#include "sample_data_out.h"
#include "streamutils.h"


static const int WORD_BW = 32;
// To build a 64-bit variant, set WORD_BW to 64.
static_assert(WORD_BW == 32 || WORD_BW == 64, "WORD_BW must be 32 or 64");

using axis_word_t = hls::axis<ap_uint<WORD_BW>, 0, 0, 0>;

static float eval_poly_horner(const float coeff[4], float x);

#endif