// incr_respond_impl.tpp
// Included from gen/incr.hpp. Types declared there are in scope.
// Do not include this file directly except via the .hpp.
//
// Hand-written body for the templated `incr_impl::respond` hook: emit a
// single (always NO_ERROR) IncrResp on the output stream.

// IncrResp / IncrError are used inside the hook body but are not visible
// through the generated kernel signature, so gen/incr.hpp doesn't
// auto-include them.  Pull them in here — each header has its own guard.
#include "include/incr_resp.h"
#include "include/incr_error.h"

namespace incr_impl {

template <int out_bw>
void respond(hls::stream<streamutils::axi4s_word<out_bw>>& m_out) {
    IncrResp resp;
    resp.status = IncrError::NO_ERROR;
    resp.write_axi4_stream<out_bw>(m_out, true);
}

}  // namespace incr_impl
