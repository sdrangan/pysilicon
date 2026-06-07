// Hand-written validation hook for the shared_mem (histogram) example.
//
// Mirrors HistAccel.validate:
// bounds-check ndata/nbins against the compile-time maxes and require the three
// buffer addresses to be word-aligned, selecting a HistError status.  Codegen
// emits the call (`ap_uint<8> status = hist_impl::validate(cmd);`) and the
// early-return on a non-NO_ERROR status; this hook is the datapath.
#include "hist.hpp"

namespace memmgr = waveflow::memmgr;

namespace hist_impl {

ap_uint<8> validate(HistCmd cmd) {
    const int ndata = (int)cmd.ndata;
    const int nbins = (int)cmd.nbins;

    // HistError is a scoped enum; the codegen carries the status as ap_uint<8>
    // (matching the generated hook signature), so convert via the integer value.
    if (ndata <= 0 || ndata > max_ndata)
        return (ap_uint<8>)static_cast<unsigned int>(HistError::INVALID_NDATA);
    if (nbins <= 0 || nbins > max_nbins)
        return (ap_uint<8>)static_cast<unsigned int>(HistError::INVALID_NBINS);
    if (!memmgr::is_word_aligned<32>(cmd.data_addr) ||
        !memmgr::is_word_aligned<32>(cmd.bin_edges_addr) ||
        !memmgr::is_word_aligned<32>(cmd.cnt_addr))
        return (ap_uint<8>)static_cast<unsigned int>(HistError::ADDRESS_ERROR);

    return (ap_uint<8>)static_cast<unsigned int>(HistError::NO_ERROR);
}

}  // namespace hist_impl
