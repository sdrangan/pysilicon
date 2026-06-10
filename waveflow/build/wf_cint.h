#ifndef WAVEFLOW_WF_CINT_H
#define WAVEFLOW_WF_CINT_H

// Waveflow complex-integer element.  std::complex<ap_int> is non-standard, so an
// integer-inner ComplexField maps to this two-ap_int struct (interleaved re/im),
// providing the same (re, im) two-arg construction and .re/.im access the generated
// serialization / arithmetic codegen emits for std::complex<T>.

#include <ap_int.h>

template <int W>
struct wf_cint {
    ap_int<W> re;
    ap_int<W> im;
    wf_cint() : re(0), im(0) {}
    wf_cint(ap_int<W> r, ap_int<W> i) : re(r), im(i) {}
};

#endif  // WAVEFLOW_WF_CINT_H
