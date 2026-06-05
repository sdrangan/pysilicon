// simp_fun_compute_impl.cpp
// Hand-written scalar math hook for the regmap example.

#include <ap_int.h>

namespace simp_fun_impl {

ap_int<32> compute(ap_int<32> x, ap_int<32> a, ap_int<32> b) {
#pragma HLS INLINE
    ap_int<32> affine = a * x + b;
    return (affine > 0) ? affine : ap_int<32>(0);
}

} // namespace simp_fun_impl
