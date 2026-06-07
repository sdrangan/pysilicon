"""Vectorized Vitis kernels for the basic_vec example — one MAC, ``y = a*b + c``.

Three kernels (int / float / fixed) over the *same* elementwise op, so the
pedagogical parallel is exact. Each reads operand bit-vectors (one value per line),
computes ``a*b + c`` in the typed C++ — **full-precision** product/sum, mirroring the
Python operators — and writes the result bits. Uniform argv ``(in_a, in_b, in_c,
out)``. The Python golden produces identical bits.

These are deliberately minimal and readable (the docs pull from them); the rigorous
all-modes/widths sweep lives in examples/schemas/fixedpoint.
"""
from __future__ import annotations

_INT_READ = """#include <ap_int.h>
#include <fstream>
#include <vector>

static std::vector<unsigned long long> rd(const char* p) {
    std::ifstream f(p); std::vector<unsigned long long> v; unsigned long long x;
    while (f >> x) v.push_back(x); return v;
}
"""


def render_int_mac(wa: int, wb: int, wc: int, wy: int) -> str:
    """``ap_int<wy> y = a*b + c`` (signed); full-precision product + sum."""
    return _INT_READ + f"""
int main(int argc, char** argv) {{
    auto A = rd(argv[1]); auto B = rd(argv[2]); auto C = rd(argv[3]);
    std::ofstream out(argv[4]);
    for (size_t i = 0; i < A.size(); ++i) {{
        ap_int<{wa}> a; a.range({wa} - 1, 0) = (ap_uint<{wa}>)A[i];
        ap_int<{wb}> b; b.range({wb} - 1, 0) = (ap_uint<{wb}>)B[i];
        ap_int<{wc}> c; c.range({wc} - 1, 0) = (ap_uint<{wc}>)C[i];
        ap_int<{wy}> y = a * b + c;
        out << (unsigned long long)y.range({wy} - 1, 0) << "\\n";
    }}
    return 0;
}}
"""


def render_float_mac() -> str:
    """``float y = a*b + c`` via a split intermediate; built with -ffp-contract=off
    (see run.tcl) so it is two roundings — matching numpy float32, not a fused FMA."""
    return """#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

static std::vector<uint32_t> rd(const char* p) {
    std::ifstream f(p); std::vector<uint32_t> v; unsigned long long x;
    while (f >> x) v.push_back((uint32_t)x); return v;
}
static float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

int main(int argc, char** argv) {
    auto A = rd(argv[1]); auto B = rd(argv[2]); auto C = rd(argv[3]);
    std::ofstream out(argv[4]);
    for (size_t i = 0; i < A.size(); ++i) {
        float a = u2f(A[i]), b = u2f(B[i]), c = u2f(C[i]);
        float t = a * b;
        float y = t + c;
        out << (unsigned long long)f2u(y) << "\\n";
    }
    return 0;
}
"""


def render_fixed_mac(type_a: str, wa: int, type_b: str, wb: int,
                     type_c: str, wc: int, type_y: str, wy: int) -> str:
    """``target_t y = a*b + c`` — full-precision a*b+c, quantize-on-assign to target."""
    return """#include <ap_fixed.h>
#include <ap_int.h>
#include <fstream>
#include <vector>

static std::vector<unsigned long long> rd(const char* p) {
    std::ifstream f(p); std::vector<unsigned long long> v; unsigned long long x;
    while (f >> x) v.push_back(x); return v;
}
""" + f"""
int main(int argc, char** argv) {{
    auto A = rd(argv[1]); auto B = rd(argv[2]); auto C = rd(argv[3]);
    std::ofstream out(argv[4]);
    for (size_t i = 0; i < A.size(); ++i) {{
        {type_a} a; a.range({wa} - 1, 0) = (ap_uint<{wa}>)A[i];
        {type_b} b; b.range({wb} - 1, 0) = (ap_uint<{wb}>)B[i];
        {type_c} c; c.range({wc} - 1, 0) = (ap_uint<{wc}>)C[i];
        {type_y} y = a * b + c;
        out << (unsigned long long)y.range({wy} - 1, 0) << "\\n";
    }}
    return 0;
}}
"""
