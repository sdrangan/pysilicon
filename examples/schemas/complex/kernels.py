"""Generated complex C++ kernels for the ComplexField conformance harness.

Per inner type, a kernel reads interleaved-I/Q operand vectors (``re`` then ``im`` per
element, one stored-int / double per line), reconstructs the complex value bit-for-bit,
performs a v1 op, and writes the result stored bits (again interleaved ``re`` then
``im``).  The Python ``DataArray[ComplexField]`` ops must produce identical bits.

C++ mapping per inner (Phase-0 reference):

- **float**  -> ``std::complex<float>`` / ``std::complex<double>``; arithmetic uses
  ``std::complex`` ``operator*``/``+`` directly -- the *float-complex-multiply edge* we
  verify empirically.  Components (de)serialize as raw IEEE bits (``memcpy``).
- **fixed**  -> ``std::complex<ap_fixed<W,I,Q,O>>``; arithmetic uses the **explicit
  component formula** at the grown result format (``std::complex::operator*`` would
  quantize the product back to the inner format -- see the reference), so the
  full-precision growth matches the composed Python model bit-for-bit.
- **int**    -> a Waveflow-emitted ``wf_cint`` struct (``std::complex<ap_int>`` is
  non-standard); explicit component formula.

All kernels share one ``argv`` signature ``(in_a, in_b, out_bits)`` so a single
``run.tcl`` drives them (round-trip / conj ignore ``in_b``).
"""
from __future__ import annotations

_PREAMBLE = """#include <ap_fixed.h>
#include <ap_int.h>
#include <complex>
#include <cstring>
#include <fstream>
#include <vector>

static std::vector<unsigned long long> read_bits(const char* path) {
    std::ifstream f(path);
    std::vector<unsigned long long> v;
    unsigned long long x;
    while (f >> x) v.push_back(x);
    return v;
}
"""

# A Waveflow complex-int struct: std::complex<ap_int> is non-standard, so int-complex
# is Waveflow's own struct (two ap_int<W>).
_CINT_STRUCT = """
template <int W> struct wf_cint { ap_int<W> re; ap_int<W> im; };
"""


# --- per-kind component reconstruct (from a stored-int / double) ---------------
def _recon(kind: str, ctype: str, W: int, name: str, src: str) -> str:
    """C++ declaring ``name`` of the inner type, set from operand expression ``src``."""
    if kind == "float":
        ute, ity = ("unsigned int", 4) if W == 32 else ("unsigned long long", 8)
        return (f"        {ctype} {name}; {{ {ute} _t = ({ute}){src}; "
                f"std::memcpy(&{name}, &_t, {ity}); }}")
    if kind == "int":
        return f"        ap_int<{W}> {name}; {name}.range({W} - 1, 0) = (ap_uint<{W}>){src};"
    return f"        {ctype} {name}; {name}.range({W} - 1, 0) = (ap_uint<{W}>){src};"


# --- per-kind component emit (stored bits of a typed var) ----------------------
def _emit(kind: str, W: int, name: str) -> str:
    if kind == "float":
        ute, ity = ("unsigned int", 4) if W == 32 else ("unsigned long long", 8)
        return (f"        {{ {ute} _t; std::memcpy(&_t, &{name}, {ity}); "
                f'out << (unsigned long long)_t << "\\n"; }}')
    if kind == "int":
        return f'        out << (unsigned long long)(ap_uint<{W}>){name}.range({W} - 1, 0) << "\\n";'
    return f'        out << (unsigned long long){name}.range({W} - 1, 0) << "\\n";'


def _main(body_decls: str, loop_body: str, reads_b: bool = True) -> str:
    reads = "    auto A = read_bits(argv[1]);\n"
    if reads_b:
        reads += "    auto B = read_bits(argv[2]);\n"
    return _PREAMBLE + _CINT_STRUCT + f"""
int main(int argc, char** argv) {{
{reads}    std::ofstream out(argv[3]);
{body_decls}    for (size_t i = 0; i < A.size(); i += 2) {{
{loop_body}    }}
    return 0;
}}
"""


# --- round-trip (load reals re, im into the complex type) ----------------------
def render_load_real(kind: str, ctype: str, W: int) -> str:
    """``y = complex(re, im)`` from interleaved doubles -> emit the stored bits.

    fixed/float quantize on assignment into the inner type; int reads stored ints."""
    if kind == "int":
        loop = "\n".join([
            _recon("int", ctype, W, "yr", "A[i]"),
            _recon("int", ctype, W, "yi", "A[i + 1]"),
            _emit("int", W, "yr"), _emit("int", W, "yi"),
        ])
        return _main("", loop + "\n", reads_b=False)
    # float / fixed: read doubles, quantize on assignment
    decl = "    double re, im;\n"
    loop = "\n".join([
        f"        {ctype} yr = re; {ctype} yi = im;",
        _emit(kind, W, "yr"), _emit(kind, W, "yi"),
    ])
    body = (_PREAMBLE + f"""
int main(int argc, char** argv) {{
    std::ifstream fin(argv[1]);
    std::ofstream out(argv[3]);
{decl}    while (fin >> re >> im) {{
{loop}
    }}
    return 0;
}}
""")
    return body


# --- cmult: (ar*br - ai*bi) + j(ar*bi + ai*br) --------------------------------
def render_cmult(kind: str, ctype: str, W: int, rtype: str, Wr: int) -> str:
    """Complex multiply; fixed/int use the explicit component formula at the grown
    format ``rtype`` (``Wr`` bits); float uses ``std::complex`` ``operator*`` (the edge)."""
    recon = "\n".join([
        _recon(kind, ctype, W, "ar", "A[i]"), _recon(kind, ctype, W, "ai", "A[i + 1]"),
        _recon(kind, ctype, W, "br", "B[i]"), _recon(kind, ctype, W, "bi", "B[i + 1]"),
    ])
    if kind == "float":
        loop = "\n".join([
            recon,
            f"        std::complex<{ctype}> a(ar, ai), b(br, bi);",
            f"        std::complex<{ctype}> y = a * b;",
            f"        {ctype} yr = y.real(); {ctype} yi = y.imag();",
            _emit(kind, Wr, "yr"), _emit(kind, Wr, "yi"),
        ])
    else:
        loop = "\n".join([
            recon,
            f"        {rtype} yr = ar * br - ai * bi;",
            f"        {rtype} yi = ar * bi + ai * br;",
            _emit(kind, Wr, "yr"), _emit(kind, Wr, "yi"),
        ])
    return _main("", loop + "\n")


# --- cadd / csub: componentwise -----------------------------------------------
def render_caddsub(op: str, kind: str, ctype: str, W: int, rtype: str, Wr: int) -> str:
    """``a +/- b`` componentwise; result components at the grown format ``rtype``."""
    recon = "\n".join([
        _recon(kind, ctype, W, "ar", "A[i]"), _recon(kind, ctype, W, "ai", "A[i + 1]"),
        _recon(kind, ctype, W, "br", "B[i]"), _recon(kind, ctype, W, "bi", "B[i + 1]"),
    ])
    if kind == "float":
        loop = "\n".join([
            recon,
            f"        std::complex<{ctype}> a(ar, ai), b(br, bi);",
            f"        std::complex<{ctype}> y = a {op} b;",
            f"        {ctype} yr = y.real(); {ctype} yi = y.imag();",
            _emit(kind, Wr, "yr"), _emit(kind, Wr, "yi"),
        ])
    else:
        loop = "\n".join([
            recon,
            f"        {rtype} yr = ar {op} br;",
            f"        {rtype} yi = ai {op} bi;",
            _emit(kind, Wr, "yr"), _emit(kind, Wr, "yi"),
        ])
    return _main("", loop + "\n")


# --- conj: ar - j(ai) ---------------------------------------------------------
def render_conj(kind: str, ctype: str, W: int, rtype: str, Wr: int) -> str:
    """Conjugate; imag negated, result components at the grown format ``rtype``."""
    recon = "\n".join([
        _recon(kind, ctype, W, "ar", "A[i]"), _recon(kind, ctype, W, "ai", "A[i + 1]"),
    ])
    if kind == "float":
        loop = "\n".join([
            recon,
            f"        {ctype} yr = ar; {ctype} yi = -ai;",
            _emit(kind, Wr, "yr"), _emit(kind, Wr, "yi"),
        ])
    else:
        loop = "\n".join([
            recon,
            f"        {rtype} yr = ar; {rtype} yi = -ai;",
            _emit(kind, Wr, "yr"), _emit(kind, Wr, "yi"),
        ])
    return _main("", loop + "\n", reads_b=False)
