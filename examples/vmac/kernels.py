"""Parameterized Vitis HLS m_axi kernel templates for the VMAC conformance harness.

A VMAC kernel implements the fused op ``D = α·A·op(B) + β·C [, reduce_rows]`` over a
**strided** region of shared memory reached through an ``m_axi`` pointer.  Each conformance
case bakes its (runtime) ``VmacCmd`` config — addresses / strides / shape / flags / mode /
shift / α,β — into a rendered kernel, while the **structural** widths set the ``ap_fixed``
types: the operand type ``ap_fixed<data_bw, int_bits>``, the wide accumulator type =
``VmacAccel.accumulator_format`` (full precision), and the output type =
``VmacAccel.output_format`` (right-shift ``SHIFT`` + round ``q_mode`` + saturate ``o_mode``
to ``out_bw``).  The kernel's ``acc += …`` over the full-precision expression then
``OUT_T y = acc;`` reproduces the Phase-2 contract bit-for-bit.

Memory image (a single flat array reached via the ``m_axi`` pointer):

- **real:** one stored ``int`` per slot (the operand's ``data_bw``-bit two's-complement bits);
- **complex:** two ints per slot — interleaved ``re`` then ``im``.

The kernel writes the dst output bits to ``out_bits.txt`` (row-major; interleaved ``re``/``im``
for complex), which the harness compares to ``VmacAccel.execute`` with zero mismatch.
"""
from __future__ import annotations

from dataclasses import dataclass

_PREAMBLE = """#include <ap_fixed.h>
#include <ap_int.h>
#include <complex>
#include <fstream>
#include <vector>

static std::vector<long long> read_mem(const char* path) {
    std::ifstream f(path);
    std::vector<long long> v;
    long long x;
    while (f >> x) v.push_back(x);
    return v;
}
"""


@dataclass
class KernelSpec:
    """The C++-ready description of one conformance case (baked into the rendered kernel)."""
    mode: str                       # "real" | "complex"
    data_bw: int
    int_bits: int
    a_t: str                        # operand ap_fixed type
    acc_t: str                      # wide accumulator ap_fixed type (full precision)
    out_t: str                      # output ap_fixed type (requantized)
    out_bw: int
    n_rows: int
    n_cols: int
    a: tuple                        # (addr, row_stride, col_stride)
    b: tuple
    c: tuple
    d: tuple                        # dst region (addr, row_stride, col_stride)
    b_one: bool
    c_zero: bool
    b_conj: bool
    reduce_rows: bool
    # alpha / beta: ("direct", re_bits, im_bits) or ("indirect", addr, stride)
    alpha: tuple
    beta: tuple
    # complex-only: the conj/op_b operand types per the golden's format growth
    opb_t: str = ""                 # ap_fixed type of op(B) component (conj grows by 1 bit)


# --- real mode ----------------------------------------------------------------
def _recon_real(t: str, w: int, name: str, idx_expr: str) -> str:
    return f"        {t} {name}; {name}.range({w} - 1, 0) = (ap_uint<{w}>)MEM[{idx_expr}];"


def _slot(reg: tuple, i: str, j: str) -> str:
    addr, rs, cs = reg
    return f"{addr} + ({i}) * {rs} + ({j}) * {cs}"


def _scalar_recon_real(spec: KernelSpec, which: str, j: str) -> str:
    """Reconstruct alpha/beta as an A_T value (direct immediate or indirect mem read)."""
    sc = spec.alpha if which == "alpha" else spec.beta
    if sc[0] == "direct":
        return f"        {spec.a_t} {which}; {which}.range({spec.data_bw} - 1, 0) = (ap_uint<{spec.data_bw}>){sc[1]}ULL;"
    addr, stride = sc[1], sc[2]
    return _recon_real(spec.a_t, spec.data_bw, which, f"{addr} + ({j}) * {stride}")


def _term_expr_real(spec: KernelSpec) -> str:
    ab = "alpha * a" if spec.b_one else "alpha * (a * b)"
    return ab if spec.c_zero else f"({ab}) + beta * c"


def _emit_real(out_bw: int, name: str) -> str:
    return f'        out << (unsigned long long){name}.range({out_bw} - 1, 0) << "\\n";'


def render_real(spec: KernelSpec) -> str:
    """A real-mode VMAC kernel: strided MAC into ``acc_t``, requantize to ``out_t``."""
    reads_a = _recon_real(spec.a_t, spec.data_bw, "a", _slot(spec.a, "i", "j"))
    reads = [reads_a]
    if not spec.b_one:
        reads.append(_recon_real(spec.a_t, spec.data_bw, "b", _slot(spec.b, "i", "j")))
    reads.append(_scalar_recon_real(spec, "alpha", "j"))
    if not spec.c_zero:
        reads.append(_recon_real(spec.a_t, spec.data_bw, "c", _slot(spec.c, "i", "j")))
        reads.append(_scalar_recon_real(spec, "beta", "j"))
    body = "\n".join(reads)
    term = _term_expr_real(spec)

    if spec.reduce_rows:
        loop = f"""    for (int j = 0; j < {spec.n_cols}; ++j) {{
        {spec.acc_t} acc = 0;
        for (int i = 0; i < {spec.n_rows}; ++i) {{
{body}
            acc += {term};
        }}
        {spec.out_t} y = acc;
{_emit_real(spec.out_bw, "y")}
    }}"""
    else:
        loop = f"""    for (int i = 0; i < {spec.n_rows}; ++i) {{
        for (int j = 0; j < {spec.n_cols}; ++j) {{
{body}
            {spec.out_t} y = {term};
{_emit_real(spec.out_bw, "y")}
        }}
    }}"""

    return _PREAMBLE + f"""
// VMAC kernel (real): D = alpha*A*op(B) + beta*C [, reduce_rows], m_axi `MEM`.
static void vmac_kernel(const std::vector<long long>& MEM, std::ofstream& out) {{
#pragma HLS INLINE off
{loop}
}}

int main(int argc, char** argv) {{
    auto MEM = read_mem(argv[1]);
    std::ofstream out(argv[3]);
    vmac_kernel(MEM, out);
    return 0;
}}
"""


# --- complex mode -------------------------------------------------------------
def _recon_cplx(t: str, w: int, name: str, slot_expr: str) -> str:
    """Reconstruct a complex operand's re/im components from interleaved mem slots."""
    return (f"        {t} {name}_re; {name}_re.range({w} - 1, 0) = (ap_uint<{w}>)MEM[2 * ({slot_expr})];\n"
            f"        {t} {name}_im; {name}_im.range({w} - 1, 0) = (ap_uint<{w}>)MEM[2 * ({slot_expr}) + 1];")


def _scalar_recon_cplx(spec: KernelSpec, which: str, j: str) -> str:
    sc = spec.alpha if which == "alpha" else spec.beta
    w = spec.data_bw
    if sc[0] == "direct":
        return (f"        {spec.a_t} {which}_re; {which}_re.range({w} - 1, 0) = (ap_uint<{w}>){sc[1]}ULL;\n"
                f"        {spec.a_t} {which}_im; {which}_im.range({w} - 1, 0) = (ap_uint<{w}>){sc[2]}ULL;")
    addr, stride = sc[1], sc[2]
    return _recon_cplx(spec.a_t, w, which, f"{addr} + ({j}) * {stride}")


def _emit_cplx(out_bw: int) -> str:
    return (f'        out << (unsigned long long)yr.range({out_bw} - 1, 0) << "\\n";\n'
            f'        out << (unsigned long long)yi.range({out_bw} - 1, 0) << "\\n";')


def render_complex(spec: KernelSpec) -> str:
    """A complex-mode VMAC kernel: the explicit component formula (cmult is
    ``(ar*br - ai*bi)`` / ``(ar*bi + ai*br)``, *not* ``std::complex`` operator*, which would
    quantize back) at full precision, then requantize re/im to ``out_t``."""
    reads = [_recon_cplx(spec.a_t, spec.data_bw, "a", _slot(spec.a, "i", "j"))]
    if not spec.b_one:
        reads.append(_recon_cplx(spec.a_t, spec.data_bw, "b", _slot(spec.b, "i", "j")))
    reads.append(_scalar_recon_cplx(spec, "alpha", "j"))
    if not spec.c_zero:
        reads.append(_recon_cplx(spec.a_t, spec.data_bw, "c", _slot(spec.c, "i", "j")))
        reads.append(_scalar_recon_cplx(spec, "beta", "j"))
    body = "\n".join(reads)

    if spec.b_one:
        ab = ["        auto ab_re = a_re;", "        auto ab_im = a_im;"]
    else:
        opb_im = "-b_im" if spec.b_conj else "b_im"   # conj(B): negate imag (no-op width: ap_fixed grows)
        ab = [
            f"        auto ab_re = a_re * b_re - a_im * ({opb_im});",
            f"        auto ab_im = a_re * ({opb_im}) + a_im * b_re;",
        ]
    term = [
        "        auto t_re = alpha_re * ab_re - alpha_im * ab_im;",
        "        auto t_im = alpha_re * ab_im + alpha_im * ab_re;",
    ]
    if not spec.c_zero:
        term += [
            "        auto bc_re = beta_re * c_re - beta_im * c_im;",
            "        auto bc_im = beta_re * c_im + beta_im * c_re;",
            "        auto tr = t_re + bc_re;",
            "        auto ti = t_im + bc_im;",
        ]
        tr, ti = "tr", "ti"
    else:
        tr, ti = "t_re", "t_im"
    compute = "\n".join(ab + term)

    if spec.reduce_rows:
        loop = f"""    for (int j = 0; j < {spec.n_cols}; ++j) {{
        {spec.acc_t} acc_re = 0, acc_im = 0;
        for (int i = 0; i < {spec.n_rows}; ++i) {{
{body}
{compute}
            acc_re += {tr};
            acc_im += {ti};
        }}
        {spec.out_t} yr = acc_re; {spec.out_t} yi = acc_im;
{_emit_cplx(spec.out_bw)}
    }}"""
    else:
        loop = f"""    for (int i = 0; i < {spec.n_rows}; ++i) {{
        for (int j = 0; j < {spec.n_cols}; ++j) {{
{body}
{compute}
            {spec.out_t} yr = {tr}; {spec.out_t} yi = {ti};
{_emit_cplx(spec.out_bw)}
        }}
    }}"""

    return _PREAMBLE + f"""
// VMAC kernel (complex): D = alpha*A*op(B) + beta*C [, reduce_rows], explicit re/im formula.
static void vmac_kernel(const std::vector<long long>& MEM, std::ofstream& out) {{
#pragma HLS INLINE off
{loop}
}}

int main(int argc, char** argv) {{
    auto MEM = read_mem(argv[1]);
    std::ofstream out(argv[3]);
    vmac_kernel(MEM, out);
    return 0;
}}
"""


def render(spec: KernelSpec) -> str:
    """Render the kernel for ``spec.mode`` (``real`` | ``complex``)."""
    return render_complex(spec) if spec.mode == "complex" else render_real(spec)

