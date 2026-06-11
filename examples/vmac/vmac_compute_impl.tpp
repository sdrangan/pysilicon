// vmac_compute_impl.tpp
// Included from gen/vmac.hpp.  The generated header brings into scope, *before* this file:
//
//   * VmacCmd                  — the command struct (DataSchemaStep).
//   * vmac_in_au / vmac_out_au — namespace aliases for the **generated ComplexField
//                                serialization** of the operand / output element types
//                                (ArrayUtilsStep over ComplexField.specialize(FixedField…)).
//     `vmac_in_au::value_type`  == std::complex<ap_fixed<DATA_BW, INT_BITS, …>>  (operand)
//     `vmac_out_au::value_type` == std::complex<ap_fixed<OUT_BW,  OUT_INT,  …>>  (writeback)
//   * complex_utils::{cmult,cadd,conj} — full-precision complex arithmetic (the explicit
//                                re/im formula, widths following the native ap_fixed growth =
//                                the Python *_format rules; cmult/cadd accept mixed operand
//                                formats, so the F -> 2F -> 3F chain composes with no widen).
//
// Hand-written body for the templated `vmac_impl::vmac_compute` hook — the single VMAC
// datapath: the **complex** fused op
//
//     D = alpha * A * op(B) + beta * C   [, reduced over rows]
//
// over a row-major shared-memory image reached through the `m_axi` pointer.  This is the C++
// contract of `VmacAccel.vmac_compute` (whose Python body is the golden `VmacAccel.execute`);
// it is bit-identical to that golden by construction.
//
// The datapath stays **complex-typed end to end** — `std::complex<ap_fixed>` lanes, a complex
// scalar, `complex_utils` ops, a complex accumulator, and an `ap_fixed`-assignment requantize —
// with no hand-split re/im and no integer-code bridging.  Full precision is carried in the
// `ap_fixed` value (no rounding until the single requantize); the wide accumulator holds the
// product at fractional depth 3*F_in (>= the per-flag 2F/3F the golden tracks), so requantizing
// its *value* to F_out = F_in reproduces the golden's quantize bit-for-bit (the intermediate
// fractional depth is irrelevant once the value is exact).
//
// ONE fixed-format kernel, configured at *run time* by the command's op flags:
//   * Template params are **widths only** -> all ap_fixed types are compile-time (no dynamic types).
//   * The op flags (b_one / c_zero / b_conj / reduce_rows) are **runtime** if/mux — loop-invariant.
//
// VMAC is complex-only: every element is one packed ComplexField (interleaved re/im), so the
// element bitwidth is 2*DATA_BW and the kernel processes PF = MEM_BW / (2*DATA_BW) complex
// columns per memory word.  Lane I/O reuses the generated serialization
// (`read_array_elem<ComplexField>` / `write_array_elem<ComplexField>`); per-row operand regions
// are word-aligned (addr / row_stride multiples of PF), so the PF contiguous columns of a row
// live in one word and are reached by a running word pointer advanced a constant per step.

#include <cassert>

#include "complex_utils.hpp"

namespace vmac_impl {

// Map an *element* index/stride to its *word* index for PF-packed memory (PF elements/word):
// the divide is a shift (PF is a compile-time power of two), with a PF-alignment check — the
// per-row operand regions (bases + strides) must be word-aligned.  (The per-column indirect
// scalar is genuinely element-addressed — word = e/PF, lane = e&(PF-1) — so it is read inline,
// not through this aligned helper.)
template <int PF, typename T>
static inline T elem_to_word(T elem) {
#pragma HLS INLINE
    assert(elem % PF == 0 && "VMAC operand region must be PF-aligned (word-packed rows)");
    return elem / PF;
}

// The datapath, taking the command as **typed scalars** (not the VmacCmd struct).  The
// synthesizable top calls this directly: a nested struct passed by value mis-decomposes
// through HLS's Array/Struct optimization at csynth (loop bounds fold to 0 -> the kernel is
// DCE'd), whereas scalar s_axilite args lower cleanly.  The fields keep their precise types
// (addresses ap_uint<MEM_AWIDTH>, strides signed, shape ap_uint<16>, scalar codes ap_int) so
// the address arithmetic is sized by MEM_AWIDTH, not a stray 32-bit int.  vmac_compute(cmd, mem)
// below is the thin struct-taking wrapper the csim conformance harness drives.
template <int MEM_BW, int MEM_AWIDTH, int DATA_BW, int INT_BITS, int ACC_BW, int OUT_BW,
          bool Q_RND, bool O_SAT, int MAX_COLS>
void vmac_compute_core(
    ap_uint<MEM_BW>* mem,
    ap_uint<16> n_rows, ap_uint<16> n_cols, bool b_one, bool c_zero, bool b_conj, bool reduce_rows,
    ap_uint<MEM_AWIDTH> a_addr, ap_int<MEM_AWIDTH> a_rs,
    ap_uint<MEM_AWIDTH> b_addr, ap_int<MEM_AWIDTH> b_rs,
    ap_uint<MEM_AWIDTH> c_addr, ap_int<MEM_AWIDTH> c_rs,
    ap_uint<MEM_AWIDTH> d_addr, ap_int<MEM_AWIDTH> d_rs,
    bool al_direct, ap_int<DATA_BW> al_re, ap_int<DATA_BW> al_im,
    ap_uint<MEM_AWIDTH> al_addr, ap_int<MEM_AWIDTH> al_stride,
    bool be_direct, ap_int<DATA_BW> be_re, ap_int<DATA_BW> be_im,
    ap_uint<MEM_AWIDTH> be_addr, ap_int<MEM_AWIDTH> be_stride) {
#pragma HLS INLINE
    // Inline into the synthesizable top so the m_axi reads/writes belong to the top's gmem
    // port (kept as a separate module, the top would have "no outputs" and gmem would dangle).
    typedef typename vmac_in_au::value_type CX;     // std::complex<ap_fixed<DATA_BW, INT_BITS, …>>
    typedef typename CX::value_type IN_FX;          // ap_fixed<DATA_BW, INT_BITS, …>
    typedef typename vmac_out_au::value_type CXO;   // std::complex<ap_fixed<OUT_BW, OUT_INT, …>>
    typedef typename CXO::value_type OUT_FX;
    typedef ap_fixed<DATA_BW + 1, INT_BITS + 1> OPB_FX;   // op(B) = conj-format (W+1, I+1)
    typedef std::complex<OPB_FX> OPB_CX;
    static const int F_IN = DATA_BW - INT_BITS;
    static const int OUT_INT = OUT_BW - F_IN;             // F_out = F_in (structural)
    static const ap_q_mode QMODE = Q_RND ? AP_RND : AP_TRN;
    static const ap_o_mode OMODE = O_SAT ? AP_SAT : AP_WRAP;
    typedef ap_fixed<OUT_BW, OUT_INT, QMODE, OMODE> REQ_FX;     // requantize target (structural Q/O)
    typedef ap_fixed<ACC_BW, ACC_BW - 3 * F_IN> ACC_FX;        // accumulator component, frac = 3*F_in
    typedef std::complex<ACC_FX> ACC_CX;
    static constexpr int PF = vmac_in_au::pf<MEM_BW>();       // complex columns / word (power of 2)

    // loop-invariant immediate scalars (one complex value each, via the (re, im) constructor)
    const CX alpha_imm = complex_utils::cx_from_codes<CX>(al_re, al_im);
    const CX beta_imm = complex_utils::cx_from_codes<CX>(be_re, be_im);

    // running word indices: row base = addr/PF, advanced by row_stride/PF per row (both exact —
    // regions are PF-aligned; elem_to_word checks it).
    const ap_uint<MEM_AWIDTH> a_w0 = elem_to_word<PF>(a_addr), b_w0 = elem_to_word<PF>(b_addr);
    const ap_uint<MEM_AWIDTH> c_w0 = elem_to_word<PF>(c_addr), d_w0 = elem_to_word<PF>(d_addr);
    const ap_int<MEM_AWIDTH> a_rsw = elem_to_word<PF>(a_rs), b_rsw = elem_to_word<PF>(b_rs);
    const ap_int<MEM_AWIDTH> c_rsw = elem_to_word<PF>(c_rs), d_rsw = elem_to_word<PF>(d_rs);

    // per-column complex accumulators (reduce_rows): summed over rows.
    ACC_CX acc[MAX_COLS];
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=16 dim=1
    if (reduce_rows) {
        for (int j = 0; j < (int)n_cols; ++j) {
#pragma HLS PIPELINE II=1
            acc[j] = ACC_CX(0, 0);
        }
    }

    // outer loop over rows (running pointers per operand), inner over contiguous columns packed
    // PF/word (the GEMM accumulation pattern).
    ap_uint<MEM_AWIDTH> a_row = a_w0, b_row = b_w0, c_row = c_w0, d_row = d_w0;
    for (int i = 0; i < (int)n_rows; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_COLS
        ap_uint<MEM_AWIDTH> a_w = a_row, b_w = b_row, c_w = c_row, d_w = d_row;
        for (int col0 = 0; col0 < (int)n_cols; col0 += PF) {
#pragma HLS PIPELINE II=1
            const int cols = ((int)n_cols - col0 < PF) ? ((int)n_cols - col0) : PF;
            CX a_lane[PF], b_lane[PF], c_lane[PF];
#pragma HLS ARRAY_PARTITION variable=a_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=b_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=c_lane complete dim=1
            vmac_in_au::read_array_elem<MEM_BW>(mem + a_w, a_lane, cols);
            if (!b_one)
                vmac_in_au::read_array_elem<MEM_BW>(mem + b_w, b_lane, cols);
            if (!c_zero)
                vmac_in_au::read_array_elem<MEM_BW>(mem + c_w, c_lane, cols);

            CXO y_lane[PF];
#pragma HLS ARRAY_PARTITION variable=y_lane complete dim=1
            for (int k = 0; k < PF; ++k) {
#pragma HLS UNROLL
                const int j = col0 + k;
                if (j >= (int)n_cols) continue;

                // per-column alpha/beta: immediate, or one complex element read from the word
                // containing it (lane = e & (PF-1), no modulo; PF is a power of two).
                CX alpha = alpha_imm, beta = beta_imm;
                if (!al_direct) {
                    const ap_uint<MEM_AWIDTH> e = al_addr + (ap_uint<MEM_AWIDTH>)(j * (int)al_stride);
                    CX sb[PF];
                    vmac_in_au::read_array_elem<MEM_BW>(mem + e / PF, sb, PF);
                    alpha = sb[e & (PF - 1)];
                }
                if (!be_direct && !c_zero) {
                    const ap_uint<MEM_AWIDTH> e = be_addr + (ap_uint<MEM_AWIDTH>)(j * (int)be_stride);
                    CX sb[PF];
                    vmac_in_au::read_array_elem<MEM_BW>(mem + e / PF, sb, PF);
                    beta = sb[e & (PF - 1)];
                }

                // alpha * A * op(B):  op(B) = conj(B) or B; the F -> 2F -> 3F chain via cmult.
                ACC_CX term;
                if (b_one) {
                    term = complex_utils::cwiden<ACC_CX>(complex_utils::cmult(alpha, a_lane[k]));
                } else {
                    OPB_CX opb = b_conj
                        ? complex_utils::conj(b_lane[k])
                        : OPB_CX((OPB_FX)b_lane[k].real(), (OPB_FX)b_lane[k].imag());
                    term = complex_utils::cwiden<ACC_CX>(
                        complex_utils::cmult(alpha, complex_utils::cmult(a_lane[k], opb)));
                }
                // + beta * C  (beta*C widened to the accumulator scale, then a complex add)
                if (!c_zero) {
                    ACC_CX bc = complex_utils::cwiden<ACC_CX>(complex_utils::cmult(beta, c_lane[k]));
                    term = complex_utils::cwiden<ACC_CX>(complex_utils::cadd(term, bc));
                }

                if (reduce_rows)
                    acc[j] = complex_utils::cwiden<ACC_CX>(complex_utils::cadd(acc[j], term));
                else
                    y_lane[k] = complex_utils::cx_requantize<CXO, REQ_FX>(term);
            }
            if (!reduce_rows)
                vmac_out_au::write_array_elem<MEM_BW>(y_lane, mem + d_w, cols);

            a_w += 1; b_w += 1; c_w += 1; d_w += 1;
        }
        a_row += a_rsw; b_row += b_rsw; c_row += c_rsw; d_row += d_rsw;
    }

    // reduce_rows writeback: one row of n_cols requantized results at the dst.
    if (reduce_rows) {
        ap_uint<MEM_AWIDTH> d_w = d_w0;
        for (int col0 = 0; col0 < (int)n_cols; col0 += PF) {
#pragma HLS PIPELINE II=1
            const int cols = ((int)n_cols - col0 < PF) ? ((int)n_cols - col0) : PF;
            CXO y_lane[PF];
#pragma HLS ARRAY_PARTITION variable=y_lane complete dim=1
            for (int k = 0; k < PF; ++k) {
#pragma HLS UNROLL
                const int j = col0 + k;
                if (j >= (int)n_cols) continue;
                y_lane[k] = complex_utils::cx_requantize<CXO, REQ_FX>(acc[j]);
            }
            vmac_out_au::write_array_elem<MEM_BW>(y_lane, mem + d_w, cols);
            d_w += 1;
        }
    }
}

// Struct-taking wrapper — extracts the command's (typed) scalar fields and calls the core.
// Used by the csim conformance harness (where the by-value struct is fine); the synthesizable
// top calls vmac_compute_core directly to avoid the nested-struct csynth pitfall above.
template <int MEM_BW, int MEM_AWIDTH, int DATA_BW, int INT_BITS, int ACC_BW, int OUT_BW,
          bool Q_RND, bool O_SAT, int MAX_COLS>
void vmac_compute(VmacCmd cmd, ap_uint<MEM_BW>* mem) {
    vmac_compute_core<MEM_BW, MEM_AWIDTH, DATA_BW, INT_BITS, ACC_BW, OUT_BW, Q_RND, O_SAT, MAX_COLS>(
        mem,
        (ap_uint<16>)cmd.n_rows, (ap_uint<16>)cmd.n_cols,
        (bool)cmd.b_one, (bool)cmd.c_zero, (bool)cmd.b_conj, (bool)cmd.reduce_rows,
        (ap_uint<MEM_AWIDTH>)cmd.a.addr, (ap_int<MEM_AWIDTH>)cmd.a.row_stride,
        (ap_uint<MEM_AWIDTH>)cmd.b.addr, (ap_int<MEM_AWIDTH>)cmd.b.row_stride,
        (ap_uint<MEM_AWIDTH>)cmd.c.addr, (ap_int<MEM_AWIDTH>)cmd.c.row_stride,
        (ap_uint<MEM_AWIDTH>)cmd.d.addr, (ap_int<MEM_AWIDTH>)cmd.d.row_stride,
        (bool)cmd.alpha.direct, (ap_int<DATA_BW>)cmd.alpha.re, (ap_int<DATA_BW>)cmd.alpha.im,
        (ap_uint<MEM_AWIDTH>)cmd.alpha.addr, (ap_int<MEM_AWIDTH>)cmd.alpha.stride,
        (bool)cmd.beta.direct, (ap_int<DATA_BW>)cmd.beta.re, (ap_int<DATA_BW>)cmd.beta.im,
        (ap_uint<MEM_AWIDTH>)cmd.beta.addr, (ap_int<MEM_AWIDTH>)cmd.beta.stride);
}

}  // namespace vmac_impl
