// vmac_compute_impl.tpp
// Included from gen/vmac.hpp.  The generated header brings into scope, *before* this file:
//
//   * VmacCmd                  — the command struct (DataSchemaStep).
//   * vmac_in_au / vmac_out_au — namespace aliases for the **generated ComplexField
//                                serialization** of the operand / output element types
//                                (ArrayUtilsStep over ComplexField.specialize(FixedField…)).
//     `vmac_in_au::value_type`  == std::complex<ap_fixed<DATA_BW, INT_BITS, …>>  (operand)
//     `vmac_out_au::value_type` == std::complex<ap_fixed<OUT_BW,  OUT_INT,  …>>  (writeback)
//
// Hand-written body for the templated `vmac_impl::vmac_compute` hook — the single VMAC
// datapath: the **complex** fused op
//
//     D = alpha * A * op(B) + beta * C   [, reduced over rows]
//
// over a row-major shared-memory image reached through the `m_axi` pointer.  This is the C++
// contract of `VmacAccel.vmac_compute` (whose Python body is the golden `VmacAccel.execute`);
// it is bit-identical to that golden by construction (full-precision integer intermediates in
// the wide accumulator, a single lossy requantize).
//
// ONE fixed-format kernel, configured at *run time* by the command's op flags:
//
//   * Template params are **widths only** (MEM_BW / DATA_BW / INT_BITS / ACC_BW / OUT_BW /
//     Q_RND / O_SAT) -> the ap_int/ap_fixed types are compile-time (no dynamic types).
//   * The op flags (b_one / c_zero / b_conj / reduce_rows) are **runtime** if/mux —
//     loop-invariant, so no II hit.
//   * The requantize shift is **derived** from the flags + the structural format
//     (F_acc = (b_one?2:3)*F_in, F_out = F_in) -> SHIFT = F_acc - F_out: a variable barrel
//     shift on the fixed-width accumulator (vmac_requantize), not a dynamic type.
//
// VMAC is complex-only: every element is one packed ComplexField (interleaved re/im), so the
// element bitwidth is 2*DATA_BW and the kernel processes PF = MEM_BW / (2*DATA_BW) complex
// columns per memory word.  The complex multiply uses the **explicit re/im formula**
// (ar*br - ai*bi, ar*bi + ai*br) over the operands' integer codes (full precision, no
// quantization until the final requantize); op(B) = conj(B) negates B's imag.
//
// Lane I/O reuses the **generated** ComplexField serialization (NOT a hand-rolled component
// interleave): `vmac_in_au::read_array_elem<MEM_BW>` fills a lane buffer of complex elements
// from one memory word, and `vmac_out_au::write_array_elem<MEM_BW>` packs the requantized
// results back, exactly as examples/schemas/complex's migrated kernel does.  Per-row operand
// regions are word-aligned (addr / row_stride are multiples of PF), so the PF contiguous
// columns of a row live in the single word at element index / PF.

#include "vmac_utils.h"

namespace vmac_impl {

template <int MEM_BW, int DATA_BW, int INT_BITS, int ACC_BW, int OUT_BW,
          bool Q_RND, bool O_SAT, int MAX_COLS>
void vmac_compute(VmacCmd cmd, ap_uint<MEM_BW>* mem) {
    typedef ap_int<DATA_BW> A_T;              // operand (re/im) integer code
    typedef ap_int<ACC_BW> ACC_T;             // full-precision integer accumulator
    typedef typename vmac_in_au::value_type CX;    // std::complex<ap_fixed<DATA_BW,…>>
    typedef typename vmac_out_au::value_type CXO;  // std::complex<ap_fixed<OUT_BW,…>>
    typedef typename CXO::value_type OUT_FX;        // inner ap_fixed<OUT_BW, OUT_INT, …>
    static const int F_IN = DATA_BW - INT_BITS;
    static const int PF = MEM_BW / (2 * DATA_BW);  // complex columns / word

    const int n_rows = (int)cmd.n_rows;
    const int n_cols = (int)cmd.n_cols;
    const bool b_one = (bool)cmd.b_one;
    const bool c_zero = (bool)cmd.c_zero;
    const bool b_conj = (bool)cmd.b_conj;
    const bool reduce_rows = (bool)cmd.reduce_rows;

    // Derived requantize shift: F_acc = (b_one?2:3)*F_in, F_out = F_in -> SHIFT = F_acc - F_in.
    const int shift = (b_one ? 1 : 2) * F_IN;
    // beta*C lands at scale 2*F_in; the alpha*A*op(B) term sits at scale (b_one?2:3)*F_in, so
    // when !b_one the addend must be aligned up by F_in to the accumulator's fractional depth.
    const int c_align = b_one ? 0 : F_IN;

    const int a_addr = (int)cmd.a.addr, a_rs = (int)cmd.a.row_stride;
    const int b_addr = (int)cmd.b.addr, b_rs = (int)cmd.b.row_stride;
    const int c_addr = (int)cmd.c.addr, c_rs = (int)cmd.c.row_stride;
    const int d_addr = (int)cmd.d.addr, d_rs = (int)cmd.d.row_stride;

    // alpha / beta immediates (direct); per-column (indirect) reads are loaded per column.
    const bool al_direct = (bool)cmd.alpha.direct, be_direct = (bool)cmd.beta.direct;
    const A_T al_re_imm = (A_T)(ap_int<DATA_BW>)cmd.alpha.re;
    const A_T al_im_imm = (A_T)(ap_int<DATA_BW>)cmd.alpha.im;
    const A_T be_re_imm = (A_T)(ap_int<DATA_BW>)cmd.beta.re;
    const A_T be_im_imm = (A_T)(ap_int<DATA_BW>)cmd.beta.im;

    // per-column accumulators (reduce_rows): one complex ACC_T per column, summed over rows.
    ACC_T acc_re[MAX_COLS], acc_im[MAX_COLS];
#pragma HLS ARRAY_PARTITION variable=acc_re cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=acc_im cyclic factor=16 dim=1
    if (reduce_rows) {
        for (int j = 0; j < n_cols; ++j) {
#pragma HLS PIPELINE II=1
            acc_re[j] = 0;
            acc_im[j] = 0;
        }
    }

    // outer loop over rows (strided by the pitch), inner over contiguous columns packed
    // PF/word (the GEMM accumulation pattern).
    for (int i = 0; i < n_rows; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_COLS
        for (int col0 = 0; col0 < n_cols; col0 += PF) {
#pragma HLS PIPELINE II=1
            const int cols = (n_cols - col0 < PF) ? (n_cols - col0) : PF;
            CX a_lane[PF], b_lane[PF], c_lane[PF];
#pragma HLS ARRAY_PARTITION variable=a_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=b_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=c_lane complete dim=1
            vmac_in_au::template read_array_elem<MEM_BW>(
                mem + (a_addr + i * a_rs + col0) / PF, a_lane, cols);
            if (!b_one)
                vmac_in_au::template read_array_elem<MEM_BW>(
                    mem + (b_addr + i * b_rs + col0) / PF, b_lane, cols);
            if (!c_zero)
                vmac_in_au::template read_array_elem<MEM_BW>(
                    mem + (c_addr + i * c_rs + col0) / PF, c_lane, cols);

            CXO y_lane[PF];
#pragma HLS ARRAY_PARTITION variable=y_lane complete dim=1
            for (int k = 0; k < PF; ++k) {
#pragma HLS UNROLL
                const int j = col0 + k;
                if (j >= n_cols) continue;

                const A_T are = a_lane[k].real().V, aim = a_lane[k].imag().V;

                // per-column alpha/beta (indirect): one complex element per column, read from
                // the word containing it (arbitrary alignment -> pick the lane).
                A_T alre = al_re_imm, alim = al_im_imm, bere = be_re_imm, beim = be_im_imm;
                if (!al_direct) {
                    const int e = (int)cmd.alpha.addr + j * (int)cmd.alpha.stride;
                    CX sb[PF];
                    vmac_in_au::template read_array_elem<MEM_BW>(mem + e / PF, sb, PF);
                    alre = sb[e % PF].real().V; alim = sb[e % PF].imag().V;
                }
                if (!be_direct && !c_zero) {
                    const int e = (int)cmd.beta.addr + j * (int)cmd.beta.stride;
                    CX sb[PF];
                    vmac_in_au::template read_array_elem<MEM_BW>(mem + e / PF, sb, PF);
                    bere = sb[e % PF].real().V; beim = sb[e % PF].imag().V;
                }

                // A * op(B)  (explicit re/im, full precision).  op(B): identity or conj.
                ACC_T abre, abim;
                if (b_one) {
                    abre = (ACC_T)are; abim = (ACC_T)aim;
                } else {
                    const A_T bre = b_lane[k].real().V, bim = b_lane[k].imag().V;
                    const ACC_T obre = bre;
                    const ACC_T obim = b_conj ? (ACC_T)(-bim) : (ACC_T)bim;
                    abre = (ACC_T)are * obre - (ACC_T)aim * obim;
                    abim = (ACC_T)are * obim + (ACC_T)aim * obre;
                }
                // alpha * (A*op(B))
                ACC_T tre = (ACC_T)alre * abre - (ACC_T)alim * abim;
                ACC_T tim = (ACC_T)alre * abim + (ACC_T)alim * abre;
                // + beta * C  (aligned up to the accumulator fractional depth when !b_one)
                if (!c_zero) {
                    const A_T cre = c_lane[k].real().V, cim = c_lane[k].imag().V;
                    ACC_T bcre = (ACC_T)bere * (ACC_T)cre - (ACC_T)beim * (ACC_T)cim;
                    ACC_T bcim = (ACC_T)bere * (ACC_T)cim + (ACC_T)beim * (ACC_T)cre;
                    tre += bcre << c_align;
                    tim += bcim << c_align;
                }

                if (reduce_rows) {
                    acc_re[j] += tre;
                    acc_im[j] += tim;
                } else {
                    OUT_FX yr, yi;
                    yr.V = vmac_requantize<OUT_BW, Q_RND, O_SAT>(tre, shift);
                    yi.V = vmac_requantize<OUT_BW, Q_RND, O_SAT>(tim, shift);
                    y_lane[k] = CXO(yr, yi);
                }
            }
            if (!reduce_rows)
                vmac_out_au::template write_array_elem<MEM_BW>(
                    y_lane, mem + (d_addr + i * d_rs + col0) / PF, cols);
        }
    }

    // reduce_rows writeback: one row of n_cols requantized results at the dst.
    if (reduce_rows) {
        for (int col0 = 0; col0 < n_cols; col0 += PF) {
#pragma HLS PIPELINE II=1
            const int cols = (n_cols - col0 < PF) ? (n_cols - col0) : PF;
            CXO y_lane[PF];
#pragma HLS ARRAY_PARTITION variable=y_lane complete dim=1
            for (int k = 0; k < PF; ++k) {
#pragma HLS UNROLL
                const int j = col0 + k;
                if (j >= n_cols) continue;
                OUT_FX yr, yi;
                yr.V = vmac_requantize<OUT_BW, Q_RND, O_SAT>(acc_re[j], shift);
                yi.V = vmac_requantize<OUT_BW, Q_RND, O_SAT>(acc_im[j], shift);
                y_lane[k] = CXO(yr, yi);
            }
            vmac_out_au::template write_array_elem<MEM_BW>(
                y_lane, mem + (d_addr + col0) / PF, cols);
        }
    }
}

}  // namespace vmac_impl
