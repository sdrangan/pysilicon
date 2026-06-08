"""``VmacAccel`` — the VMAC accelerator: an ``HwComponent`` + the bit-exact Python golden.

A VMAC instance is parameterized by its **structural** widths — fixed at synthesis, the
HLS template params — declared as ``HwParam[int]`` fields: ``mem_dwidth`` (MEM_BW),
``mem_awidth``, ``data_bw`` (IN_BW), ``acc_bw``, ``out_bw``.  The instance's values drive
the command schema through the computed :pyattr:`Cmd` property — the **instance → type
bridge** that unifies the component's ``HwParam`` with the schema's symbolic ``Param``
(one parameter concept, two binding sites)::

    accel = VmacAccel(data_bw=16)             # HwParam values bind at instantiation
    cmd   = accel.Cmd(...)                     # VmacCmd.specialize(mem_awidth=…, data_bw=…)
    dst   = accel.execute(cmd, mem)            # the bit-exact golden (instance method)

``execute`` runs the fused op ``D = α·A·op(B) + β·C [, reduce]`` over a shared-memory
array, composing the merged ``FixedField`` / ``ComplexField`` operators (``mult`` /
``cmult``, ``add`` / ``cadd``, ``conj``), the wide-accumulator column reduction
:func:`~waveflow.hw.complexfield.csum` for ``reduce_rows``, and the output requantize
(right-shift ``SHIFT`` + round + saturate) via the ``ap_fixed``-exact integer requantizer.
Vectorized — no per-element Python loop.

The datapath: **multiply (data_bw × data_bw) → wide accumulate (full precision, ≤ acc_bw)
→ right-shift shift → round + saturate → write (out_bw)**.  The right-shift is the single
lossy step (an ``ap_fixed`` assignment), so the golden is bit-exact with the Vitis kernel.
``mem`` is the shared memory: a 1-D ``int64`` array of stored integers (``real`` mode) or a
1-D structured ``[('re','im')]`` array (``complex`` mode); operands are strided regions
``M[i, j] = mem[addr + i·row_stride + j·col_stride]``.  ``execute`` writes the requantized
result into ``mem`` at the ``d`` region (so commands compose) and returns the dst
``DataArray``.

The ``m_axi`` port + ``synth_fn`` (the actual kernel) are **not** here — they belong to
VMAC's own next phase, which this component shape unblocks.  Constructed without a ``sim``,
``VmacAccel`` is a lightweight params + golden object (no ``Simulation`` needed); a ``sim``
gives the full ``HwComponent`` (ports/synth, later).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from examples.vmac.vmac_cmd import VmacCmd, VmacMode
from waveflow.hw import fixpoint
from waveflow.hw.complexfield import ComplexField, cadd, cmult, conj, csum
from waveflow.hw.dataschema import DataArray
from waveflow.hw.fixpoint import FixedField
from waveflow.hw.hw_component import HwComponent, HwParam
from waveflow.hw.named import NamedObject
from waveflow.utils import complexutils as cx
from waveflow.utils import fixputils
from waveflow.utils.fixputils import Format, add_format, mult_format, sum_format


@dataclass
class VmacAccel(HwComponent):
    """A VMAC accelerator: structural ``HwParam`` widths + the Python golden ``execute``."""

    cpp_kernel_name: ClassVar[str | None] = "vmac"

    # structural params (synthesis-time HLS template params)
    mem_dwidth: HwParam[int] = 512          # MEM_BW (memory-interface width / throughput)
    mem_awidth: HwParam[int] = 32           # m_axi address width
    data_bw: HwParam[int] = 32              # IN_BW — operand element width
    acc_bw: HwParam[int] = 64               # accumulator width budget
    out_bw: HwParam[int] = 32               # writeback element width

    def __post_init__(self) -> None:
        if self.sim is None:
            # params + golden mode — usable without a Simulation (the m_axi port + synth_fn,
            # which need a sim, arrive in VMAC's kernel phase that this shape unblocks).
            self._wrap_hw_params()
            NamedObject.__post_init__(self)
            object.__setattr__(self, "_hw_construction_complete", True)
        else:
            super().__post_init__()

    @property
    def Cmd(self) -> type[VmacCmd]:
        """The command schema specialized to this accelerator's widths — the instance → type
        bridge from ``HwParam`` values to the schema's ``Param`` specialization."""
        return VmacCmd.specialize(mem_awidth=int(self.mem_awidth), data_bw=int(self.data_bw))

    # --- private datapath helpers (structural-param-free → static / class) -------
    @staticmethod
    def _fixed_cls(fmt: Format) -> type[FixedField]:
        return FixedField.specialize(fmt.W, fmt.int_bits, fmt.signed, fmt.q_mode, fmt.o_mode)

    @staticmethod
    def _region_idx(reg, n_rows: int, n_cols: int) -> np.ndarray:
        """The strided index matrix: ``addr + i·row_stride + j·col_stride``."""
        rows = np.arange(n_rows)[:, None] * int(reg.row_stride)
        cols = np.arange(n_cols)[None, :] * int(reg.col_stride)
        return int(reg.addr) + rows + cols

    @classmethod
    def _operand(cls, M: np.ndarray, in_fmt: Format, complex_mode: bool) -> DataArray:
        """Wrap a strided matrix view (stored ints / structured) as a DataArray operand."""
        inner = cls._fixed_cls(in_fmt)
        elem = ComplexField.specialize(inner) if complex_mode else inner
        shape = M.shape if M.shape else (1,)
        return DataArray.specialize(elem, max_shape=tuple(shape))(M)

    @classmethod
    def _scalar(cls, sc, n_cols: int, in_fmt: Format, mem: np.ndarray,
                complex_mode: bool) -> DataArray:
        """Build an alpha/beta operand: direct immediate (shape (1,)) or indirect per-column."""
        if bool(sc.direct):
            if complex_mode:
                M = cx.make_complex([int(sc.re)], [int(sc.im)], in_fmt)
            else:
                M = np.array([int(sc.re)], dtype=np.int64)
        else:
            idx = int(sc.addr) + np.arange(n_cols) * int(sc.stride)
            M = mem[idx]
        return cls._operand(M, in_fmt, complex_mode)

    @staticmethod
    def _mult(a: DataArray, b: DataArray, complex_mode: bool) -> DataArray:
        return cmult(a, b) if complex_mode else fixpoint.mult(a, b)

    @staticmethod
    def _add(a: DataArray, b: DataArray, complex_mode: bool) -> DataArray:
        return cadd(a, b) if complex_mode else fixpoint.add(a, b)

    @staticmethod
    def _requantize(t: DataArray, out_cls: type[FixedField], complex_mode: bool) -> DataArray:
        """Requantize the wide accumulator ``t`` into the output ``FixedField`` ``out_cls`` —
        right-shift + round + saturate, an ``ap_fixed`` assignment via the ``ap_fixed``-exact
        integer requantizer (``fixputils.quantize``)."""
        target = out_cls.get_format()
        if complex_mode:
            src = t.element_type.inner_format()
            re = fixputils.quantize(cx.re_of(t.val), src, target)
            im = fixputils.quantize(cx.im_of(t.val), src, target)
            elem = ComplexField.specialize(out_cls)
            struct = cx.make_complex(re, im, target)
            return DataArray.specialize(elem, max_shape=struct.shape)(struct)
        return fixpoint.quantize(t, out_cls)

    @staticmethod
    def _writeback(mem: np.ndarray, reg, dst: DataArray) -> None:
        val = np.asarray(dst.val)
        if val.ndim == 1:                                   # reduced -> single row of columns
            idx = int(reg.addr) + np.arange(val.shape[0]) * int(reg.col_stride)
        else:
            idx = VmacAccel._region_idx(reg, val.shape[0], val.shape[1])
        mem[idx] = val

    # --- numeric model: datapath format derivation (the codegen spec) ------------
    def _in_fmt(self, cmd: VmacCmd) -> Format:
        """The operand ``FixedField`` format: ``W = data_bw`` (structural), ``I = int_bits``
        (runtime), so ``F = data_bw - int_bits`` fractional bits."""
        return Format(int(self.data_bw), int(cmd.int_bits), signed=True)

    def accumulator_format(self, cmd: VmacCmd) -> Format:
        """The wide-accumulator ``FixedField`` format for ``cmd`` — full precision, no loss.

        Composes the datapath as pure format algebra (no data):

        - **product** ``A·op(B)``: ``data_bw × data_bw`` (fraction bits add) — real uses
          ``mult_format``, complex ``cmult_format`` (``sub_format``-based, +1 integer bit);
        - **scale** by ``alpha``: one more multiply;
        - **+ β·C**: aligned add (``add_format``) — fractions align, integer bits +1 (carry);
        - **row reduction**: ``+⌈log₂ n_rows⌉`` integer bits (``sum_format``).

        Fractional growth is identical for real and complex (complex's extra bit lands in
        the *integer* part), so ``F_acc`` depends only on the multiply depth."""
        complex_mode = VmacMode(int(cmd.mode)) == VmacMode.COMPLEX
        mul = cx.cmult_format if complex_mode else mult_format
        in_fmt = self._in_fmt(cmd)
        if bool(cmd.b_one):
            ab = in_fmt                                         # op(B) = 1, A·op(B) = A
        else:
            op_b = in_fmt
            if bool(cmd.b_conj) and complex_mode:               # conj grows the inner: (W+1, I+1)
                op_b = cx.conj_format(in_fmt)
            ab = mul(in_fmt, op_b)
        acc = mul(in_fmt, ab)                                   # alpha · A·op(B)
        if not bool(cmd.c_zero):
            acc = add_format(acc, mul(in_fmt, in_fmt))          # + beta · C (aligned, +1 int bit)
        if bool(cmd.reduce_rows):
            acc = sum_format(acc, int(cmd.n_rows))              # + ceil(log2 n_rows) int bits
        return acc

    def output_format(self, cmd: VmacCmd) -> type[FixedField]:
        """The exact output (per-lane) ``FixedField`` for ``cmd`` — the codegen target.

        The single lossy step is the right-shift ``SHIFT``, which picks the output binary
        point: ``F_out = F_acc − SHIFT`` (so ``I_out = out_bw − F_out``), then round
        (``q_mode``) + saturate (``o_mode``) into ``out_bw`` bits.  Fail-loud on a mis-sized
        accelerator: an accumulator wider than ``acc_bw``, a ``SHIFT`` past the accumulator's
        fractional bits, or an ``out_bw`` too small to hold the integer part.  (Complex
        output is a ``ComplexField`` over this per-lane format.)"""
        acc = self.accumulator_format(cmd)
        if acc.W > int(self.acc_bw):
            raise ValueError(
                f"accumulator width {acc.W} exceeds acc_bw={int(self.acc_bw)}; widen acc_bw.")
        shift = int(cmd.shift)
        out_frac = acc.frac_bits - shift
        if out_frac < 0:
            raise ValueError(
                f"SHIFT={shift} exceeds accumulator fractional bits {acc.frac_bits}; "
                "the right-shift would reach into the integer bits (SHIFT out of range).")
        out_bw = int(self.out_bw)
        int_bits = out_bw - out_frac
        if int_bits < 0:
            raise ValueError(
                f"out_bw={out_bw} is too small for the integer part: SHIFT={shift} keeps "
                f"{out_frac} fractional bits, exceeding out_bw (need out_bw >= {out_frac}).")
        # binary-point relationship: F_out = F_acc − SHIFT, hence I_out = out_bw − F_acc + SHIFT
        assert int_bits == out_bw - (acc.frac_bits - shift)
        return FixedField.specialize(out_bw, int_bits, acc.signed, cmd.q_mode, cmd.o_mode)

    # --- the golden ----------------------------------------------------------
    def execute(self, cmd: VmacCmd, mem: np.ndarray) -> DataArray:
        """Execute a ``VmacCmd`` over ``mem``; write the dst region and return the dst array."""
        complex_mode = VmacMode(int(cmd.mode)) == VmacMode.COMPLEX
        in_fmt = self._in_fmt(cmd)
        n, m = int(cmd.n_rows), int(cmd.n_cols)
        mem = np.asarray(mem)
        out_cls = self.output_format(cmd)       # fail-loud config guards (acc_bw / SHIFT / out_bw)

        def region(reg) -> DataArray:
            return self._operand(mem[self._region_idx(reg, n, m)], in_fmt, complex_mode)

        # op(B) and A·op(B)
        a = region(cmd.a)
        if bool(cmd.b_one):
            ab = a
        else:
            b = region(cmd.b)
            if bool(cmd.b_conj) and complex_mode:           # conj is a no-op for real data
                b = conj(b)
            ab = self._mult(a, b, complex_mode)

        # alpha · A·op(B)
        alpha = self._scalar(cmd.alpha, m, in_fmt, mem, complex_mode)
        t = self._mult(alpha, ab, complex_mode)

        # + beta · C
        if not bool(cmd.c_zero):
            beta = self._scalar(cmd.beta, m, in_fmt, mem, complex_mode)
            t = self._add(t, self._mult(beta, region(cmd.c), complex_mode), complex_mode)

        # optional row reduction (wide accumulator)
        if bool(cmd.reduce_rows):
            t = csum(t, axis=0)

        # invariant: the actual accumulator format must equal the derived spec (the format
        # algebra in accumulator_format mirrors the operators it composes).
        actual = t.element_type.inner_format() if complex_mode else t.element_type.get_format()
        spec = self.accumulator_format(cmd)
        if (actual.W, actual.int_bits, actual.signed) != (spec.W, spec.int_bits, spec.signed):
            raise AssertionError(
                f"accumulator format mismatch: actual {actual} != derived {spec}")

        dst = self._requantize(t, out_cls, complex_mode)
        self._writeback(mem, cmd.d, dst)
        return dst
