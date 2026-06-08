"""``VmacCmd`` — the VMAC fused-instruction DataSchema (runtime tier).

A configurable vector MAC engine (VMAC) executes the fused op

    D = alpha · A · op(B) + beta · C   [, reduced over rows]

over a **strided** region of shared memory.  Parameters split into two tiers (see
:class:`~examples.vmac.golden.VmacAccel`): the **structural** widths that size silicon
(``mem_dwidth`` / ``mem_awidth`` / ``data_bw`` / ``acc_bw`` / ``out_bw``) live on the
accelerator, and the **runtime** instruction lives here in ``VmacCmd``: the operand
regions (addr + strides; the matrix shape is global), the ``alpha`` / ``beta`` scalars
(direct immediate or indirect pointer+stride, scalar or per-column), the op flags
(``b_one`` / ``c_zero`` / ``b_conj`` / ``reduce_rows``), the ``real`` | ``complex`` mode,
the fractional split ``int_bits``, the output ``shift``, and the round / saturate flags.

The width-bearing fields are set by the accelerator that produces/consumes the command
(``addr`` is ``mem_awidth`` bits; the immediate ``re`` / ``im`` are ``data_bw`` bits).
These schemas are **Level-2 declarative** (:class:`~waveflow.hw.dataschema.ParamSchema`):
each declares :class:`~waveflow.hw.param.Param` attributes and a dict-literal ``elements``
that references them directly — the core ``IntField.specialize`` / ``Region.specialize``
calls defer on the symbolic params, and ``specialize(**vals)`` resolves + caches (with the
``Region`` / ``Scalar`` cascade sharing params).  ``VmacCmd`` is still a plain ``DataList``,
so it serializes / deserializes and code-generates like any schema.
"""
from __future__ import annotations

from enum import IntEnum

from waveflow.hw import BooleanField, EnumField, IntField, Param, ParamSchema
from waveflow.utils.fixputils import OMode, QMode

# --- field aliases (names match the auto-generated IntField subclass __name__) ----
UInt16 = IntField.specialize(16, signed=False)
UInt8 = IntField.specialize(8, signed=False)


class VmacMode(IntEnum):
    """Datapath mode — sets the element width (``IN_BW`` real / ``2·IN_BW`` complex)."""
    REAL = 0
    COMPLEX = 1


ModeField = EnumField.specialize(VmacMode)


class Region(ParamSchema):
    """A strided operand region: ``M[i, j] = mem[addr + i·row_stride + j·col_stride]``."""

    mem_awidth = Param(32)
    elements = {
        "addr": {"schema": IntField.specialize(mem_awidth, signed=False),
                 "description": "base offset into shared memory"},
        "row_stride": IntField.specialize(mem_awidth, signed=True),
        "col_stride": IntField.specialize(mem_awidth, signed=True),
    }


class Scalar(ParamSchema):
    """An ``alpha`` / ``beta`` operand: direct immediate (``re`` / ``im`` stored ints) or
    indirect (per-column pointer ``addr`` + ``stride``; ``stride 0`` broadcasts)."""

    mem_awidth = Param(32)
    data_bw = Param(32)
    elements = {
        "direct": {"schema": BooleanField,
                   "description": "True = immediate re/im; False = indirect addr/stride"},
        "re": IntField.specialize(data_bw, signed=True),    # immediate stored int (real part)
        "im": IntField.specialize(data_bw, signed=True),    # immediate stored int (imag part)
        "addr": IntField.specialize(mem_awidth, signed=False),
        "stride": IntField.specialize(mem_awidth, signed=True),
    }


class VmacCmd(ParamSchema):
    """The VMAC fused instruction — the runtime tier (see module docstring).

    Region/scalar field widths track the accelerator's ``mem_awidth`` (addresses) and
    ``data_bw`` (immediates); the cascade runs through ``Region`` / ``Scalar`` specialize
    (both share ``mem_awidth`` with ``VmacCmd`` and resolve to the same cached classes).
    """

    mem_awidth = Param(32)
    data_bw = Param(32)
    elements = {
        # global matrix shape (operands share it; dst is (1, n_cols) when reduced)
        "n_rows": UInt16,
        "n_cols": UInt16,
        # strided operand / destination regions (cascade: share mem_awidth)
        "a": Region.specialize(mem_awidth=mem_awidth),
        "b": Region.specialize(mem_awidth=mem_awidth),
        "c": Region.specialize(mem_awidth=mem_awidth),
        "d": Region.specialize(mem_awidth=mem_awidth),
        # scaling scalars (cascade: share mem_awidth and data_bw)
        "alpha": Scalar.specialize(mem_awidth=mem_awidth, data_bw=data_bw),
        "beta": Scalar.specialize(mem_awidth=mem_awidth, data_bw=data_bw),
        # op flags
        "b_one": {"schema": BooleanField, "description": "op(B) = 1 (skip the A·B multiply)"},
        "c_zero": {"schema": BooleanField, "description": "drop the beta·C term"},
        "b_conj": {"schema": BooleanField, "description": "op(B) = conj(B) (complex; no-op for real)"},
        "reduce_rows": {"schema": BooleanField, "description": "sum the rows (per-column reduction)"},
        # datapath mode + runtime numeric format
        "mode": ModeField,
        "int_bits": UInt8,        # I of the operand format (F = data_bw - int_bits)
        "shift": UInt8,           # output right-shift (the single lossy step)
        "q_rnd": {"schema": BooleanField, "description": "output rounding: False = AP_TRN, True = AP_RND"},
        "o_sat": {"schema": BooleanField, "description": "output overflow: False = AP_WRAP, True = AP_SAT"},
    }

    @property
    def q_mode(self) -> QMode:
        """The quantization mode this command selects (from its ``q_rnd`` flag)."""
        return QMode.AP_RND if bool(self.q_rnd) else QMode.AP_TRN

    @property
    def o_mode(self) -> OMode:
        """The overflow mode this command selects (from its ``o_sat`` flag)."""
        return OMode.AP_SAT if bool(self.o_sat) else OMode.AP_WRAP
