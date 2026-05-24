"""HwStmt — synthesizable statement IR.

Users never construct these nodes manually.  They are produced by
``HwStmtExtractor`` when ``HwComponent.build()`` parses ``run_proc``.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Expression nodes
# ---------------------------------------------------------------------------

class HwExpr:
    """Base class for synthesizable expression nodes."""


@dataclass
class Ref(HwExpr):
    """A reference to a bound ``HwVar``."""
    var: HwVar


@dataclass
class FieldRef(HwExpr):
    """Field access on a bound ``HwVar`` (e.g. ``cmd_hdr.nsamp``)."""
    var: HwVar
    field: str


# ---------------------------------------------------------------------------
# Variable binding
# ---------------------------------------------------------------------------

@dataclass
class HwVar:
    """A symbolic variable produced by a synthesizable statement.

    Created by ``HwStmtExtractor`` for each binding in ``run_proc``.
    """
    name: str
    typ: object  # type[DataSchema] | None (unresolved)
    producer: HwStmt | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Statement nodes
# ---------------------------------------------------------------------------

class HwStmt:
    """Base class for synthesizable statement IR nodes."""


@dataclass
class SeqStmt(HwStmt):
    """Sequential list of statements."""
    stmts: list[HwStmt]


@dataclass
class WhileStmt(HwStmt):
    """``while True:`` loop — maps to ``ap_ctrl_none`` in HLS."""
    body: SeqStmt


@dataclass
class CaseStmt(HwStmt):
    """Restricted ``if var.field <op> value:`` — maps to switch/if-else in C++.

    ``op`` is ``'=='`` or ``'!='``.  ``field`` is ``None`` when the compared
    expression is the bare ``HwVar`` itself (e.g. ``if err != NO_ERROR:``).
    """
    var: HwVar
    field: str | None
    value: object  # enum value or literal
    if_true: SeqStmt
    if_false: SeqStmt | None = None
    op: str = '=='


@dataclass
class ContinueStmt(HwStmt):
    """``continue`` inside ``while True``."""


@dataclass
class SynthCallStmt(HwStmt):
    """A call to a ``@synthesizable`` method with a ``synth_fn``.

    Base class for endpoint-owned statement types
    (``StreamGetStmt``, ``MMArrayReadStmt``, etc.).
    """
    method: object              # bound callable with _is_synthesizable=True
    inputs: list               # HwVar | InterfaceEndpoint | ast node
    outputs: list[HwVar]


@dataclass
class FunctionStmt(SynthCallStmt):
    """Call to a user-written ``@synthesizable`` method (no ``synth_fn``).

    Codegen emits a forward declaration in ``<component>.hpp`` and a call
    site in ``<component>.cpp``. The implementation lives in the impl file
    (default ``<component>_<function>_impl.cpp``), hand-written by the user.
    """
    impl_file: str | None = None


@dataclass
class ReturnStmt(HwStmt):
    """``return`` from the kernel function. Optional return value."""
    value: HwExpr | None = None


# ---------------------------------------------------------------------------
# Testbench-mode IR nodes (Phase 14)
# ---------------------------------------------------------------------------


@dataclass
class DutBindStmt(HwStmt):
    """A ``dut = <HwComponentSubclass>(**kwargs)`` binding inside ``main()``.

    The testbench emitter expands this into a block of local-variable
    declarations: one ``hls::stream<...>`` per stream endpoint plus one
    AXI-Lite scalar/array per regmap field.  The emitter also records a
    binding from the Python-side ``dut`` name to those locals so later
    statements (``dut.s_in.push(...)``, ``dut.run()`` etc.) can find the
    right C++ symbol.
    """
    local_name: str                 # Python-side local (e.g. "dut")
    comp_class: type                # The HwComponent subclass being bound
    kwargs: dict[str, object]       # Construction kwargs (resolved to literals)


@dataclass
class KernelCallStmt(HwStmt):
    """A ``<dut>.run()`` call site: invoke the DUT's kernel function.

    The emitter resolves the kernel name from the DUT's component class
    (via ``cpp_kernel_name``) and emits ``poly(s_in, m_out, halted,
    error, tx_id, coeffs);`` — passing the locals introduced by the
    matching ``DutBindStmt`` in canonical kernel-signature order.
    """
    local_name: str                 # Python-side DUT local (e.g. "dut")


@dataclass
class TbCallStmt(HwStmt):
    """A testbench-mode method call (push, pop, read_uint32_file, ...).

    Generic carrier: ``method_name`` is the Python-side method (e.g.
    ``"push"``), ``receiver`` is the bound object (a ``StreamIFSlave``,
    a ``DataSchema`` class, a regmap, etc. — resolved by the extractor),
    ``inputs`` and ``outputs`` follow the same shape as ``SynthCallStmt``.
    The testbench-mode emitter dispatches on ``method_name`` +
    ``type(receiver)`` to pick the right C++ lowering.
    """
    method_name: str
    receiver: object
    inputs: list
    outputs: list[HwVar]
    # Optional: for stream ops, the bound DUT's local name + endpoint attr
    # so the emitter can resolve "dut.s_in" → the right C++ stream variable.
    dut_local: str | None = None
    endpoint_attr: str | None = None
