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
    typ: object  # type[DataSchema] | SchemaArray | None (unresolved)
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
