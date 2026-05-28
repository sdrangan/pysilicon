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


@dataclass
class SchemaBindStmt(HwStmt):
    """A ``name = <DataSchemaSubclass>()`` binding inside ``main()``.

    Declares a C++ schema local matching the Python instance.  For
    structured schemas (``DataSchema``, ``DataList``) this emits
    ``<SchemaCpp> <name>;``.  For raw-storage ``DataArray`` subclasses,
    this emits a C-array of the element type sized at the schema's
    declared ``max_shape[0]`` (e.g. ``float samp_in[128] = {};``).
    """
    local_name: str
    schema_class: type   # DataSchema subclass


@dataclass
class TbFileIOStmt(HwStmt):
    """File I/O on a TB schema local: ``read_uint32_file*`` /
    ``write_uint32_file*``.

    Captured as a single IR node with ``is_write`` and ``is_array``
    booleans.  The emitter dispatches to either ``streamutils::*`` (for
    structured schemas) or ``<elem>_array_utils::*`` (for raw arrays).
    """
    target_local: str
    path: object             # AST expr — emitted via _emit_str_expr
    is_write: bool
    is_array: bool
    count: object | None = None     # only for the ``_array`` variant


@dataclass
class TbStreamIOStmt(HwStmt):
    """Stream IO on a DUT endpoint: ``dut.<ep>.push/pop[_array](...)``.

    ``is_pop`` distinguishes push from pop; ``is_array`` distinguishes
    single-schema from bulk-array variants.  ``value_local`` is the
    schema local being pushed / popped into (for ``pop`` the schema is
    populated by the call); ``count`` is the element count for array
    variants.
    """
    dut_local: str
    endpoint_attr: str
    value_local: str
    is_pop: bool
    is_array: bool
    count: object | None = None


@dataclass
class TbRegmapFileReadStmt(HwStmt):
    """Regmap file preload helper in testbench mode.

    Supports both:

    - ``dut.regmap.read_uint32_file(field, path)`` for single-word scalar fields
    - ``dut.regmap.read_uint32_file_array(field, path, count=...)`` for raw arrays
    """
    dut_local: str
    field_name: str
    path: object
    count: object | None


@dataclass
class TbStatusJsonStmt(HwStmt):
    """``dut.regmap.write_status_json(path, fields=[...])`` — write the
    final values of the named regmap fields as a flat JSON object.

    Field values are read from the C++ regmap locals that the DUT
    binding introduced (the same locals the kernel call took as
    references).  The emitter lowers this to an inline ``std::ofstream``
    block matching the hand-written ``examples/poly/poly_tb.cpp``
    schema.
    """
    dut_local: str
    path: object
    field_names: list[str]
