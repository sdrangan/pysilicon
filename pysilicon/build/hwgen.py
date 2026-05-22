"""Walk a resolved ``HwStmt`` tree and emit C++ source as a string."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

from pysilicon.hw.dataschema import DataSchema, EnumField, FloatField, IntField
from pysilicon.hw.hwstmt import (
    CaseStmt,
    ContinueStmt,
    FieldRef,
    FunctionStmt,
    HwStmt,
    HwVar,
    Ref,
    ReturnStmt,
    SeqStmt,
    WhileStmt,
)
from pysilicon.hw.interface import (
    StreamDrainStmt,
    StreamGetStmt,
    StreamWriteStmt,
)
from pysilicon.hw.regmap import RegMapGetStmt, RegMapSetStmt

if TYPE_CHECKING:
    from pysilicon.hw.hw_component import HwComponent


@dataclass
class CodegenCtx:
    comp: HwComponent
    params: dict[str, str] = field(default_factory=dict)
    endpoint_names: dict[int, str] = field(default_factory=dict)
    indent: int = 1

    def pad(self) -> str:
        return "    " * self.indent

    def child(self) -> CodegenCtx:
        return CodegenCtx(
            comp=self.comp,
            params=self.params,
            endpoint_names=self.endpoint_names,
            indent=self.indent + 1,
        )


def to_cpp(stmt: HwStmt, ctx: CodegenCtx) -> str:
    """Emit C++ source for a statement (and its children). Returns a string."""
    if isinstance(stmt, WhileStmt):
        return _emit_while(stmt, ctx)
    if isinstance(stmt, SeqStmt):
        return _emit_seq(stmt, ctx)
    if isinstance(stmt, ReturnStmt):
        return _emit_return(stmt, ctx)
    if isinstance(stmt, ContinueStmt):
        return f"{ctx.pad()}continue;"
    if isinstance(stmt, CaseStmt):
        return _emit_case(stmt, ctx)
    if isinstance(stmt, StreamGetStmt):
        return _emit_stream_get(stmt, ctx)
    if isinstance(stmt, StreamWriteStmt):
        return _emit_stream_write(stmt, ctx)
    if isinstance(stmt, StreamDrainStmt):
        return _emit_stream_drain(stmt, ctx)
    if isinstance(stmt, RegMapGetStmt):
        return _emit_regmap_get(stmt, ctx)
    if isinstance(stmt, RegMapSetStmt):
        return _emit_regmap_set(stmt, ctx)
    if isinstance(stmt, FunctionStmt):
        return _emit_function_call(stmt, ctx)
    raise NotImplementedError(
        f"Codegen for {type(stmt).__name__} not implemented yet"
    )


def _emit_while(stmt: WhileStmt, ctx: CodegenCtx) -> str:
    body = to_cpp(stmt.body, ctx.child())
    return f"{ctx.pad()}while (true) {{\n{body}\n{ctx.pad()}}}"


def _emit_seq(stmt: SeqStmt, ctx: CodegenCtx) -> str:
    return "\n".join(to_cpp(child, ctx) for child in stmt.stmts)


def _emit_return(stmt: ReturnStmt, ctx: CodegenCtx) -> str:
    if stmt.value is None:
        return f"{ctx.pad()}return;"
    return f"{ctx.pad()}return {_emit_expr(stmt.value, ctx)};"


def _emit_case(stmt: CaseStmt, ctx: CodegenCtx) -> str:
    lhs = stmt.var.name if stmt.field is None else f"{stmt.var.name}.{stmt.field}"
    rhs = _emit_literal(stmt.value)
    cond = f"{lhs} {stmt.op} {rhs}"
    lines = [f"{ctx.pad()}if ({cond}) {{"]
    lines.append(to_cpp(stmt.if_true, ctx.child()))
    lines.append(f"{ctx.pad()}}}")
    if stmt.if_false is not None:
        lines[-1] = f"{ctx.pad()}}} else {{"
        lines.append(to_cpp(stmt.if_false, ctx.child()))
        lines.append(f"{ctx.pad()}}}")
    return "\n".join(lines)


def _emit_expr(expr, ctx: CodegenCtx) -> str:
    if isinstance(expr, HwVar):
        return expr.name
    if isinstance(expr, Ref):
        return expr.var.name
    if isinstance(expr, FieldRef):
        return f"{expr.var.name}.{expr.field}"
    return _emit_literal(expr)


def _emit_stream_get(stmt: StreamGetStmt, ctx: CodegenCtx) -> str:
    # stmt.inputs = [schema_class]; stmt.outputs = [HwVar]
    schema_cls = stmt.inputs[0]
    out = stmt.outputs[0]
    cpp_type = schema_cls.cpp_class_name()
    ep = stmt.method.__self__  # type: ignore[attr-defined]
    stream_name = _endpoint_name(ep, ctx)
    tmpl = _stream_template_arg(ep)
    pad = ctx.pad()
    return (
        f"{pad}{cpp_type} {out.name};\n"
        f"{pad}{out.name}.read_axi4_stream<{tmpl}>({stream_name});"
    )


def _emit_stream_write(stmt: StreamWriteStmt, ctx: CodegenCtx) -> str:
    # stmt.inputs = [HwVar of the value to write]
    value = stmt.inputs[0]
    ep = stmt.method.__self__  # type: ignore[attr-defined]
    stream_name = _endpoint_name(ep, ctx)
    tmpl = _stream_template_arg(ep)
    pad = ctx.pad()
    return f"{pad}{value.name}.write_axi4_stream<{tmpl}>({stream_name}, true);"


def _emit_stream_drain(stmt: StreamDrainStmt, ctx: CodegenCtx) -> str:
    ep = stmt.method.__self__  # type: ignore[attr-defined]
    stream_name = _endpoint_name(ep, ctx)
    tmpl = _stream_template_arg(ep)
    pad = ctx.pad()
    return f"{pad}streamutils::flush_axi4_stream_to_tlast<{tmpl}>({stream_name});"


def _endpoint_name(endpoint, ctx: CodegenCtx) -> str:
    """Find the Python attribute name on ``ctx.comp`` that this endpoint is bound to."""
    key = id(endpoint)
    if key in ctx.endpoint_names:
        return ctx.endpoint_names[key]
    for name, val in vars(ctx.comp).items():
        if val is endpoint:
            ctx.endpoint_names[key] = name
            return name
    raise RuntimeError(
        f"Endpoint {endpoint!r} not found on component {type(ctx.comp).__name__}"
    )


def _emit_regmap_get(stmt: RegMapGetStmt, ctx: CodegenCtx) -> str:
    # stmt.inputs = [field_name str]; stmt.outputs = [HwVar]
    field_name = stmt.inputs[0]
    out = stmt.outputs[0]
    cpp_type = (
        out.typ.cpp_class_name()
        if hasattr(out.typ, 'cpp_class_name')
        else 'auto'
    )
    return f"{ctx.pad()}{cpp_type} {out.name} = {field_name};"


def _emit_regmap_set(stmt: RegMapSetStmt, ctx: CodegenCtx) -> str:
    # stmt.inputs = [field_name str, value (HwVar | FieldRef | literal)]
    field_name = stmt.inputs[0]
    value = stmt.inputs[1]
    rhs = _emit_expr(value, ctx)
    return f"{ctx.pad()}{field_name} = {rhs};"


def _emit_function_call(stmt: FunctionStmt, ctx: CodegenCtx) -> str:
    """Emit a call to a ``@synthesizable`` user hook.

    The hook is a free function in C++ (no class wrapping); when a namespace
    is resolved for the component, the call site is explicitly qualified
    (``demo::process(...)``). Inputs become positional arguments; the
    return value (if any) is assigned to the output ``HwVar``, which is
    also declared inline.
    """
    if len(stmt.outputs) > 1:
        raise NotImplementedError(
            "Multi-output FunctionStmt (tuple return) is not supported"
        )
    func_name = stmt.method.__name__  # type: ignore[attr-defined]
    ns = resolved_namespace(type(ctx.comp))
    qualified = func_name if ns is None else f"{ns}::{func_name}"
    args = [_emit_call_arg(a, ctx) for a in stmt.inputs]
    arg_str = ", ".join(args)
    pad = ctx.pad()
    if not stmt.outputs:
        return f"{pad}{qualified}({arg_str});"
    out = stmt.outputs[0]
    cpp_t = (
        out.typ.cpp_class_name()
        if hasattr(out.typ, 'cpp_class_name')
        else _cpp_type_for(out.typ)
    )
    return f"{pad}{cpp_t} {out.name} = {qualified}({arg_str});"


def _cpp_type_for(typ) -> str:
    """Fallback C++ type for ``HwVar.typ`` values without ``cpp_class_name``."""
    if typ is None:
        return 'auto'
    # IntEnum subclasses lack cpp_class_name but render as themselves in C++.
    name = getattr(typ, '__name__', None)
    if name:
        return name
    return 'auto'


def _emit_call_arg(arg, ctx: CodegenCtx) -> str:
    """Emit one argument: HwVar -> name, endpoint -> attr name, literal -> literal."""
    if isinstance(arg, HwVar):
        return arg.name
    if isinstance(arg, FieldRef):
        return f"{arg.var.name}.{arg.field}"
    from pysilicon.hw.interface import InterfaceEndpoint
    if isinstance(arg, InterfaceEndpoint):
        return _endpoint_name(arg, ctx)
    return _emit_literal(arg)


def _emit_literal(value) -> str:
    """Emit a Python value as a C++ literal."""
    from enum import IntEnum
    if isinstance(value, IntEnum):
        return f"{type(value).__name__}::{value.name}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return f'"{value}"'
    return repr(value)


def cpp_type(typ) -> str:
    """Translate a Python type or ``HwVar.typ`` marker to its HLS C++ representation.

    Rules (matching the kernel-files plan):

    - ``IntField`` (``signed=False``) → ``ap_uint<bw>``
    - ``IntField`` (``signed=True``)  → ``ap_int<bw>``
    - ``EnumField`` (and bare ``IntEnum`` subclasses) → ``ap_uint<8>`` (v1)
    - ``FloatField`` (bitwidth 32) → ``float``; (bitwidth 64) → ``double``
    - Other ``DataSchema`` subclasses → ``cls.cpp_class_name()``
    - The 2-tuple ``('SchemaArray', elem_type)`` → ``<elem>[MAX_N] /* TODO ... */``
    - ``None`` → ``RuntimeError`` (unresolved type leaked through)
    """
    if typ is None:
        raise RuntimeError("cpp_type called with None — unresolved HwVar leaked")
    if isinstance(typ, tuple) and len(typ) == 2 and typ[0] == 'SchemaArray':
        inner = cpp_type(typ[1])
        return f"{inner}[MAX_N] /* TODO: real SchemaArray typing */"
    if isinstance(typ, type) and issubclass(typ, DataSchema):
        if issubclass(typ, IntField):
            bw = typ.get_bitwidth()
            signed = getattr(typ, 'signed', False)
            return f"ap_int<{bw}>" if signed else f"ap_uint<{bw}>"
        if issubclass(typ, FloatField):
            bw = typ.get_bitwidth()
            if bw == 32:
                return "float"
            if bw == 64:
                return "double"
            raise RuntimeError(
                f"FloatField bitwidth {bw} not supported (32 or 64 only)"
            )
        if issubclass(typ, EnumField):
            return "ap_uint<8>"  # v1 simplification; widen later if needed
        return typ.cpp_class_name()
    # Bare IntEnum subclasses (e.g. resolved FunctionStmt return types) map the
    # same as EnumField. The plan explicitly bounds enums at 8 bits for v1.
    if isinstance(typ, type) and issubclass(typ, IntEnum):
        return "ap_uint<8>"
    raise RuntimeError(f"cpp_type: cannot translate {typ!r}")


def _validate_no_name_collisions(comp) -> None:
    """Raise if ``HwParam``, regmap-field, and endpoint-attr names overlap."""
    import sys
    import typing
    from pysilicon.build.hwcodegen import SynthesisError
    from pysilicon.hw.hw_component import HwParam

    hw_param_names: set[str] = set()
    for klass in type(comp).__mro__:
        for name, hint in getattr(klass, '__annotations__', {}).items():
            if isinstance(hint, str):
                mod = sys.modules.get(klass.__module__)
                globs: dict = vars(mod) if mod is not None else {}
                try:
                    hint = eval(hint, globs)  # noqa: S307
                except Exception:
                    continue
            if typing.get_origin(hint) is HwParam:
                hw_param_names.add(name)

    endpoint_attrs = {name for name, _ in _discover_stream_endpoints(comp)}
    regmap_field_names: set[str] = set()
    regmap_slave = _discover_regmap(comp)
    if regmap_slave is not None:
        regmap_field_names = set(regmap_slave.regmap._fields.keys())

    pairs = [
        ('HwParam', 'endpoint',     hw_param_names & endpoint_attrs),
        ('HwParam', 'regmap field', hw_param_names & regmap_field_names),
        ('endpoint', 'regmap field', endpoint_attrs & regmap_field_names),
    ]
    collisions = [(a, b, names) for a, b, names in pairs if names]
    if collisions:
        msg = "; ".join(
            f"{a} vs {b}: {sorted(n)}" for a, b, n in collisions
        )
        raise SynthesisError(
            f"Name collisions in component {type(comp).__name__}: {msg}"
        )


def _collect_template_params(comp) -> list[str]:
    """Return ordered, deduplicated ``HwParam`` names used by endpoint bitwidths."""
    from pysilicon.hw.hw_component import HwParamValue
    params: list[str] = []
    seen: set[str] = set()
    for _attr, ep in _discover_stream_endpoints(comp):
        bw = ep.bitwidth  # type: ignore[attr-defined]
        if isinstance(bw, HwParamValue) and bw.param_name not in seen:
            params.append(bw.param_name)
            seen.add(bw.param_name)
    return params


def _stream_template_arg(ep) -> str:
    """Return the template argument string for an endpoint's bitwidth.

    ``HwParamValue`` → the param name. Plain ``int`` → the literal value.
    """
    from pysilicon.hw.hw_component import HwParamValue
    bw = ep.bitwidth
    if isinstance(bw, HwParamValue):
        return bw.param_name
    return str(int(bw))


def _discover_stream_endpoints(comp) -> list[tuple[str, object]]:
    """Return ``[(attr_name, endpoint)]`` for each stream endpoint on ``comp``."""
    from pysilicon.hw.interface import StreamIFMaster, StreamIFSlave
    result: list[tuple[str, object]] = []
    for name, val in vars(comp).items():
        if isinstance(val, (StreamIFSlave, StreamIFMaster)):
            result.append((name, val))
    return result


def _discover_regmap(comp):
    """Return the component's ``VitisRegMapMMIFSlave`` endpoint, or ``None``."""
    from pysilicon.hw.regmap import VitisRegMapMMIFSlave
    for val in vars(comp).values():
        if isinstance(val, VitisRegMapMMIFSlave):
            return val
    return None


def kernel_signature(comp) -> str:
    """Build the kernel function signature and ``#pragma HLS INTERFACE`` lines.

    Returns a multi-line string ending with the opening ``{`` of the function
    body and the pragmas. The caller appends the body and the closing ``}``.

    When ``HwParam``-driven endpoint bitwidths exist, a ``template <int p1,
    int p2>`` block is emitted preceding ``void <name>(``.
    """
    _validate_no_name_collisions(comp)
    name = cpp_kernel_name(type(comp))
    template_params = _collect_template_params(comp)
    template_decl = ""
    if template_params:
        decl_block = ", ".join(f"int {p}" for p in template_params)
        template_decl = f"template <{decl_block}>\n"

    arg_lines: list[str] = []
    pragma_lines: list[str] = []
    for attr, ep in _discover_stream_endpoints(comp):
        tmpl_arg = _stream_template_arg(ep)
        arg_lines.append(
            f"    hls::stream<streamutils::axi4s_word<{tmpl_arg}>>& {attr}"
        )
        pragma_lines.append(f"#pragma HLS INTERFACE axis port={attr}")
    regmap_slave = _discover_regmap(comp)
    if regmap_slave is not None:
        for fname, fld in regmap_slave.regmap._fields.items():
            if fld.is_vitis_auto:
                continue
            arg_lines.append(f"    {cpp_type(fld.schema)}& {fname}")
            pragma_lines.append(
                f"#pragma HLS INTERFACE s_axilite port={fname:<12} bundle=control"
            )
    pragma_lines.append(
        "#pragma HLS INTERFACE s_axilite port=return       bundle=control"
    )
    arg_block = ",\n".join(arg_lines)
    pragma_block = "\n".join(pragma_lines)
    return f"{template_decl}void {name}(\n{arg_block}\n) {{\n{pragma_block}"


def hook_signature(
    method, template_params: list[str] | None = None,
) -> tuple[str, list[tuple[str, str]]]:
    """Return ``(return_cpp_type, [(arg_name, arg_decl), ...])`` for a hook.

    Drops ``self``. Unwraps ``ProcessGen[T]`` return annotations (which alias
    to ``Generator[Event, Any, T]``). Stream-endpoint args become axis
    ``hls::stream<...>&`` references; other args translate via :func:`cpp_type`.

    When ``template_params`` is non-empty, the i-th stream-typed arg uses
    the i-th template-param name in its ``axi4s_word<...>`` expression
    instead of the symbolic ``WORD_BW``.
    """
    import collections.abc
    import inspect
    import typing

    from pysilicon.hw.interface import StreamIFMaster, StreamIFSlave

    hints = typing.get_type_hints(method)
    sig = inspect.signature(method)
    param_names = [name for name in sig.parameters if name != 'self']
    tparam_queue = list(template_params) if template_params else []
    args: list[tuple[str, str]] = []
    for name in param_names:
        annot = hints.get(name)
        if annot is None:
            raise RuntimeError(
                f"Hook '{method.__name__}' parameter '{name}' has no type annotation"
            )
        if isinstance(annot, type) and issubclass(annot, (StreamIFSlave, StreamIFMaster)):
            tmpl = tparam_queue.pop(0) if tparam_queue else "WORD_BW"
            args.append(
                (name, f"hls::stream<streamutils::axi4s_word<{tmpl}>>& {name}")
            )
            continue
        args.append((name, f"{cpp_type(annot)} {name}"))
    ret = hints.get('return')
    if ret is None:
        ret_cpp = "void"
    else:
        # ProcessGen[T] = Generator[Event, Any, T]; unwrap to T.
        if typing.get_origin(ret) is collections.abc.Generator:
            gen_args = typing.get_args(ret)
            if len(gen_args) == 3:
                ret = gen_args[2]
        ret_cpp = "void" if ret is type(None) else cpp_type(ret)
    return ret_cpp, args


def hook_signature_str(
    method, template_params: list[str] | None = None,
) -> str:
    """Return the full C++ declaration string for a hook (no body, no semicolon).

    When ``template_params`` is non-empty, the result is prefixed with a
    ``template <int p1, int p2, ...>`` line.
    """
    ret_cpp, args = hook_signature(method, template_params=template_params)
    arg_str = ", ".join(arg_decl for _, arg_decl in args)
    prefix = ""
    if template_params:
        params = ", ".join(f"int {p}" for p in template_params)
        prefix = f"template <{params}>\n"
    return f"{prefix}{ret_cpp} {method.__name__}({arg_str})"


# ---------------------------------------------------------------------------
# Hook templating (Phase 10): detect HwParam-driven stream args at the
# call site and decide whether a hook is emitted as a template (in a .tpp
# include) or a plain free function (in a .cpp impl).
# ---------------------------------------------------------------------------


def hook_template_params(stmt: FunctionStmt) -> list[str]:
    """Return the ordered, deduplicated ``HwParam`` names this hook is templated on.

    Walks ``stmt.inputs`` (call-site arguments), collecting the ``param_name``
    of any stream endpoint whose ``bitwidth`` is a ``HwParamValue``. An empty
    list means the hook is NOT templated and should be emitted in a ``.cpp``
    file. A non-empty list means the hook is templated and goes in a ``.tpp``.
    """
    from pysilicon.hw.hw_component import HwParamValue
    from pysilicon.hw.interface import StreamIFMaster, StreamIFSlave

    params: list[str] = []
    seen: set[str] = set()
    for inp in stmt.inputs:
        if isinstance(inp, (StreamIFSlave, StreamIFMaster)):
            bw = inp.bitwidth
            if isinstance(bw, HwParamValue) and bw.param_name not in seen:
                params.append(bw.param_name)
                seen.add(bw.param_name)
    return params


def _stmt_children(node) -> list:
    """Return immediate child statements of ``node`` for recursive tree walks."""
    if isinstance(node, WhileStmt):
        return [node.body]
    if isinstance(node, SeqStmt):
        return list(node.stmts)
    if isinstance(node, CaseStmt):
        children = [node.if_true]
        if node.if_false is not None:
            children.append(node.if_false)
        return children
    return []


def _validate_single_call_site(tree, hook_method, template_params: list[str]) -> None:
    """Templated hooks must be called from one site (or consistently across sites).

    Walks the tree, collects ``hook_template_params`` for every
    ``FunctionStmt`` targeting ``hook_method``, and raises ``SynthesisError``
    if any site yields a different param list.
    """
    from pysilicon.build.hwcodegen import SynthesisError

    sites: list[list[str]] = []

    def visit(node):
        if isinstance(node, FunctionStmt) and node.method is hook_method:
            sites.append(hook_template_params(node))
        for child in _stmt_children(node):
            visit(child)

    visit(tree)
    if len(sites) > 1 and any(s != sites[0] for s in sites[1:]):
        raise SynthesisError(
            f"Templated hook '{hook_method.__name__}' called from "
            f"{len(sites)} sites with inconsistent template params: {sites}"
        )


def cpp_kernel_name(comp_class) -> str:
    """Default kernel function name.

    ``CamelCase`` → ``snake_case`` with a trailing ``_component`` stripped:
    ``DemoComponent → demo``, ``PolyAccelComponent → poly_accel``. Override
    per class by setting ``cpp_kernel_name: ClassVar[str | None] = "..."``.
    """
    override = getattr(comp_class, 'cpp_kernel_name', None)
    if override:
        return override
    name = comp_class.__name__
    snake = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
    return snake.removesuffix('_component')


def resolved_namespace(comp_class) -> str | None:
    """Return the C++ namespace to use for this component's hooks.

    ``None`` means "no namespace, emit in global." Otherwise the returned
    string is the namespace name to wrap hooks in. Resolution rules:

    - ``cpp_namespace = ""`` → ``None`` (opt out; hooks emitted in global).
    - ``cpp_namespace = None`` → auto-derive from :func:`cpp_kernel_name`.
    - ``cpp_namespace = "<name>"`` → return ``"<name>"`` verbatim.
    """
    ns = getattr(comp_class, 'cpp_namespace', None)
    if ns == "":
        return None
    if ns is None:
        return cpp_kernel_name(comp_class)
    return ns


def _collect_schemas(tree: HwStmt, comp) -> list[type]:
    """Return the unique ``DataSchema`` subclasses referenced by the kernel.

    ``IntField`` and ``FloatField`` subclasses are excluded — they map to
    primitive C++ types (``ap_uint<N>`` / ``float``) and have no header of
    their own.  Matches the include set in ``poly.hpp``.

    Sources walked:
      - ``HwVar.typ`` on every ``stmt.outputs`` in the resolved tree.
      - Every regmap field schema (if the component carries a regmap).
    """
    def _has_header(typ) -> bool:
        return (
            isinstance(typ, type)
            and issubclass(typ, DataSchema)
            and not issubclass(typ, (IntField, FloatField))
        )

    schemas: dict[str, type] = {}

    def visit(node):
        if isinstance(node, WhileStmt):
            visit(node.body)
        elif isinstance(node, SeqStmt):
            for c in node.stmts:
                visit(c)
        elif isinstance(node, CaseStmt):
            visit(node.if_true)
            if node.if_false is not None:
                visit(node.if_false)
        for v in getattr(node, 'outputs', []):
            if _has_header(v.typ):
                schemas[v.typ.cpp_class_name()] = v.typ

    visit(tree)
    regmap_slave = _discover_regmap(comp)
    if regmap_slave is not None:
        for fld in regmap_slave.regmap._fields.values():
            if _has_header(fld.schema):
                schemas[fld.schema.cpp_class_name()] = fld.schema
    return list(schemas.values())


def _collect_hooks(tree: HwStmt) -> list:
    """Return the bound methods of every ``FunctionStmt`` reachable from ``tree``."""
    result: list = []
    seen: set[int] = set()

    def visit(node):
        if isinstance(node, FunctionStmt):
            key = id(node.method)
            if key not in seen:
                seen.add(key)
                result.append(node.method)
        if isinstance(node, WhileStmt):
            visit(node.body)
        elif isinstance(node, SeqStmt):
            for c in node.stmts:
                visit(c)
        elif isinstance(node, CaseStmt):
            visit(node.if_true)
            if node.if_false is not None:
                visit(node.if_false)

    visit(tree)
    return result


def _kernel_signature_decl(comp) -> str:
    """Same as :func:`kernel_signature` but without pragmas; trailing ``;``."""
    full = kernel_signature(comp)
    head, _sep, _tail = full.partition('\n) {')
    return head + '\n);'


def header_to_cpp(comp) -> str:
    """Build the content of ``<component>.hpp``.

    Kernel forward declaration stays in the global namespace.  Hook
    forward declarations are grouped inside a single
    ``namespace <ns> { ... }`` block when one is resolved; an opt-out
    component (``cpp_namespace = ""``) leaves them in global.
    """
    from pysilicon.build.hwcodegen import extract_kernel

    tree = extract_kernel(comp)
    schemas = _collect_schemas(tree, comp)
    lines = ['#pragma once', '']
    lines.append('#include "include/streamutils_hls.h"')
    for s in schemas:
        lines.append(f'#include "include/{s.cpp_class_name().lower()}.h"')  # type: ignore[attr-defined]
    lines.append('')
    lines.append(_kernel_signature_decl(comp))
    hooks = _collect_hooks(tree)
    if hooks:
        ns = resolved_namespace(type(comp))
        lines.append('')
        if ns is None:
            for hook in hooks:
                lines.append(f"{hook_signature_str(hook)};")
        else:
            lines.append(f"namespace {ns} {{")
            for hook in hooks:
                lines.append(f"    {hook_signature_str(hook)};")
            lines.append("}")
    return "\n".join(lines) + "\n"


def kernel_body_to_cpp(comp: HwComponent) -> str:
    """Top-level driver: extract + resolve + codegen.

    Returns the body of the kernel function (opened with ``{``, closed with
    ``}``).  No function signature, no pragmas — just the body.  Wrap
    separately in the next phase.
    """
    from pysilicon.build.hwcodegen import extract_kernel
    from pysilicon.hw.hw_component import SynthContext

    tree = extract_kernel(comp)
    synth_ctx = SynthContext.from_component(comp)
    ctx = CodegenCtx(comp=comp, params=synth_ctx.params, indent=1)
    body = to_cpp(tree, ctx)
    return f"{{\n{body}\n}}"


def kernel_to_cpp(comp) -> str:
    """Build the content of ``<component>.cpp`` — kernel function with body."""
    header_name = f"{cpp_kernel_name(type(comp))}.hpp"
    sig_with_pragmas = kernel_signature(comp)
    body = kernel_body_to_cpp(comp)
    body_inner = body.removeprefix("{\n").removesuffix("\n}")
    return (
        f'#include "{header_name}"\n\n'
        f"{sig_with_pragmas}\n"
        f"{body_inner}\n"
        f"}}\n"
    )


def _stub_default_return(ret_cpp: str) -> str:
    """Best-effort default ``return ...;`` for a stub body so the file compiles."""
    if ret_cpp == "void":
        return ""
    if ret_cpp.startswith("ap_uint") or ret_cpp.startswith("ap_int"):
        return f"return {ret_cpp}(0);"
    if ret_cpp in ("float", "double"):
        return f"return {ret_cpp}(0);"
    # struct: default-construct
    return f"return {ret_cpp}{{}};"


def impl_stub_to_cpp(comp, hook_method) -> str:
    """Build the first-time stub content for one hook impl file.

    When a namespace is resolved for the component, the function
    definition is wrapped in a ``namespace <ns> { ... }`` block.
    """
    header_name = f"{cpp_kernel_name(type(comp))}.hpp"
    ret_cpp, args = hook_signature(hook_method)
    arg_str = ", ".join(arg_decl for _, arg_decl in args)
    default = _stub_default_return(ret_cpp)
    body_lines = [f"    // TODO: implement {hook_method.__name__}"]
    if default:
        body_lines.append(f"    {default}")
    body = "\n".join(body_lines)
    func_def = (
        f"{ret_cpp} {hook_method.__name__}({arg_str}) {{\n"
        f"{body}\n"
        f"}}"
    )
    ns = resolved_namespace(type(comp))
    if ns is not None:
        func_def = f"namespace {ns} {{\n{func_def}\n}}"
    return (
        f'#include "{header_name}"\n\n'
        f"{func_def}\n"
    )


def kernel_files_to_str(comp) -> dict[str, str]:
    """Top-level driver. Returns ``{filename: contents}`` for all generated files."""
    from pysilicon.build.hwcodegen import extract_kernel

    name = cpp_kernel_name(type(comp))
    files: dict[str, str] = {
        f"{name}.hpp": header_to_cpp(comp),
        f"{name}.cpp": kernel_to_cpp(comp),
    }
    tree = extract_kernel(comp)
    for hook in _collect_hooks(tree):
        files[f"{name}_{hook.__name__}_impl.cpp"] = impl_stub_to_cpp(comp, hook)
    return files
