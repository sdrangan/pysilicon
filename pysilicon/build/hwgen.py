"""Walk a resolved ``HwStmt`` tree and emit C++ source as a string."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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
    stream_name = _endpoint_name(stmt.method.__self__, ctx)  # type: ignore[attr-defined]
    pad = ctx.pad()
    return (
        f"{pad}{cpp_type} {out.name};\n"
        f"{pad}{out.name}.read_axi4_stream<WORD_BW>({stream_name});"
    )


def _emit_stream_write(stmt: StreamWriteStmt, ctx: CodegenCtx) -> str:
    # stmt.inputs = [HwVar of the value to write]
    value = stmt.inputs[0]
    stream_name = _endpoint_name(stmt.method.__self__, ctx)  # type: ignore[attr-defined]
    pad = ctx.pad()
    return f"{pad}{value.name}.write_axi4_stream<WORD_BW>({stream_name}, true);"


def _emit_stream_drain(stmt: StreamDrainStmt, ctx: CodegenCtx) -> str:
    stream_name = _endpoint_name(stmt.method.__self__, ctx)  # type: ignore[attr-defined]
    pad = ctx.pad()
    return f"{pad}streamutils::flush_axi4_stream_to_tlast<WORD_BW>({stream_name});"


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

    The hook is a free function in C++ (no class wrapping). Inputs become
    positional arguments; the return value (if any) is assigned to the
    output ``HwVar``, which is also declared inline.
    """
    if len(stmt.outputs) > 1:
        raise NotImplementedError(
            "Multi-output FunctionStmt (tuple return) is not supported"
        )
    func_name = stmt.method.__name__  # type: ignore[attr-defined]
    args = [_emit_call_arg(a, ctx) for a in stmt.inputs]
    arg_str = ", ".join(args)
    pad = ctx.pad()
    if not stmt.outputs:
        return f"{pad}{func_name}({arg_str});"
    out = stmt.outputs[0]
    cpp_type = (
        out.typ.cpp_class_name()
        if hasattr(out.typ, 'cpp_class_name')
        else _cpp_type_for(out.typ)
    )
    return f"{pad}{cpp_type} {out.name} = {func_name}({arg_str});"


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
