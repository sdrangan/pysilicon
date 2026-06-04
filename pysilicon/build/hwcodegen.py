"""HwStmtExtractor — build-time AST → HwStmt IR converter.

``HwStmtExtractor`` is an ``ast.NodeVisitor`` that parses the ``run_proc``
source of an ``HwComponent``, recognises the synthesizable subset, and
returns a rooted ``HwStmt`` tree.  Everything outside the subset raises
``SynthesisError`` at build time.
"""
from __future__ import annotations

import ast
import inspect
import logging
import textwrap

_log = logging.getLogger(__name__)

from pysilicon.hw.hwstmt import (
    CaseStmt,
    ContinueStmt,
    DutBindStmt,
    FieldRef,
    FunctionStmt,
    HwStmt,
    HwVar,
    KernelCallStmt,
    Ref,
    ReturnStmt,
    SchemaBindStmt,
    SeqStmt,
    SynthCallStmt,
    TbFileIOStmt,
    TbRegmapFileReadStmt,
    TbStatusJsonStmt,
    TbStreamIOStmt,
    WhileStmt,
)


_TB_FILE_IO_METHODS = frozenset({
    'read_uint32_file', 'write_uint32_file',
    'read_uint32_file_array', 'write_uint32_file_array',
})
_TB_STREAM_PUSH_METHODS = frozenset({'push', 'push_array'})
_TB_STREAM_POP_METHODS = frozenset({'pop', 'pop_array'})


class SynthesisError(Exception):
    """Raised when ``run_proc`` contains a non-synthesizable pattern."""


_PIPELINED_OP_NAMES = frozenset({'get_pipelined', 'write_pipelined'})


class HwStmtExtractor:
    """Parse ``run_proc`` of an ``HwComponent`` into an ``HwStmt`` tree.

    Usage::

        extractor = HwStmtExtractor(comp)
        tree = extractor.extract()

    The returned tree is the root ``HwStmt`` (typically a ``WhileStmt`` for
    free-running components or a ``SeqStmt`` for per-invocation ones).
    """

    def __init__(
        self,
        comp,
        method_name: str = 'run_proc',
        is_testbench: bool = False,
    ) -> None:
        self._comp = comp
        self._method_name = method_name
        self._is_testbench = is_testbench
        self._scope: dict[str, HwVar] = {}
        # Testbench-mode side tables — empty in kernel mode, populated by
        # the extractor as it walks the body in Phase 3/4.
        self._duts: dict[str, object] = {}      # local_name -> HwComponent instance
        self._tb_locals: dict[str, object] = {} # local_name -> object (binding)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def extract(self) -> HwStmt:
        method = getattr(self._comp, self._method_name)
        src = inspect.getsource(method)
        src = textwrap.dedent(src)
        tree = ast.parse(src)
        func_def = tree.body[0]
        if not isinstance(func_def, ast.FunctionDef):
            raise SynthesisError(
                f"{self._method_name} source did not parse as a function"
            )
        if not self._is_testbench:
            # The implicit-capture rule applies to @synthesizable hook bodies
            # and on_start/run_proc.  Testbench main() bodies legitimately
            # read attributes of local DUTs (``dut.s_in.push(...)``), local
            # schema instances (``cmd_hdr.write_uint32_file(...)``), etc.,
            # so the rule isn't meaningful in TB mode — structural pattern
            # matching in _visit_stmt_tb handles validation instead.
            self._validate_no_implicit_capture(func_def)
        stmts = self._visit_stmts(func_def.body)
        if len(stmts) == 1:
            return stmts[0]
        return SeqStmt(stmts=stmts)

    # ------------------------------------------------------------------
    # Implicit-capture pre-pass
    # ------------------------------------------------------------------

    def _validate_no_implicit_capture(self, func_def: ast.FunctionDef) -> None:
        """Reject ``self.X`` attribute reads that aren't part of a known pattern.

        Allowed reads:
          - The function chain of a call whose target is ``@synthesizable`` or
            ``@sim_only`` (e.g. ``self.s_in.get`` in ``self.s_in.get(...)``).
          - Reads that resolve to a ``@sim_only`` callable.
          - Reads that resolve to an ``InterfaceEndpoint`` or a ``RegMap``
            (endpoint references passed into synthesizable calls).

        Anything else — a plain field like ``self.proc_latency`` used in an
        expression — raises ``SynthesisError``.

        Subtrees of ``@sim_only`` call statements are skipped entirely so the
        rule doesn't apply to their arguments either.
        """
        from pysilicon.hw.interface import InterfaceEndpoint
        from pysilicon.hw.regmap import RegMap

        extractor = self

        class _Validator(ast.NodeVisitor):
            def __init__(self) -> None:
                self._allowed_attr_ids: set[int] = set()

            def visit_Call(self, node: ast.Call) -> None:
                method = extractor._resolve_method(node.func)
                if getattr(method, '_is_sim_only', False):
                    return  # drop entire @sim_only call subtree
                self._mark_allowed(node.func)
                for arg in node.args:
                    self.visit(arg)
                for kw in node.keywords:
                    self.visit(kw.value)

            def _mark_allowed(self, attr_node: ast.expr) -> None:
                while isinstance(attr_node, ast.Attribute):
                    self._allowed_attr_ids.add(id(attr_node))
                    attr_node = attr_node.value

            def visit_Attribute(self, node: ast.Attribute) -> None:
                if id(node) in self._allowed_attr_ids:
                    return
                root: ast.expr = node
                while isinstance(root, ast.Attribute):
                    root = root.value
                if not (isinstance(root, ast.Name) and root.id == 'self'):
                    self.generic_visit(node)
                    return
                obj = extractor._resolve_obj(node)
                if obj is not None and (
                    getattr(obj, '_is_sim_only', False)
                    or getattr(obj, '_is_synthesizable', False)
                    or isinstance(obj, (InterfaceEndpoint, RegMap))
                ):
                    return
                lineno = getattr(node, 'lineno', '?')
                raise SynthesisError(
                    f"Implicit capture of 'self.{node.attr}' at line {lineno}. "
                    f"Reads of self.X inside a synthesizable method are forbidden "
                    f"unless 'X' is @sim_only, an endpoint, or a RegMap. Mark the "
                    f"value @sim_only or pass it explicitly."
                )

        _Validator().visit(func_def)

    # ------------------------------------------------------------------
    # Statement dispatch
    # ------------------------------------------------------------------

    def _visit_stmts(self, stmts: list[ast.stmt]) -> list[HwStmt]:
        return [s for s in (self._visit_stmt(n) for n in stmts) if s is not None]

    def _visit_stmt(self, stmt: ast.stmt) -> HwStmt | None:
        if self._is_testbench:
            tb_result = self._visit_stmt_tb(stmt)
            if tb_result is not None:
                return tb_result
            # Fall through to the kernel-mode handlers for statements that
            # apply identically in both modes (Return, If, Continue,
            # bare-Constant docstrings, etc.).
        if isinstance(stmt, ast.While):
            return self._visit_while(stmt)
        if isinstance(stmt, ast.Continue):
            return ContinueStmt()
        if isinstance(stmt, ast.Assign):
            return self._visit_assign(stmt)
        if isinstance(stmt, ast.AnnAssign):
            return self._visit_ann_assign(stmt)
        if isinstance(stmt, ast.Expr):
            return self._visit_expr_stmt(stmt)
        if isinstance(stmt, ast.If):
            return self._visit_if(stmt)
        if isinstance(stmt, ast.Return):
            return self._visit_return(stmt)
        raise SynthesisError(
            f"Non-synthesizable statement {type(stmt).__name__}"
            + (f" at line {stmt.lineno}" if hasattr(stmt, 'lineno') else "")
        )

    # ------------------------------------------------------------------
    # Testbench-mode statement handlers
    # ------------------------------------------------------------------

    def _visit_stmt_tb(self, stmt: ast.stmt) -> HwStmt | None:
        """Match TB-only stmt shapes. Return ``None`` to fall through to
        the kernel-mode handler (for stmts that have identical semantics in
        both modes, e.g. ``return``, ``if``, bare docstring constants).
        """
        # `name = <ClassRef>(**kwargs)` — DUT or schema binding.
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Call)
        ):
            local_name = stmt.targets[0].id
            bind = self._try_dut_bind(local_name, stmt.value, stmt)
            if bind is not None:
                return bind
            schema_bind = self._try_schema_bind(local_name, stmt.value, stmt)
            if schema_bind is not None:
                return schema_bind
        # Expression-statement calls (no return assignment): the bulk of TB
        # mode lives here — push/pop, file IO, regmap helpers, dut.run().
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            result = self._visit_tb_call(stmt.value, stmt)
            if result is not None:
                return result
        return None

    # ------------------------------------------------------------------
    # TB-mode call dispatch
    # ------------------------------------------------------------------

    def _visit_tb_call(
        self, call: ast.Call, parent: ast.stmt,
    ) -> HwStmt | None:
        """Lower a bare ``<chain>(...)`` expression statement to TB IR."""
        if not isinstance(call.func, ast.Attribute):
            return None
        attr = call.func.attr
        receiver_chain = call.func.value

        # `dut.run()` — kernel call.
        if (
            attr == 'run'
            and isinstance(receiver_chain, ast.Name)
            and receiver_chain.id in self._duts
        ):
            if call.args or call.keywords:
                raise SynthesisError(
                    f"dut.run() takes no arguments (line {parent.lineno})"
                )
            return KernelCallStmt(local_name=receiver_chain.id)

        # `<schema_local>.read_uint32_file*(...)` / `.write_uint32_file*(...)`
        if (
            attr in _TB_FILE_IO_METHODS
            and isinstance(receiver_chain, ast.Name)
            and receiver_chain.id in self._tb_locals
        ):
            return self._make_file_io_stmt(
                receiver_chain.id, attr, call, parent,
            )

        # `dut.<ep>.push(...)`, `.push_array(...)`, `.pop(...)`, `.pop_array(...)`
        if (
            (attr in _TB_STREAM_PUSH_METHODS or attr in _TB_STREAM_POP_METHODS)
            and isinstance(receiver_chain, ast.Attribute)
            and isinstance(receiver_chain.value, ast.Name)
            and receiver_chain.value.id in self._duts
        ):
            return self._make_stream_io_stmt(
                dut_local=receiver_chain.value.id,
                endpoint_attr=receiver_chain.attr,
                method_attr=attr,
                call=call,
                parent=parent,
            )

        # `dut.regmap.read_uint32_file_array(field, path, count=...)`
        # `dut.regmap.write_status_json(path, fields=[...])`
        if (
            isinstance(receiver_chain, ast.Attribute)
            and receiver_chain.attr == 'regmap'
            and isinstance(receiver_chain.value, ast.Name)
            and receiver_chain.value.id in self._duts
        ):
            return self._make_regmap_call_stmt(
                dut_local=receiver_chain.value.id,
                method_attr=attr,
                call=call,
                parent=parent,
            )

        return None

    # ------------------------------------------------------------------
    # TB-mode helpers
    # ------------------------------------------------------------------

    def _try_schema_bind(
        self,
        local_name: str,
        call_node: ast.Call,
        parent_stmt: ast.stmt,
    ) -> SchemaBindStmt | None:
        """If ``call_node`` constructs a ``DataSchema`` subclass with no
        arguments, return a ``SchemaBindStmt`` and record the binding."""
        from pysilicon.hw.dataschema import DataSchema

        cls = self._resolve_tb_class(call_node.func)
        if not (isinstance(cls, type) and issubclass(cls, DataSchema)):
            return None
        if call_node.args or call_node.keywords:
            raise SynthesisError(
                f"DataSchema construction in TB mode takes no arguments "
                f"(line {parent_stmt.lineno})"
            )
        self._tb_locals[local_name] = cls
        return SchemaBindStmt(local_name=local_name, schema_class=cls)

    def _make_file_io_stmt(
        self,
        target_local: str,
        method: str,
        call: ast.Call,
        parent: ast.stmt,
    ) -> TbFileIOStmt:
        """Build a :class:`TbFileIOStmt` from a ``<local>.<method>(...)`` call."""
        is_array = method.endswith('_array')
        is_write = method.startswith('write_')
        if not call.args:
            raise SynthesisError(
                f"{method}() requires a path argument (line {parent.lineno})"
            )
        path = call.args[0]
        count: object | None = None
        if is_array:
            count = self._extract_count_kwarg(call, method, parent)
            if len(call.args) > 1:
                raise SynthesisError(
                    f"{method}() takes path + count=...; extra positional "
                    f"args at line {parent.lineno}"
                )
        else:
            if len(call.args) > 1 or call.keywords:
                raise SynthesisError(
                    f"{method}() takes only a path argument "
                    f"(line {parent.lineno})"
                )
        return TbFileIOStmt(
            target_local=target_local,
            path=path,
            is_write=is_write,
            is_array=is_array,
            count=count,
        )

    def _make_stream_io_stmt(
        self,
        dut_local: str,
        endpoint_attr: str,
        method_attr: str,
        call: ast.Call,
        parent: ast.stmt,
    ) -> TbStreamIOStmt:
        """Build a :class:`TbStreamIOStmt` from a ``dut.<ep>.<method>(...)`` call."""
        is_pop = method_attr in _TB_STREAM_POP_METHODS
        is_array = method_attr.endswith('_array')
        if not call.args:
            raise SynthesisError(
                f"{method_attr}() requires a value argument "
                f"(line {parent.lineno})"
            )
        value_node = call.args[0]
        if not isinstance(value_node, ast.Name):
            raise SynthesisError(
                f"{method_attr}() value must be a local-name reference "
                f"(line {parent.lineno})"
            )
        value_local = value_node.id
        if value_local not in self._tb_locals:
            raise SynthesisError(
                f"{method_attr}({value_local}, ...) — '{value_local}' is "
                f"not a bound schema local (line {parent.lineno})"
            )
        count: object | None = None
        if is_array:
            count = self._extract_count_kwarg(call, method_attr, parent)
            if len(call.args) > 1:
                raise SynthesisError(
                    f"{method_attr}() takes value + count=...; extra "
                    f"positional args at line {parent.lineno}"
                )
        else:
            if len(call.args) > 1 or call.keywords:
                raise SynthesisError(
                    f"{method_attr}() takes only a value argument "
                    f"(line {parent.lineno})"
                )
        return TbStreamIOStmt(
            dut_local=dut_local,
            endpoint_attr=endpoint_attr,
            value_local=value_local,
            is_pop=is_pop,
            is_array=is_array,
            count=count,
        )

    def _make_regmap_call_stmt(
        self,
        dut_local: str,
        method_attr: str,
        call: ast.Call,
        parent: ast.stmt,
    ) -> HwStmt | None:
        """Handle ``dut.regmap.<method>(...)`` calls in TB mode."""
        if method_attr == 'read_uint32_file':
            if len(call.args) != 2 or call.keywords:
                raise SynthesisError(
                    f"dut.regmap.read_uint32_file(field, path) "
                    f"requires exactly two positional args "
                    f"(line {parent.lineno})"
                )
            field_node = call.args[0]
            if not (isinstance(field_node, ast.Constant)
                    and isinstance(field_node.value, str)):
                raise SynthesisError(
                    f"dut.regmap.read_uint32_file(field, ...): "
                    f"'field' must be a string literal (line {parent.lineno})"
                )
            return TbRegmapFileReadStmt(
                dut_local=dut_local,
                field_name=field_node.value,
                path=call.args[1],
                count=None,
            )
        if method_attr == 'read_uint32_file_array':
            if len(call.args) != 2:
                raise SynthesisError(
                    f"dut.regmap.read_uint32_file_array(field, path, count=...) "
                    f"requires exactly two positional args "
                    f"(line {parent.lineno})"
                )
            field_node = call.args[0]
            if not (isinstance(field_node, ast.Constant)
                    and isinstance(field_node.value, str)):
                raise SynthesisError(
                    f"dut.regmap.read_uint32_file_array(field, ...): "
                    f"'field' must be a string literal (line {parent.lineno})"
                )
            count = self._extract_count_kwarg(call, method_attr, parent)
            return TbRegmapFileReadStmt(
                dut_local=dut_local,
                field_name=field_node.value,
                path=call.args[1],
                count=count,
            )
        if method_attr == 'write_status_json':
            if len(call.args) != 1:
                raise SynthesisError(
                    f"dut.regmap.write_status_json(path, fields=[...]) "
                    f"requires exactly one positional arg "
                    f"(line {parent.lineno})"
                )
            fields_kw = next(
                (kw for kw in call.keywords if kw.arg == 'fields'), None,
            )
            if fields_kw is None or not isinstance(fields_kw.value, ast.List):
                raise SynthesisError(
                    f"dut.regmap.write_status_json(...) requires fields=[...]"
                    f" as a list literal (line {parent.lineno})"
                )
            names: list[str] = []
            for elt in fields_kw.value.elts:
                if not (isinstance(elt, ast.Constant)
                        and isinstance(elt.value, str)):
                    raise SynthesisError(
                        f"write_status_json fields must be string literals "
                        f"(line {parent.lineno})"
                    )
                names.append(elt.value)
            names = self._filter_vitis_auto_fields(dut_local, names)
            return TbStatusJsonStmt(
                dut_local=dut_local,
                path=call.args[0],
                field_names=names,
            )
        return None

    def _filter_vitis_auto_fields(
        self,
        dut_local: str,
        field_names: list[str],
    ) -> list[str]:
        """Drop ``is_vitis_auto`` fields from a ``write_status_json`` list.

        Vitis HLS auto-generates ``ap_start``/``ap_done`` inside the
        ``s_axilite`` control register — they are not C++ kernel
        parameters, so the generated TB cannot read them as locals.
        The Python side records them naturally; matching the symmetric
        Python shape just means listing them here too. This filter lets
        users write the idiomatic symmetric list and lowers it to the
        subset the C++ TB can actually emit.
        """
        regmap = self._dut_regmap_or_none(dut_local)
        if regmap is None:
            return field_names
        kept: list[str] = []
        skipped: list[str] = []
        for n in field_names:
            fld = regmap._fields.get(n)
            if fld is not None and getattr(fld, 'is_vitis_auto', False):
                skipped.append(n)
            else:
                kept.append(n)
        if skipped:
            _log.debug(
                "write_status_json on '%s': dropped is_vitis_auto fields %s "
                "(no C++ local in the generated TB)",
                dut_local, skipped,
            )
        return kept

    def _dut_regmap_or_none(self, dut_local: str):
        """Return the ``VitisRegMap`` of the DUT bound to ``dut_local``, or
        ``None`` if the DUT has no regmap.

        The DUT class is held in ``self._duts`` as the class itself; this
        helper instantiates it lazily with ``name``/``sim`` placeholders
        so we can inspect the regmap's field declarations at parse time.
        """
        cls = self._duts.get(dut_local)
        if cls is None or not isinstance(cls, type):
            return None
        from pysilicon.simulation.simulation import Simulation
        try:
            comp = cls(name="_codegen", sim=Simulation())
        except Exception:
            return None
        return getattr(comp, 'regmap', None)

    def _extract_count_kwarg(
        self,
        call: ast.Call,
        method_name: str,
        parent: ast.stmt,
    ) -> object:
        """Pull the required ``count=<expr>`` keyword from an array-mode call."""
        for kw in call.keywords:
            if kw.arg == 'count':
                return kw.value
        raise SynthesisError(
            f"{method_name}() requires count=<int> (line {parent.lineno})"
        )

    def _try_dut_bind(
        self,
        local_name: str,
        call_node: ast.Call,
        parent_stmt: ast.stmt,
    ) -> DutBindStmt | None:
        """If ``call_node`` constructs a ``HwComponent`` subclass, return a
        matching ``DutBindStmt``; otherwise ``None`` so the dispatcher can
        try other patterns.
        """
        from pysilicon.hw.hw_component import HwComponent

        cls = self._resolve_tb_class(call_node.func)
        if not (isinstance(cls, type) and issubclass(cls, HwComponent)):
            return None
        if call_node.args:
            raise SynthesisError(
                f"DUT construction must use keyword arguments only "
                f"(line {parent_stmt.lineno})"
            )
        kwargs: dict[str, object] = {}
        for kw in call_node.keywords:
            if kw.arg is None:
                raise SynthesisError(
                    f"DUT construction does not accept **kwargs "
                    f"(line {parent_stmt.lineno})"
                )
            if not isinstance(kw.value, ast.Constant):
                raise SynthesisError(
                    f"DUT construction kwargs must be literal constants "
                    f"(line {parent_stmt.lineno})"
                )
            kwargs[kw.arg] = kw.value.value
        self._duts[local_name] = cls
        return DutBindStmt(
            local_name=local_name, comp_class=cls, kwargs=kwargs,
        )

    def _resolve_tb_class(self, func_node: ast.expr) -> object | None:
        """Resolve a class reference in the testbench's ``main`` globals.

        Supports bare names (``PolyAccelComponent``) and attribute chains
        rooted in a global (``mod.PolyAccelComponent``).
        """
        method = getattr(self._comp, self._method_name)
        globs = getattr(method, '__globals__', {})
        if isinstance(func_node, ast.Name):
            return globs.get(func_node.id)
        if isinstance(func_node, ast.Attribute):
            path: list[str] = []
            node: ast.expr = func_node
            while isinstance(node, ast.Attribute):
                path.append(node.attr)
                node = node.value
            if not isinstance(node, ast.Name):
                return None
            obj = globs.get(node.id)
            for attr in reversed(path):
                if obj is None:
                    return None
                obj = getattr(obj, attr, None)
            return obj
        return None

    # ------------------------------------------------------------------
    # Individual statement handlers
    # ------------------------------------------------------------------

    def _visit_while(self, node: ast.While) -> WhileStmt:
        if not (isinstance(node.test, ast.Constant) and node.test.value is True):
            raise SynthesisError(
                f"Only 'while True:' is synthesizable (line {node.lineno})"
            )
        if node.orelse:
            raise SynthesisError(
                f"'while/else' is not synthesizable (line {node.lineno})"
            )
        body_stmts = self._visit_stmts(node.body)
        return WhileStmt(body=SeqStmt(stmts=body_stmts))

    def _visit_assign(self, node: ast.Assign) -> HwStmt:
        # x = yield from self.ep.method(...)
        if isinstance(node.value, ast.YieldFrom):
            return self._make_call_with_binding(node.value.value, node.targets)
        # a, b = self.hook(var1, var2)   — direct call, no yield from
        if isinstance(node.value, ast.Call):
            method = self._resolve_method(node.value.func)
            self._check_not_pipelined(method, node)
            self._require_synthesizable(method, node)
            assert method is not None
            inputs = self._resolve_call_args(node.value)
            outputs = self._make_output_vars(node.targets)
            stmt = self._make_call_stmt(method, inputs, outputs)
            for v in outputs:
                v.producer = stmt
                self._scope[v.name] = v
            return stmt
        raise SynthesisError(
            f"Non-synthesizable assignment at line {node.lineno}"
        )

    def _visit_ann_assign(self, node: ast.AnnAssign) -> HwStmt:
        # x: T = yield from self.ep.method(...)
        if node.value is None:
            raise SynthesisError(
                f"Bare annotated assignment is not synthesizable (line {node.lineno})"
            )
        if isinstance(node.value, ast.YieldFrom):
            return self._make_call_with_binding(
                node.value.value, [node.target]
            )
        raise SynthesisError(
            f"Non-synthesizable annotated assignment at line {node.lineno}"
        )

    def _visit_expr_stmt(self, node: ast.Expr) -> HwStmt | None:
        val = node.value
        # Docstring or other bare constant — drop silently.
        if isinstance(val, ast.Constant):
            return None
        # yield from self.ep.method(...)
        if isinstance(val, ast.YieldFrom):
            return self._make_call_stmt_from_node(val.value, node)
        # self.hook(...) — direct call, no yield from
        if isinstance(val, ast.Call):
            method = self._resolve_method(val.func)
            if getattr(method, '_is_sim_only', False):
                return None  # @sim_only — skip during synthesis
            self._check_not_pipelined(method, node)
            self._require_synthesizable(method, node)
            assert method is not None
            inputs = self._resolve_call_args(val)
            return self._make_call_stmt(method, inputs, [])
        raise SynthesisError(
            f"Non-synthesizable expression statement at line {node.lineno}"
        )

    def _visit_return(self, node: ast.Return) -> ReturnStmt:
        if node.value is None:
            return ReturnStmt(value=None)
        val = node.value
        if isinstance(val, ast.Name):
            if val.id not in self._scope:
                raise SynthesisError(
                    f"Return of undefined variable '{val.id}' at line {node.lineno}"
                )
            return ReturnStmt(value=Ref(var=self._scope[val.id]))
        if (isinstance(val, ast.Attribute)
                and isinstance(val.value, ast.Name)
                and val.value.id in self._scope):
            return ReturnStmt(
                value=FieldRef(var=self._scope[val.value.id], field=val.attr)
            )
        raise SynthesisError(
            f"Non-synthesizable return value at line {node.lineno}"
        )

    def _visit_if(self, node: ast.If) -> CaseStmt:
        # Allowed: 'if var.field <op> value:' or 'if var <op> value:'
        test = node.test
        if not isinstance(test, ast.Compare):
            raise SynthesisError(
                f"Non-synthesizable 'if' condition at line {node.lineno}; "
                f"only 'if var.field == value:' or 'if var <op> value:' is allowed"
            )
        if (len(test.ops) != 1
                or not isinstance(test.ops[0], (ast.Eq, ast.NotEq))):
            raise SynthesisError(
                f"Non-synthesizable 'if' condition at line {node.lineno}; "
                f"only '==' and '!=' comparisons are allowed"
            )
        left = test.left
        field_name: str | None
        if isinstance(left, ast.Attribute) and isinstance(left.value, ast.Name):
            var_name = left.value.id
            field_name = left.attr
        elif isinstance(left, ast.Name):
            var_name = left.id
            field_name = None
        else:
            raise SynthesisError(
                f"Non-synthesizable 'if' condition at line {node.lineno}; "
                f"only 'if var.field <op> value:' or 'if var <op> value:' is allowed"
            )
        if var_name not in self._scope:
            raise SynthesisError(
                f"Undefined variable '{var_name}' in 'if' at line {node.lineno}"
            )
        hw_var = self._scope[var_name]
        cmp_val = self._eval_const(test.comparators[0])
        op = '==' if isinstance(test.ops[0], ast.Eq) else '!='
        body_stmts = self._visit_stmts(node.body)
        else_stmts = self._visit_stmts(node.orelse) if node.orelse else None
        return CaseStmt(
            var=hw_var,
            field=field_name,
            value=cmp_val,
            if_true=SeqStmt(stmts=body_stmts),
            if_false=SeqStmt(stmts=else_stmts) if else_stmts is not None else None,
            op=op,
        )

    # ------------------------------------------------------------------
    # Call-building helpers
    # ------------------------------------------------------------------

    def _make_call_with_binding(
        self,
        call_node: ast.expr,
        targets: list[ast.expr],
    ) -> HwStmt:
        if not isinstance(call_node, ast.Call):
            raise SynthesisError("'yield from' must be followed by a call expression")
        method = self._resolve_method(call_node.func)
        self._check_not_pipelined(method, call_node)
        self._require_synthesizable(method, call_node)
        assert method is not None
        inputs = self._resolve_call_args(call_node)
        outputs = self._make_output_vars(targets)
        stmt = self._make_call_stmt(method, inputs, outputs)
        for v in outputs:
            v.producer = stmt
            self._scope[v.name] = v
        return stmt

    def _make_call_stmt_from_node(
        self, call_node: ast.expr, parent: ast.stmt
    ) -> HwStmt:
        if not isinstance(call_node, ast.Call):
            raise SynthesisError(
                f"'yield from' must be followed by a call expression "
                f"(line {getattr(parent, 'lineno', '?')})"
            )
        method = self._resolve_method(call_node.func)
        self._check_not_pipelined(method, parent)
        self._require_synthesizable(method, parent)
        assert method is not None
        inputs = self._resolve_call_args(call_node)
        return self._make_call_stmt(method, inputs, [])

    def _make_call_stmt(
        self,
        method: object,
        inputs: list,
        outputs: list[HwVar],
    ) -> HwStmt:
        stmt_class = getattr(method, '_stmt_class', None)
        if stmt_class is not None:
            return stmt_class(method=method, inputs=inputs, outputs=outputs)
        synth_fn = getattr(method, '_synth_fn', None)
        if synth_fn is not None:
            return SynthCallStmt(method=method, inputs=inputs, outputs=outputs)
        impl_file = getattr(method, '_impl_file', None)
        return FunctionStmt(
            method=method, inputs=inputs, outputs=outputs, impl_file=impl_file,
        )

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    def _resolve_method(self, func_node: ast.expr) -> object | None:
        """Resolve ``self.ep.method`` to the actual bound method."""
        if not isinstance(func_node, ast.Attribute):
            return None
        obj = self._resolve_obj(func_node.value)
        if obj is None:
            return None
        return getattr(obj, func_node.attr, None)

    def _resolve_obj(self, node: ast.expr) -> object | None:
        """Resolve ``self`` or attribute chains rooted at ``self``."""
        if isinstance(node, ast.Name) and node.id == 'self':
            return self._comp
        if isinstance(node, ast.Attribute):
            parent = self._resolve_obj(node.value)
            if parent is None:
                return None
            return getattr(parent, node.attr, None)
        return None

    def _resolve_call_args(self, call_node: ast.Call) -> list:
        result: list = []
        for arg in call_node.args:
            if isinstance(arg, ast.Name) and arg.id in self._scope:
                result.append(self._scope[arg.id])
            elif isinstance(arg, ast.Attribute):
                obj = self._resolve_obj(arg)
                result.append(obj if obj is not None else arg)
            else:
                result.append(arg)
        return result

    def _make_output_vars(self, targets: list[ast.expr]) -> list[HwVar]:
        names = self._extract_names(targets)
        return [HwVar(name=n, typ=None) for n in names]

    @staticmethod
    def _extract_names(targets: list[ast.expr]) -> list[str]:
        names: list[str] = []
        for target in targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        names.append(elt.id)
        return names

    def _eval_const(self, node: ast.expr) -> object:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Attribute):
            obj = self._resolve_obj(node.value)
            if obj is not None:
                return getattr(obj, node.attr, node.attr)
            # Module-level reference (e.g. DemoCmdType.END) — preserve the AST
            # node so the resolver can turn it into the real Python value.
            return node
        if isinstance(node, ast.Name):
            return node
        raise SynthesisError(
            f"Cannot evaluate constant expression: {ast.dump(node)}"
        )

    @staticmethod
    def _require_synthesizable(method: object | None, node: ast.AST) -> None:
        if method is None or not getattr(method, '_is_synthesizable', False):
            lineno = getattr(node, 'lineno', '?')
            name = (
                getattr(method, '__name__', repr(method))
                if method is not None
                else '<unresolved>'
            )
            raise SynthesisError(
                f"Call to non-synthesizable method '{name}' at line {lineno}. "
                f"Mark it @synthesizable or @sim_only."
            )

    @staticmethod
    def _check_not_pipelined(method: object | None, node: ast.AST) -> None:
        if method is None:
            return
        method_name = getattr(method, '__name__', None)
        if method_name in _PIPELINED_OP_NAMES:
            lineno = getattr(node, 'lineno', '?')
            raise SynthesisError(
                f"Pipelined stream operation '{method_name}' at line {lineno} of "
                f"the extracted body. Pipelined ops are only legal inside "
                f"@synthesizable hook bodies (their C++ lowering requires "
                f"hand-written pipelined loops with #pragma HLS PIPELINE). "
                f"Refactor to call a hook that takes the stream as an argument "
                f"and does the pipelined I/O internally."
            )


# ---------------------------------------------------------------------------
# Module-level kernel-body selection policy
# ---------------------------------------------------------------------------


def extract_kernel(comp) -> HwStmt:
    """Extract the kernel body and return a fully-resolved ``HwStmt`` tree.

    Picks ``on_start`` if the component has a ``VitisRegMapMMIFSlave``
    endpoint, otherwise ``run_proc``.  The returned tree has every ``ast.*``
    node replaced with the real Python value and every output ``HwVar``
    typed where possible.

    For ``HwTestbench`` subclasses this routes to :func:`extract_testbench`
    instead — the testbench-mode entry point is ``main()`` with a different
    rule profile.
    """
    if getattr(type(comp), '_is_testbench', False):
        return extract_testbench(comp)
    from pysilicon.build.hwresolve import resolve_kernel  # local: avoid cycle
    from pysilicon.hw.regmap import VitisRegMapMMIFSlave
    for ep in getattr(comp, 'endpoints', {}).values():
        if isinstance(ep, VitisRegMapMMIFSlave):
            tree = HwStmtExtractor(comp, method_name='on_start').extract()
            return resolve_kernel(tree, comp)
    tree = HwStmtExtractor(comp, method_name='run_proc').extract()
    return resolve_kernel(tree, comp)


def extract_testbench(comp) -> HwStmt:
    """Extract the testbench ``main()`` body and return a resolved ``HwStmt`` tree.

    Companion to :func:`extract_kernel`, but for the testbench-source
    pathway introduced by Phase 14: the component is a ``HwTestbench``
    subclass whose ``main()`` method is the codegen root.  Rule profile
    differs from the kernel side — blocking stream ops, file I/O, and
    DUT construction are legal, while pipelined ops stay forbidden.
    """
    from pysilicon.build.hwresolve import resolve_testbench
    extractor = HwStmtExtractor(comp, method_name='main', is_testbench=True)
    tree = extractor.extract()
    return resolve_testbench(tree, comp)
