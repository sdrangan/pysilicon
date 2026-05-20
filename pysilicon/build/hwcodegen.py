"""HwStmtExtractor — build-time AST → HwStmt IR converter.

``HwStmtExtractor`` is an ``ast.NodeVisitor`` that parses the ``run_proc``
source of an ``HwComponent``, recognises the synthesizable subset, and
returns a rooted ``HwStmt`` tree.  Everything outside the subset raises
``SynthesisError`` at build time.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pysilicon.hw.hw_component import HwComponent

from pysilicon.hw.hwstmt import (
    CaseStmt,
    ContinueStmt,
    FieldRef,
    HookStmt,
    HwStmt,
    HwVar,
    Ref,
    ReturnStmt,
    SeqStmt,
    SynthCallStmt,
    WhileStmt,
)


class SynthesisError(Exception):
    """Raised when ``run_proc`` contains a non-synthesizable pattern."""


class HwStmtExtractor:
    """Parse ``run_proc`` of an ``HwComponent`` into an ``HwStmt`` tree.

    Usage::

        extractor = HwStmtExtractor(comp)
        tree = extractor.extract()

    The returned tree is the root ``HwStmt`` (typically a ``WhileStmt`` for
    free-running components or a ``SeqStmt`` for per-invocation ones).
    """

    def __init__(self, comp: HwComponent, method_name: str = 'run_proc') -> None:
        self._comp = comp
        self._method_name = method_name
        self._scope: dict[str, HwVar] = {}

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
        stmts = self._visit_stmts(func_def.body)
        if len(stmts) == 1:
            return stmts[0]
        return SeqStmt(stmts=stmts)

    # ------------------------------------------------------------------
    # Statement dispatch
    # ------------------------------------------------------------------

    def _visit_stmts(self, stmts: list[ast.stmt]) -> list[HwStmt]:
        return [s for s in (self._visit_stmt(n) for n in stmts) if s is not None]

    def _visit_stmt(self, stmt: ast.stmt) -> HwStmt:
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
        # yield from self.ep.method(...)
        if isinstance(val, ast.YieldFrom):
            return self._make_call_stmt_from_node(val.value, node)
        # self.hook(...) — direct call, no yield from
        if isinstance(val, ast.Call):
            method = self._resolve_method(val.func)
            if getattr(method, '_is_sim_only', False):
                return None  # @sim_only — skip during synthesis
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
        # Only: if var.field == EnumValue:  or  if var.field != EnumValue:
        test = node.test
        if not isinstance(test, ast.Compare):
            raise SynthesisError(
                f"Non-synthesizable 'if' condition at line {node.lineno}; "
                f"only 'if var.field == value:' or 'if var.field != value:' is allowed"
            )
        if (len(test.ops) != 1
                or not isinstance(test.ops[0], (ast.Eq, ast.NotEq))
                or not isinstance(test.left, ast.Attribute)
                or not isinstance(test.left.value, ast.Name)):
            raise SynthesisError(
                f"Non-synthesizable 'if' condition at line {node.lineno}; "
                f"only 'if var.field == value:' or 'if var.field != value:' is allowed"
            )
        var_name = test.left.value.id
        if var_name not in self._scope:
            raise SynthesisError(
                f"Undefined variable '{var_name}' in 'if' at line {node.lineno}"
            )
        hw_var = self._scope[var_name]
        field_name = test.left.attr
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
        return HookStmt(method=method, inputs=inputs, outputs=outputs)

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
            return node.attr
        if isinstance(node, ast.Name):
            return node.id
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


# ---------------------------------------------------------------------------
# Module-level kernel-body selection policy
# ---------------------------------------------------------------------------


def extract_kernel(comp: HwComponent) -> HwStmt:
    """Pick ``on_start`` if the component has a ``VitisRegMapMMIFSlave`` endpoint;
    otherwise extract ``run_proc``."""
    from pysilicon.hw.regmap import VitisRegMapMMIFSlave  # local: avoid cycle
    for ep in getattr(comp, 'endpoints', {}).values():
        if isinstance(ep, VitisRegMapMMIFSlave):
            return HwStmtExtractor(comp, method_name='on_start').extract()
    return HwStmtExtractor(comp, method_name='run_proc').extract()
