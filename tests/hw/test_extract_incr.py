"""Phase 2 extractor tests: m_axi array read/write become synthesizable IR."""
from __future__ import annotations

from pysilicon.build.hwcodegen import extract_kernel
from pysilicon.hw.hwstmt import (
    FieldRef,
    FunctionStmt,
    MMArrayReadStmt,
    MMArrayWriteStmt,
    SeqStmt,
)
from pysilicon.hw.interface import StreamGetStmt
from pysilicon.simulation.simulation import Simulation


def _extract():
    from examples.increment.incr import IncrAccel
    comp = IncrAccel(name="a", sim=Simulation())
    tree = extract_kernel(comp)
    stmts = tree.stmts if isinstance(tree, SeqStmt) else [tree]
    return comp, stmts


def test_kernel_root_is_straightline_sequence():
    comp, stmts = _extract()
    kinds = [type(s).__name__ for s in stmts]
    assert kinds == [
        "StreamGetStmt",       # cmd = s_in.get(IncrCmd)
        "MMArrayReadStmt",     # buf = m_mem.read_array(...)
        "FunctionStmt",        # out = transform(buf, cmd.n)
        "MMArrayWriteStmt",    # m_mem.write_array(out, ..., cmd.n)
        "FunctionStmt",        # respond(m_out)
    ]


def test_mm_array_read_stmt_fields():
    from examples.increment.incr import IncrCmd, Uint32Field
    comp, stmts = _extract()
    rd = next(s for s in stmts if isinstance(s, MMArrayReadStmt))

    assert rd.port is comp.m_mem
    assert rd.elem_type is Uint32Field
    # count = cmd.n, addr = cmd.addr (both FieldRefs on the cmd HwVar)
    assert isinstance(rd.count_expr, FieldRef)
    assert rd.count_expr.field == "n"
    assert rd.count_expr.var.typ is IncrCmd
    assert isinstance(rd.addr_expr, FieldRef)
    assert rd.addr_expr.field == "addr"
    # target buffer
    assert rd.target_var is not None
    assert rd.target_var.name == "buf"


def test_mm_array_write_stmt_fields():
    from examples.increment.incr import Uint32Field
    comp, stmts = _extract()
    wr = next(s for s in stmts if isinstance(s, MMArrayWriteStmt))

    assert wr.port is comp.m_mem
    assert wr.source_expr.name == "buf"   # transform is in-place; same buffer
    assert wr.elem_type is Uint32Field
    assert isinstance(wr.addr_expr, FieldRef)
    assert wr.addr_expr.field == "addr"
    assert isinstance(wr.count_expr, FieldRef)
    assert wr.count_expr.field == "n"


def test_stream_get_precedes_mm_access():
    comp, stmts = _extract()
    assert isinstance(stmts[0], StreamGetStmt)
    assert stmts[0].outputs[0].name == "cmd"


def test_transform_and_respond_are_function_stmts():
    comp, stmts = _extract()
    fns = [s for s in stmts if isinstance(s, FunctionStmt)]
    names = {s.method.__name__ for s in fns}
    assert names == {"transform", "respond"}
