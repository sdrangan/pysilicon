"""Tests for the C++ codegen pass (pysilicon/build/hwgen.py)."""
from __future__ import annotations

from enum import IntEnum

import pytest

from pysilicon.build.hwgen import CodegenCtx, to_cpp
from pysilicon.hw.hw_component import HwComponent
from pysilicon.hw.hwstmt import (
    CaseStmt,
    ContinueStmt,
    HwVar,
    ReturnStmt,
    SeqStmt,
    WhileStmt,
)
from pysilicon.simulation.simulation import Simulation


class DemoCmdType(IntEnum):
    DATA = 0
    END  = 1


class DemoError(IntEnum):
    OK         = 0
    BAD_INPUT  = 1


def _ctx() -> CodegenCtx:
    comp = HwComponent(name='c', sim=Simulation())
    return CodegenCtx(comp=comp)


# ---------------------------------------------------------------------------
# Phase 1: control-flow statements
# ---------------------------------------------------------------------------

def test_return_no_value():
    assert to_cpp(ReturnStmt(value=None), _ctx()) == "    return;"


def test_continue_stmt():
    assert to_cpp(ContinueStmt(), _ctx()) == "    continue;"


def test_while_with_return_body():
    stmt = WhileStmt(body=SeqStmt(stmts=[ReturnStmt(value=None)]))
    expected = (
        "    while (true) {\n"
        "        return;\n"
        "    }"
    )
    assert to_cpp(stmt, _ctx()) == expected


def test_case_stmt_bare_var():
    stmt = CaseStmt(
        var=HwVar(name='err', typ=None),
        field=None,
        value=DemoError.OK,
        op='!=',
        if_true=SeqStmt(stmts=[ReturnStmt(value=None)]),
    )
    expected = (
        "    if (err != DemoError::OK) {\n"
        "        return;\n"
        "    }"
    )
    assert to_cpp(stmt, _ctx()) == expected


def test_case_stmt_field_access():
    stmt = CaseStmt(
        var=HwVar(name='cmd', typ=None),
        field='cmd_type',
        value=DemoCmdType.END,
        op='==',
        if_true=SeqStmt(stmts=[ReturnStmt(value=None)]),
    )
    expected = (
        "    if (cmd.cmd_type == DemoCmdType::END) {\n"
        "        return;\n"
        "    }"
    )
    assert to_cpp(stmt, _ctx()) == expected


def test_seq_stmt_joins_with_newlines():
    stmt = SeqStmt(stmts=[ReturnStmt(value=None), ContinueStmt()])
    assert to_cpp(stmt, _ctx()) == "    return;\n    continue;"


def test_case_stmt_with_else():
    stmt = CaseStmt(
        var=HwVar(name='err', typ=None),
        field=None,
        value=DemoError.OK,
        op='==',
        if_true=SeqStmt(stmts=[ContinueStmt()]),
        if_false=SeqStmt(stmts=[ReturnStmt(value=None)]),
    )
    expected = (
        "    if (err == DemoError::OK) {\n"
        "        continue;\n"
        "    } else {\n"
        "        return;\n"
        "    }"
    )
    assert to_cpp(stmt, _ctx()) == expected


def test_unhandled_stmt_raises_not_implemented():
    class _Bogus:
        pass

    with pytest.raises(NotImplementedError):
        to_cpp(_Bogus(), _ctx())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Phase 2: Stream statements
# ---------------------------------------------------------------------------

class _FakeSchema:
    @classmethod
    def cpp_class_name(cls) -> str:
        return "DemoCmdHdr"


class _FakeEndpoint:
    """Stand-in object used as the bound `self` of a stream method."""


class _FakeBoundMethod:
    """Carries ``__self__`` so ``_endpoint_name`` can locate the endpoint."""
    def __init__(self, endpoint):
        self.__self__ = endpoint


def _comp_with_endpoints(**endpoints):
    """Create a HwComponent and stash endpoints as attributes for vars() lookup."""
    comp = HwComponent(name='c', sim=Simulation())
    for name, ep in endpoints.items():
        setattr(comp, name, ep)
    return comp


def test_stream_get_emits_decl_and_read():
    from pysilicon.hw.interface import StreamGetStmt
    s_in = _FakeEndpoint()
    comp = _comp_with_endpoints(s_in=s_in)
    ctx = CodegenCtx(comp=comp)
    stmt = StreamGetStmt(
        method=_FakeBoundMethod(s_in),
        inputs=[_FakeSchema],
        outputs=[HwVar(name='cmd', typ=_FakeSchema)],
    )
    expected = (
        "    DemoCmdHdr cmd;\n"
        "    cmd.read_axi4_stream<WORD_BW>(s_in);"
    )
    assert to_cpp(stmt, ctx) == expected


def test_stream_write_emits_call():
    from pysilicon.hw.interface import StreamWriteStmt
    m_out = _FakeEndpoint()
    comp = _comp_with_endpoints(m_out=m_out)
    ctx = CodegenCtx(comp=comp)
    stmt = StreamWriteStmt(
        method=_FakeBoundMethod(m_out),
        inputs=[HwVar(name='resp', typ=None)],
        outputs=[],
    )
    assert to_cpp(stmt, ctx) == "    resp.write_axi4_stream<WORD_BW>(m_out, true);"


def test_stream_drain_emits_flush():
    from pysilicon.hw.interface import StreamDrainStmt
    s_in = _FakeEndpoint()
    comp = _comp_with_endpoints(s_in=s_in)
    ctx = CodegenCtx(comp=comp)
    stmt = StreamDrainStmt(
        method=_FakeBoundMethod(s_in),
        inputs=[],
        outputs=[],
    )
    assert to_cpp(stmt, ctx) == (
        "    streamutils::flush_axi4_stream_to_tlast<WORD_BW>(s_in);"
    )


# ---------------------------------------------------------------------------
# Phase 3: Regmap statements
# ---------------------------------------------------------------------------

def test_regmap_set_literal():
    from pysilicon.hw.regmap import RegMapSetStmt
    stmt = RegMapSetStmt(
        method=None,
        inputs=['halted', 1],
        outputs=[],
    )
    assert to_cpp(stmt, _ctx()) == "    halted = 1;"


def test_regmap_set_hwvar():
    from pysilicon.hw.regmap import RegMapSetStmt
    stmt = RegMapSetStmt(
        method=None,
        inputs=['error', HwVar(name='err', typ=DemoError)],
        outputs=[],
    )
    assert to_cpp(stmt, _ctx()) == "    error = err;"


def test_regmap_set_field_ref():
    from pysilicon.hw.hwstmt import FieldRef
    from pysilicon.hw.regmap import RegMapSetStmt
    stmt = RegMapSetStmt(
        method=None,
        inputs=['tx_id', FieldRef(var=HwVar(name='cmd', typ=None), field='tx_id')],
        outputs=[],
    )
    assert to_cpp(stmt, _ctx()) == "    tx_id = cmd.tx_id;"


def test_regmap_get_with_typed_output():
    from pysilicon.hw.regmap import RegMapGetStmt

    class _CoeffArray:
        @classmethod
        def cpp_class_name(cls):
            return "CoeffArray"

    stmt = RegMapGetStmt(
        method=None,
        inputs=['coeffs'],
        outputs=[HwVar(name='coeffs', typ=_CoeffArray)],
    )
    assert to_cpp(stmt, _ctx()) == "    CoeffArray coeffs = coeffs;"


def test_regmap_get_with_untyped_output_falls_back_to_auto():
    from pysilicon.hw.regmap import RegMapGetStmt
    stmt = RegMapGetStmt(
        method=None,
        inputs=['coeffs'],
        outputs=[HwVar(name='coeffs', typ=None)],
    )
    assert to_cpp(stmt, _ctx()) == "    auto coeffs = coeffs;"


# ---------------------------------------------------------------------------
# Phase 4: FunctionStmt call sites
# ---------------------------------------------------------------------------

class _FakeMethod:
    def __init__(self, name):
        self.__name__ = name


def test_function_stmt_with_typed_output():
    from pysilicon.hw.hwstmt import FunctionStmt
    stmt = FunctionStmt(
        method=_FakeMethod('process'),
        inputs=[HwVar(name='cmd', typ=None)],
        outputs=[HwVar(name='err', typ=DemoError)],
    )
    assert to_cpp(stmt, _ctx()) == "    DemoError err = process(cmd);"


def test_function_stmt_no_outputs():
    from pysilicon.hw.hwstmt import FunctionStmt
    stmt = FunctionStmt(
        method=_FakeMethod('process'),
        inputs=[HwVar(name='cmd', typ=None)],
        outputs=[],
    )
    assert to_cpp(stmt, _ctx()) == "    process(cmd);"


def test_function_stmt_with_endpoint_arg():
    from pysilicon.hw.hwstmt import FunctionStmt
    from pysilicon.hw.interface import StreamIFMaster, StreamIFSlave
    sim = Simulation()
    s_in = StreamIFSlave(name='s_in_ep', sim=sim, bitwidth=32)
    m_out = StreamIFMaster(name='m_out_ep', sim=sim, bitwidth=32)
    comp = HwComponent(name='c', sim=sim)
    comp.s_in = s_in
    comp.m_out = m_out
    ctx = CodegenCtx(comp=comp)
    stmt = FunctionStmt(
        method=_FakeMethod('process'),
        inputs=[HwVar(name='cmd', typ=None), s_in, m_out],
        outputs=[],
    )
    assert to_cpp(stmt, ctx) == "    process(cmd, s_in, m_out);"


def test_function_stmt_multi_output_raises():
    from pysilicon.hw.hwstmt import FunctionStmt
    stmt = FunctionStmt(
        method=_FakeMethod('split'),
        inputs=[HwVar(name='x', typ=None)],
        outputs=[
            HwVar(name='a', typ=None),
            HwVar(name='b', typ=None),
        ],
    )
    with pytest.raises(NotImplementedError, match="Multi-output"):
        to_cpp(stmt, _ctx())


# ---------------------------------------------------------------------------
# Phase 5: kernel_body_to_cpp end-to-end
# ---------------------------------------------------------------------------

def test_kernel_body_to_cpp_demo_component_contains_expected_substrings():
    from pysilicon.build.hwgen import kernel_body_to_cpp
    # Reuse the DemoComponent fixture from tests/hw/test_resolve.py.
    from tests.hw.test_resolve import DemoComponent

    comp = DemoComponent(name="demo", sim=Simulation())
    body = kernel_body_to_cpp(comp)

    expected = [
        "while (true)",
        "DemoCmdHdr cmd;",
        "cmd.read_axi4_stream<WORD_BW>(s_in);",
        "if (cmd.cmd_type == DemoCmdType::END)",
        "return;",
        "DemoError err = process(cmd",
        "if (err != DemoError::OK)",
        "error = err;",
        "tx_id = cmd.tx_id;",
        "halted = 1;",
    ]
    for sub in expected:
        assert sub in body, f"Missing substring: {sub!r}\n--- body ---\n{body}"


def test_endpoint_name_not_found_raises():
    from pysilicon.hw.interface import StreamGetStmt
    rogue = _FakeEndpoint()
    comp = _comp_with_endpoints()  # no endpoint set
    ctx = CodegenCtx(comp=comp)
    stmt = StreamGetStmt(
        method=_FakeBoundMethod(rogue),
        inputs=[_FakeSchema],
        outputs=[HwVar(name='x', typ=_FakeSchema)],
    )
    with pytest.raises(RuntimeError, match="not found"):
        to_cpp(stmt, ctx)
