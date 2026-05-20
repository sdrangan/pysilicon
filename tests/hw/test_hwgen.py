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
from pysilicon.hw.interface import StreamIFSlave as _StreamIFSlave
from pysilicon.hw.synth import synthesizable as _synthesizable
from pysilicon.simulation.simobj import ProcessGen as _ProcessGen
from pysilicon.simulation.simulation import Simulation
from tests.hw.test_resolve import DemoCmdHdr as _DemoCmdHdr
from tests.hw.test_resolve import DemoError as _DemoError


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
    # _ctx() uses bare HwComponent → resolved_namespace == "hw".
    assert to_cpp(stmt, _ctx()) == "    DemoError err = hw::process(cmd);"


def test_function_stmt_no_outputs():
    from pysilicon.hw.hwstmt import FunctionStmt
    stmt = FunctionStmt(
        method=_FakeMethod('process'),
        inputs=[HwVar(name='cmd', typ=None)],
        outputs=[],
    )
    assert to_cpp(stmt, _ctx()) == "    hw::process(cmd);"


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
    assert to_cpp(stmt, ctx) == "    hw::process(cmd, s_in, m_out);"


def test_function_stmt_opt_out_no_qualifier():
    from typing import ClassVar
    from pysilicon.hw.hwstmt import FunctionStmt

    class _NoNs(HwComponent):
        cpp_namespace: ClassVar[str | None] = ""

    sim = Simulation()
    comp = _NoNs(name='c', sim=sim)
    ctx = CodegenCtx(comp=comp)
    stmt = FunctionStmt(
        method=_FakeMethod('process'),
        inputs=[HwVar(name='cmd', typ=None)],
        outputs=[],
    )
    assert to_cpp(stmt, ctx) == "    process(cmd);"


def test_function_stmt_custom_namespace():
    from typing import ClassVar
    from pysilicon.hw.hwstmt import FunctionStmt

    class _CustomNs(HwComponent):
        cpp_namespace: ClassVar[str | None] = "custom"

    sim = Simulation()
    comp = _CustomNs(name='c', sim=sim)
    ctx = CodegenCtx(comp=comp)
    stmt = FunctionStmt(
        method=_FakeMethod('process'),
        inputs=[HwVar(name='cmd', typ=None)],
        outputs=[],
    )
    assert to_cpp(stmt, ctx) == "    custom::process(cmd);"


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

# ---------------------------------------------------------------------------
# Kernel-files Phase 1: cpp_type + cpp_kernel_name
# ---------------------------------------------------------------------------

def test_cpp_type_intfield_unsigned():
    from pysilicon.build.hwgen import cpp_type
    from pysilicon.hw.dataschema import IntField
    assert cpp_type(IntField.specialize(bitwidth=16, signed=False)) == "ap_uint<16>"


def test_cpp_type_intfield_signed():
    from pysilicon.build.hwgen import cpp_type
    from pysilicon.hw.dataschema import IntField
    assert cpp_type(IntField.specialize(bitwidth=8, signed=True)) == "ap_int<8>"


def test_cpp_type_floatfield_32():
    from pysilicon.build.hwgen import cpp_type
    from pysilicon.hw.dataschema import FloatField
    assert cpp_type(FloatField.specialize(bitwidth=32)) == "float"


def test_cpp_type_floatfield_64():
    from pysilicon.build.hwgen import cpp_type
    from pysilicon.hw.dataschema import FloatField
    assert cpp_type(FloatField.specialize(bitwidth=64)) == "double"


def test_cpp_type_enumfield():
    from pysilicon.build.hwgen import cpp_type
    from pysilicon.hw.dataschema import EnumField
    assert cpp_type(EnumField.specialize(enum_type=DemoError)) == "ap_uint<8>"


def test_cpp_type_intenum_direct():
    """IntEnum subclasses (e.g. resolved FunctionStmt return types) map to ap_uint<8>."""
    from pysilicon.build.hwgen import cpp_type
    assert cpp_type(DemoError) == "ap_uint<8>"


def test_cpp_type_dataschema_subclass_uses_cpp_class_name():
    from pysilicon.build.hwgen import cpp_type
    from pysilicon.hw.dataschema import DataList, IntField

    class MyMsg(DataList):
        elements = {
            "x": {"schema": IntField.specialize(bitwidth=8, signed=False)},
        }

    assert cpp_type(MyMsg) == "MyMsg"


def test_cpp_type_schemaarray_placeholder():
    from pysilicon.build.hwgen import cpp_type
    from pysilicon.hw.dataschema import FloatField
    Float32 = FloatField.specialize(bitwidth=32)
    out = cpp_type(('SchemaArray', Float32))
    assert "float[MAX_N]" in out
    assert "TODO" in out


def test_cpp_type_none_raises():
    from pysilicon.build.hwgen import cpp_type
    with pytest.raises(RuntimeError):
        cpp_type(None)


def test_cpp_kernel_name_demo():
    from pysilicon.build.hwgen import cpp_kernel_name
    from tests.hw.test_resolve import DemoComponent
    assert cpp_kernel_name(DemoComponent) == "demo"


def test_cpp_kernel_name_poly_accel():
    from pysilicon.build.hwgen import cpp_kernel_name
    import sys
    from pathlib import Path
    POLY_DIR = Path(__file__).resolve().parents[2] / "examples" / "poly"
    if str(POLY_DIR) not in sys.path:
        sys.path.insert(0, str(POLY_DIR))
    from poly import PolyAccelComponent
    assert cpp_kernel_name(PolyAccelComponent) == "poly_accel"


def test_cpp_kernel_name_override():
    from pysilicon.build.hwgen import cpp_kernel_name
    from typing import ClassVar

    class _Overridden(HwComponent):
        cpp_kernel_name: ClassVar[str | None] = "my_custom_name"

    assert cpp_kernel_name(_Overridden) == "my_custom_name"


# ---------------------------------------------------------------------------
# Namespace Phase 1: resolved_namespace
# ---------------------------------------------------------------------------

def test_resolved_namespace_default_uses_kernel_name():
    from pysilicon.build.hwgen import resolved_namespace
    from tests.hw.test_resolve import DemoComponent
    assert resolved_namespace(DemoComponent) == "demo"


def test_resolved_namespace_explicit_string():
    from typing import ClassVar
    from pysilicon.build.hwgen import resolved_namespace

    class _NsCustom(HwComponent):
        cpp_namespace: ClassVar[str | None] = "custom"

    assert resolved_namespace(_NsCustom) == "custom"


def test_resolved_namespace_empty_opts_out():
    from typing import ClassVar
    from pysilicon.build.hwgen import resolved_namespace

    class _NsOptOut(HwComponent):
        cpp_namespace: ClassVar[str | None] = ""

    assert resolved_namespace(_NsOptOut) is None


def test_resolved_namespace_explicit_none_is_auto():
    from typing import ClassVar
    from pysilicon.build.hwgen import resolved_namespace

    class _NsAuto(HwComponent):
        cpp_namespace: ClassVar[str | None] = None

    # Auto-derives from cpp_kernel_name; default of HwComponent name is "hw".
    assert resolved_namespace(_NsAuto) == cpp_kernel_name_for(_NsAuto)


def cpp_kernel_name_for(cls):
    from pysilicon.build.hwgen import cpp_kernel_name
    return cpp_kernel_name(cls)


def test_resolved_namespace_independent_from_kernel_name_override():
    from typing import ClassVar
    from pysilicon.build.hwgen import cpp_kernel_name, resolved_namespace

    class _Both(HwComponent):
        cpp_namespace: ClassVar[str | None] = "alpha"
        cpp_kernel_name: ClassVar[str | None] = "beta"

    assert resolved_namespace(_Both) == "alpha"
    assert cpp_kernel_name(_Both) == "beta"


# ---------------------------------------------------------------------------
# Kernel-files Phase 2: hook_signature derivation
# ---------------------------------------------------------------------------
# Hook fixtures defined at module level so typing.get_type_hints() can
# resolve their annotations via the module's __globals__ (annotations are
# stringified under `from __future__ import annotations`).

@_synthesizable
def _hook_evaluate(self, cmd: _DemoCmdHdr) -> _ProcessGen[_DemoError]:
    yield None
    return _DemoError.OK


@_synthesizable
def _hook_with_stream(
    self,
    cmd: _DemoCmdHdr,
    s_in: _StreamIFSlave,
) -> _ProcessGen[None]:
    yield None


@_synthesizable
def _hook_bad(self, x):  # no annotation
    return x


@_synthesizable
def _hook_no_return(self, cmd: _DemoCmdHdr):
    pass


def test_hook_signature_str_simple():
    from pysilicon.build.hwgen import hook_signature_str
    assert hook_signature_str(_hook_evaluate) == (
        "ap_uint<8> _hook_evaluate(DemoCmdHdr cmd)"
    )


def test_hook_signature_str_with_stream_endpoint():
    from pysilicon.build.hwgen import hook_signature_str
    sig = hook_signature_str(_hook_with_stream)
    assert "DemoCmdHdr cmd" in sig
    assert "hls::stream<streamutils::axi4s_word<WORD_BW>>& s_in" in sig
    assert sig.startswith("void _hook_with_stream(")


def test_hook_signature_missing_annotation_raises():
    from pysilicon.build.hwgen import hook_signature
    with pytest.raises(RuntimeError, match="'x'"):
        hook_signature(_hook_bad)


def test_hook_signature_no_return_annotation_is_void():
    from pysilicon.build.hwgen import hook_signature
    ret, _args = hook_signature(_hook_no_return)
    assert ret == "void"


# ---------------------------------------------------------------------------
# Kernel-files Phase 3: kernel_signature
# ---------------------------------------------------------------------------

def test_kernel_signature_demo_component():
    from pysilicon.build.hwgen import kernel_signature
    from tests.hw.test_resolve import DemoComponent

    comp = DemoComponent(name="demo", sim=Simulation())
    sig = kernel_signature(comp)
    expected_substrings = [
        "void demo(",
        "hls::stream<streamutils::axi4s_word<WORD_BW>>& s_in",
        "hls::stream<streamutils::axi4s_word<WORD_BW>>& m_out",
        "ap_uint<1>& ap_start",
        "ap_uint<1>& halted",
        "ap_uint<8>& error",
        "ap_uint<16>& tx_id",
        "#pragma HLS INTERFACE axis port=s_in",
        "#pragma HLS INTERFACE axis port=m_out",
        "#pragma HLS INTERFACE s_axilite port=halted",
        "#pragma HLS INTERFACE s_axilite port=return",
    ]
    for sub in expected_substrings:
        assert sub in sig, f"Missing substring: {sub!r}\n--- sig ---\n{sig}"


# ---------------------------------------------------------------------------
# Kernel-files Phase 4: header_to_cpp
# ---------------------------------------------------------------------------

def test_header_to_cpp_demo_component_substrings():
    from pysilicon.build.hwgen import header_to_cpp
    from tests.hw.test_resolve import DemoComponent

    comp = DemoComponent(name="demo", sim=Simulation())
    hpp = header_to_cpp(comp)

    for sub in [
        "#pragma once",
        '#include "include/streamutils_hls.h"',
        '#include "include/democmdhdr.h"',
        "void demo(",
        ");",  # forward decl terminator
        "ap_uint<8> process(DemoCmdHdr cmd);",
    ]:
        assert sub in hpp, f"Missing substring: {sub!r}\n--- hpp ---\n{hpp}"


def test_header_to_cpp_forward_decl_has_no_pragmas():
    from pysilicon.build.hwgen import header_to_cpp
    from tests.hw.test_resolve import DemoComponent

    comp = DemoComponent(name="demo", sim=Simulation())
    hpp = header_to_cpp(comp)
    # The .hpp must NOT include any HLS pragmas — those live in the .cpp.
    assert "#pragma HLS" not in hpp


# ---------------------------------------------------------------------------
# Kernel-files Phase 5: kernel_to_cpp + impl_stub_to_cpp + driver
# ---------------------------------------------------------------------------

def test_kernel_to_cpp_substrings():
    from pysilicon.build.hwgen import kernel_to_cpp
    from tests.hw.test_resolve import DemoComponent

    comp = DemoComponent(name="demo", sim=Simulation())
    cpp = kernel_to_cpp(comp)

    for sub in [
        '#include "demo.hpp"',
        "void demo(",
        "#pragma HLS INTERFACE axis port=s_in",
        "while (true)",
    ]:
        assert sub in cpp, f"Missing substring: {sub!r}\n--- cpp ---\n{cpp}"
    # Closing brace at end of file.
    assert cpp.rstrip().endswith("}")


def test_impl_stub_to_cpp_substrings():
    from pysilicon.build.hwcodegen import extract_kernel
    from pysilicon.build.hwgen import _collect_hooks, impl_stub_to_cpp
    from tests.hw.test_resolve import DemoComponent

    comp = DemoComponent(name="demo", sim=Simulation())
    tree = extract_kernel(comp)
    hooks = _collect_hooks(tree)
    assert len(hooks) == 1
    process = hooks[0]

    stub = impl_stub_to_cpp(comp, process)
    for sub in [
        '#include "demo.hpp"',
        "ap_uint<8> process(DemoCmdHdr cmd) {",
        "// TODO: implement process",
        "return ap_uint<8>(0);",
    ]:
        assert sub in stub, f"Missing substring: {sub!r}\n--- stub ---\n{stub}"
    assert stub.rstrip().endswith("}")


def test_kernel_files_to_str_keys():
    from pysilicon.build.hwgen import kernel_files_to_str
    from tests.hw.test_resolve import DemoComponent

    comp = DemoComponent(name="demo", sim=Simulation())
    files = kernel_files_to_str(comp)
    assert set(files.keys()) == {
        "demo.hpp",
        "demo.cpp",
        "demo_process_impl.cpp",
    }


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
        "DemoError err = demo::process(cmd",
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
