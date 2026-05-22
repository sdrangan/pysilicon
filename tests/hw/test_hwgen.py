"""Tests for the C++ codegen pass (pysilicon/build/hwgen.py)."""
from __future__ import annotations

from dataclasses import dataclass as _dataclass
from enum import IntEnum
from typing import ClassVar

import pytest

from pysilicon.build.hwgen import CodegenCtx, to_cpp
from pysilicon.hw.hw_component import HwComponent, HwParam
from pysilicon.hw.hwstmt import (
    CaseStmt,
    ContinueStmt,
    HwVar,
    ReturnStmt,
    SeqStmt,
    WhileStmt,
)
from pysilicon.hw.interface import StreamIFMaster as _StreamIFMaster
from pysilicon.hw.interface import StreamIFSlave as _StreamIFSlave
from pysilicon.hw.regmap import (
    Bit as _Bit,
    RegAccess as _RegAccess,
    RegField as _RegField,
    VitisRegMap as _VitisRegMap,
    VitisRegMapMMIFSlave as _VitisRegMapMMIFSlave,
)
from pysilicon.hw.synth import synthesizable as _synthesizable
from pysilicon.simulation.simobj import ProcessGen as _ProcessGen
from pysilicon.simulation.simulation import Simulation
from tests.hw.test_resolve import DemoCmdHdr
from tests.hw.test_resolve import DemoCmdHdr as _DemoCmdHdr
from tests.hw.test_resolve import DemoError as _DemoError
from tests.hw.test_resolve import DemoErrorField


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
    bitwidth: int = 32


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
        "    cmd.read_axi4_stream<32>(s_in);"
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
    assert to_cpp(stmt, ctx) == "    resp.write_axi4_stream<32>(m_out, true);"


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
        "    streamutils::flush_axi4_stream_to_tlast<32>(s_in);"
    )


class _FakeParamEndpoint:
    """Stand-in endpoint whose bitwidth is a HwParamValue."""
    def __init__(self, param_name: str, value: int = 32) -> None:
        from pysilicon.hw.hw_component import HwParamValue
        self.bitwidth = HwParamValue(value, param_name)


def test_stream_get_uses_template_name_for_hwparam_endpoint():
    from pysilicon.hw.interface import StreamGetStmt
    s_in = _FakeParamEndpoint(param_name="in_bw")
    comp = _comp_with_endpoints(s_in=s_in)
    ctx = CodegenCtx(comp=comp)
    stmt = StreamGetStmt(
        method=_FakeBoundMethod(s_in),
        inputs=[_FakeSchema],
        outputs=[HwVar(name='cmd', typ=_FakeSchema)],
    )
    out = to_cpp(stmt, ctx)
    assert "read_axi4_stream<in_bw>(s_in)" in out
    assert "<32>" not in out


def test_stream_write_uses_template_name_for_hwparam_endpoint():
    from pysilicon.hw.interface import StreamWriteStmt
    m_out = _FakeParamEndpoint(param_name="out_bw")
    comp = _comp_with_endpoints(m_out=m_out)
    ctx = CodegenCtx(comp=comp)
    stmt = StreamWriteStmt(
        method=_FakeBoundMethod(m_out),
        inputs=[HwVar(name='resp', typ=None)],
        outputs=[],
    )
    out = to_cpp(stmt, ctx)
    assert "write_axi4_stream<out_bw>(m_out, true)" in out


def test_stream_drain_uses_template_name_for_hwparam_endpoint():
    from pysilicon.hw.interface import StreamDrainStmt
    s_in = _FakeParamEndpoint(param_name="in_bw")
    comp = _comp_with_endpoints(s_in=s_in)
    ctx = CodegenCtx(comp=comp)
    stmt = StreamDrainStmt(
        method=_FakeBoundMethod(s_in),
        inputs=[],
        outputs=[],
    )
    out = to_cpp(stmt, ctx)
    assert "flush_axi4_stream_to_tlast<in_bw>(s_in)" in out


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
    from pysilicon.build.hwgen import resolved_namespace

    class _NsCustom(HwComponent):
        cpp_namespace: ClassVar[str | None] = "custom"

    assert resolved_namespace(_NsCustom) == "custom"


def test_resolved_namespace_empty_opts_out():
    from pysilicon.build.hwgen import resolved_namespace

    class _NsOptOut(HwComponent):
        cpp_namespace: ClassVar[str | None] = ""

    assert resolved_namespace(_NsOptOut) is None


def test_resolved_namespace_explicit_none_is_auto():
    from pysilicon.build.hwgen import resolved_namespace

    class _NsAuto(HwComponent):
        cpp_namespace: ClassVar[str | None] = None

    # Auto-derives from cpp_kernel_name; default of HwComponent name is "hw".
    assert resolved_namespace(_NsAuto) == cpp_kernel_name_for(_NsAuto)


def cpp_kernel_name_for(cls):
    from pysilicon.build.hwgen import cpp_kernel_name
    return cpp_kernel_name(cls)


def test_resolved_namespace_independent_from_kernel_name_override():
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


@_dataclass
class _TmplComp(HwComponent):
    """Component whose synthesizable hook takes a stream arg (triggers templating)."""

    cpp_kernel_name: ClassVar[str | None] = "tcomp"
    in_bw: HwParam[int] = 32

    def __post_init__(self) -> None:
        super().__post_init__()
        self.s_in = _StreamIFSlave(
            name=f'{self.name}_s_in', sim=self.sim, bitwidth=self.in_bw,
        )
        self.m_out = _StreamIFMaster(
            name=f'{self.name}_m_out', sim=self.sim, bitwidth=self.in_bw,
        )
        self.regmap = _VitisRegMap({
            "halted": _RegField(_Bit, _RegAccess.R),
            "error": _RegField(DemoErrorField, _RegAccess.R),
        })
        self.s_lite = _VitisRegMapMMIFSlave(
            name=f'{self.name}_s_lite', sim=self.sim, bitwidth=32,
            regmap=self.regmap, on_start=self.on_start,
        )
        for ep in (self.s_in, self.m_out, self.s_lite):
            self.add_endpoint(ep)

    def on_start(self) -> _ProcessGen[None]:
        while True:
            cmd: DemoCmdHdr = yield from self.s_in.get(DemoCmdHdr)
            if cmd.cmd_type == 0:
                return
            yield from self.process(cmd, self.s_in)
            return

    @_synthesizable
    def process(
        self,
        cmd: DemoCmdHdr,
        s_in: _StreamIFSlave,
    ) -> _ProcessGen[DemoError]:
        yield self.env.timeout(0)
        return DemoError.OK


@_synthesizable
def _hook_two_streams(
    self,
    cmd: _DemoCmdHdr,
    s_in: _StreamIFSlave,
    m_out: _StreamIFMaster,
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
        "template <int in_bw, int out_bw>",
        "void demo(",
        "hls::stream<streamutils::axi4s_word<in_bw>>& s_in",
        "hls::stream<streamutils::axi4s_word<out_bw>>& m_out",
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
    # ap_start is_vitis_auto -> generated by `port=return bundle=control`, NOT
    # by an explicit argument. Verify it does not appear in the signature.
    assert "ap_start" not in sig, f"ap_start should not appear in signature:\n{sig}"
    # Literal '32' should NOT appear in stream type expressions any more.
    assert "axi4s_word<32>" not in sig


def test_kernel_signature_raises_on_name_collision():
    """A HwParam that shares a name with a regmap field triggers SynthesisError."""
    from dataclasses import dataclass

    from pysilicon.build.hwcodegen import SynthesisError
    from pysilicon.build.hwgen import kernel_signature
    from pysilicon.hw.interface import StreamIFSlave
    from pysilicon.hw.regmap import (
        Bit, RegAccess, RegField, VitisRegMap, VitisRegMapMMIFSlave,
    )

    @dataclass
    class _CollidingComp(HwComponent):
        # 'halted' clashes with the regmap field name below.
        halted: HwParam[int] = 32

        def __post_init__(self) -> None:
            super().__post_init__()
            self.s_in = StreamIFSlave(
                name=f'{self.name}_s_in', sim=self.sim, bitwidth=32,
            )
            self.regmap = VitisRegMap({
                "halted": RegField(Bit, RegAccess.R),
            })
            self.s_lite = VitisRegMapMMIFSlave(
                name=f'{self.name}_s_lite', sim=self.sim, bitwidth=32,
                regmap=self.regmap, on_start=None,
            )
            for ep in (self.s_in, self.s_lite):
                self.add_endpoint(ep)

    comp = _CollidingComp(name="bad", sim=Simulation())
    with pytest.raises(SynthesisError, match="halted"):
        kernel_signature(comp)


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
        "namespace demo {",
        "ap_uint<8> process(DemoCmdHdr cmd);",
    ]:
        assert sub in hpp, f"Missing substring: {sub!r}\n--- hpp ---\n{hpp}"

    # Kernel decl must be outside (before) the namespace block;
    # the hook decl must be inside.
    kernel_idx = hpp.index("void demo(")
    ns_idx = hpp.index("namespace demo {")
    process_idx = hpp.index("ap_uint<8> process(DemoCmdHdr cmd);")
    assert kernel_idx < ns_idx < process_idx


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
        "namespace demo {",
        "ap_uint<8> process(DemoCmdHdr cmd) {",
        "// TODO: implement process",
        "return ap_uint<8>(0);",
    ]:
        assert sub in stub, f"Missing substring: {sub!r}\n--- stub ---\n{stub}"
    assert stub.rstrip().endswith("}")


def test_header_to_cpp_opt_out_emits_no_namespace_block():
    from typing import ClassVar
    from dataclasses import dataclass
    from pysilicon.build.hwgen import header_to_cpp
    from tests.hw.test_resolve import DemoComponent

    @dataclass
    class _OptOutDemo(DemoComponent):
        cpp_namespace: ClassVar[str | None] = ""

    comp = _OptOutDemo(name="optout", sim=Simulation())
    hpp = header_to_cpp(comp)
    assert "namespace " not in hpp
    # The hook decl still appears, just not wrapped.
    assert "ap_uint<8> process(DemoCmdHdr cmd);" in hpp


def test_impl_stub_to_cpp_opt_out_emits_no_namespace_block():
    from typing import ClassVar
    from dataclasses import dataclass
    from pysilicon.build.hwcodegen import extract_kernel
    from pysilicon.build.hwgen import _collect_hooks, impl_stub_to_cpp
    from tests.hw.test_resolve import DemoComponent

    @dataclass
    class _OptOutDemo(DemoComponent):
        cpp_namespace: ClassVar[str | None] = ""

    comp = _OptOutDemo(name="optout", sim=Simulation())
    tree = extract_kernel(comp)
    process = _collect_hooks(tree)[0]
    stub = impl_stub_to_cpp(comp, process)
    assert "namespace " not in stub
    assert "ap_uint<8> process(DemoCmdHdr cmd) {" in stub


def test_header_to_cpp_custom_namespace():
    from typing import ClassVar
    from dataclasses import dataclass
    from pysilicon.build.hwgen import header_to_cpp
    from tests.hw.test_resolve import DemoComponent

    @dataclass
    class _CustomNsDemo(DemoComponent):
        cpp_namespace: ClassVar[str | None] = "custom"

    comp = _CustomNsDemo(name="cn", sim=Simulation())
    hpp = header_to_cpp(comp)
    assert "namespace custom {" in hpp


def test_impl_stub_to_cpp_custom_namespace():
    from typing import ClassVar
    from dataclasses import dataclass
    from pysilicon.build.hwcodegen import extract_kernel
    from pysilicon.build.hwgen import _collect_hooks, impl_stub_to_cpp
    from tests.hw.test_resolve import DemoComponent

    @dataclass
    class _CustomNsDemo(DemoComponent):
        cpp_namespace: ClassVar[str | None] = "custom"

    comp = _CustomNsDemo(name="cn", sim=Simulation())
    tree = extract_kernel(comp)
    process = _collect_hooks(tree)[0]
    stub = impl_stub_to_cpp(comp, process)
    assert "namespace custom {" in stub


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
        # DemoComponent.s_in has bitwidth = HwParamValue(in_bw, 32) →
        # template arg is "in_bw", not the literal 32.
        "cmd.read_axi4_stream<in_bw>(s_in);",
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
    # Literal '32' should no longer appear in the stream template arg.
    assert "read_axi4_stream<32>" not in body


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


# ---------------------------------------------------------------------------
# Hook-template Phase 1: hook_template_params + single-call-site validation
# ---------------------------------------------------------------------------

def _stream_endpoint(name: str, bitwidth, slave: bool = True):
    """Build a real StreamIF endpoint with the given (possibly-HwParamValue) bitwidth."""
    from pysilicon.hw.interface import StreamIFMaster, StreamIFSlave
    cls = StreamIFSlave if slave else StreamIFMaster
    return cls(name=name, sim=Simulation(), bitwidth=bitwidth)


def test_hook_template_params_empty_inputs():
    from pysilicon.build.hwgen import hook_template_params
    from pysilicon.hw.hwstmt import FunctionStmt
    stmt = FunctionStmt(method=_FakeMethod("h"), inputs=[], outputs=[])
    assert hook_template_params(stmt) == []


def test_hook_template_params_hwvar_only_inputs():
    from pysilicon.build.hwgen import hook_template_params
    from pysilicon.hw.hwstmt import FunctionStmt
    stmt = FunctionStmt(
        method=_FakeMethod("h"),
        inputs=[HwVar(name="x", typ=None), HwVar(name="y", typ=None)],
        outputs=[],
    )
    assert hook_template_params(stmt) == []


def test_hook_template_params_single_param_endpoint():
    from pysilicon.build.hwgen import hook_template_params
    from pysilicon.hw.hw_component import HwParamValue
    from pysilicon.hw.hwstmt import FunctionStmt
    ep = _stream_endpoint("s_in_ep", HwParamValue(32, "in_bw"))
    stmt = FunctionStmt(
        method=_FakeMethod("h"),
        inputs=[HwVar(name="cmd", typ=None), ep],
        outputs=[],
    )
    assert hook_template_params(stmt) == ["in_bw"]


def test_hook_template_params_two_endpoints_same_param_dedupes():
    from pysilicon.build.hwgen import hook_template_params
    from pysilicon.hw.hw_component import HwParamValue
    from pysilicon.hw.hwstmt import FunctionStmt
    bw = HwParamValue(32, "in_bw")
    e1 = _stream_endpoint("a", bw)
    e2 = _stream_endpoint("b", bw, slave=False)
    stmt = FunctionStmt(method=_FakeMethod("h"), inputs=[e1, e2], outputs=[])
    assert hook_template_params(stmt) == ["in_bw"]


def test_hook_template_params_two_endpoints_different_params():
    from pysilicon.build.hwgen import hook_template_params
    from pysilicon.hw.hw_component import HwParamValue
    from pysilicon.hw.hwstmt import FunctionStmt
    e1 = _stream_endpoint("a", HwParamValue(32, "in_bw"))
    e2 = _stream_endpoint("b", HwParamValue(64, "out_bw"), slave=False)
    stmt = FunctionStmt(method=_FakeMethod("h"), inputs=[e1, e2], outputs=[])
    assert hook_template_params(stmt) == ["in_bw", "out_bw"]


def test_hook_template_params_raw_int_endpoint_skipped():
    from pysilicon.build.hwgen import hook_template_params
    from pysilicon.hw.hw_component import HwParamValue
    from pysilicon.hw.hwstmt import FunctionStmt
    raw = _stream_endpoint("raw", 32)
    param = _stream_endpoint("p", HwParamValue(32, "in_bw"))
    stmt = FunctionStmt(method=_FakeMethod("h"), inputs=[raw, param], outputs=[])
    assert hook_template_params(stmt) == ["in_bw"]


def test_validate_single_call_site_single_site_ok():
    from pysilicon.build.hwgen import _validate_single_call_site
    from pysilicon.hw.hw_component import HwParamValue
    from pysilicon.hw.hwstmt import FunctionStmt, SeqStmt
    method = _FakeMethod("h")
    ep = _stream_endpoint("a", HwParamValue(32, "in_bw"))
    tree = SeqStmt(stmts=[
        FunctionStmt(method=method, inputs=[ep], outputs=[]),
    ])
    _validate_single_call_site(tree, method, ["in_bw"])


def test_validate_single_call_site_consistent_ok():
    from pysilicon.build.hwgen import _validate_single_call_site
    from pysilicon.hw.hw_component import HwParamValue
    from pysilicon.hw.hwstmt import FunctionStmt, SeqStmt
    method = _FakeMethod("h")
    ep = _stream_endpoint("a", HwParamValue(32, "in_bw"))
    tree = SeqStmt(stmts=[
        FunctionStmt(method=method, inputs=[ep], outputs=[]),
        FunctionStmt(method=method, inputs=[ep], outputs=[]),
    ])
    _validate_single_call_site(tree, method, ["in_bw"])


# ---------------------------------------------------------------------------
# Hook-template Phase 2: hook_signature_str accepts template_params
# ---------------------------------------------------------------------------

def test_hook_signature_str_without_template_params_unchanged():
    """Default behaviour (no template_params) still emits symbolic WORD_BW."""
    from pysilicon.build.hwgen import hook_signature_str
    sig = hook_signature_str(_hook_with_stream)
    assert sig.startswith("void _hook_with_stream(")
    assert "axi4s_word<WORD_BW>" in sig
    assert "template <" not in sig


def test_hook_signature_str_with_one_template_param():
    from pysilicon.build.hwgen import hook_signature_str
    sig = hook_signature_str(_hook_with_stream, template_params=["in_bw"])
    assert sig.startswith("template <int in_bw>\n")
    assert "axi4s_word<in_bw>" in sig
    assert "axi4s_word<WORD_BW>" not in sig


def test_hook_signature_str_with_two_template_params():
    """Two stream args, two template params — substituted in order."""
    from pysilicon.build.hwgen import hook_signature_str
    sig = hook_signature_str(
        _hook_two_streams, template_params=["in_bw", "out_bw"],
    )
    assert sig.startswith("template <int in_bw, int out_bw>\n")
    assert "axi4s_word<in_bw>" in sig
    assert "axi4s_word<out_bw>" in sig
    assert "WORD_BW" not in sig


def test_validate_single_call_site_inconsistent_raises():
    from pysilicon.build.hwcodegen import SynthesisError
    from pysilicon.build.hwgen import _validate_single_call_site
    from pysilicon.hw.hw_component import HwParamValue
    from pysilicon.hw.hwstmt import FunctionStmt, SeqStmt
    method = _FakeMethod("h")
    e1 = _stream_endpoint("a", HwParamValue(32, "in_bw"))
    e2 = _stream_endpoint("b", HwParamValue(64, "out_bw"))
    tree = SeqStmt(stmts=[
        FunctionStmt(method=method, inputs=[e1], outputs=[]),
        FunctionStmt(method=method, inputs=[e2], outputs=[]),
    ])
    with pytest.raises(SynthesisError, match="inconsistent"):
        _validate_single_call_site(tree, method, ["in_bw"])


# ---------------------------------------------------------------------------
# Hook-template Phase 3: header_to_cpp emits templated decls + .tpp includes
# ---------------------------------------------------------------------------

def test_header_to_cpp_non_templated_hook_no_tpp_include():
    """DemoComponent's process hook (no stream args) keeps the existing shape."""
    from pysilicon.build.hwgen import header_to_cpp
    from tests.hw.test_resolve import DemoComponent

    comp = DemoComponent(name="demo", sim=Simulation())
    hpp = header_to_cpp(comp)
    assert "ap_uint<8> process(DemoCmdHdr cmd);" in hpp
    # No tpp include for a non-templated hook.
    assert "_impl.tpp" not in hpp
    # No template prefix either.
    assert "template <" not in hpp.split("namespace demo {")[1]


def test_header_to_cpp_templated_hook_emits_template_and_tpp_include():
    """A hook that takes a HwParam-driven stream endpoint gets templated."""
    from pysilicon.build.hwgen import header_to_cpp

    comp = _TmplComp(name="tcomp", sim=Simulation())
    hpp = header_to_cpp(comp)
    # Inside the namespace block: template prefix + templated arg type.
    assert "template <int in_bw>" in hpp
    assert "axi4s_word<in_bw>" in hpp
    # .tpp include at the bottom for this hook.
    assert '#include "tcomp_process_impl.tpp"' in hpp
    # The include must come AFTER the templated decl.
    assert hpp.index("template <int in_bw>") < hpp.index("tcomp_process_impl.tpp")


# ---------------------------------------------------------------------------
# Hook-template Phase 4: impl_stub_to_tpp + kernel_files_to_str routing
# ---------------------------------------------------------------------------

def test_impl_stub_to_tpp_substrings():
    from pysilicon.build.hwcodegen import extract_kernel
    from pysilicon.build.hwgen import (
        _collect_hooks_with_params,
        impl_stub_to_tpp,
    )

    comp = _TmplComp(name="tcomp", sim=Simulation())
    tree = extract_kernel(comp)
    hooks = _collect_hooks_with_params(tree)
    assert len(hooks) == 1
    hook, tparams = hooks[0]
    assert tparams == ["in_bw"]

    stub = impl_stub_to_tpp(comp, hook, tparams)
    for sub in [
        "// This file is included from tcomp.hpp",
        "template <int in_bw>",
        "ap_uint<8> process(",
        "axi4s_word<in_bw>",
        "// TODO: implement process",
        "return ap_uint<8>(0);",
        "namespace tcomp {",
    ]:
        assert sub in stub, f"Missing substring: {sub!r}\n--- stub ---\n{stub}"


def test_impl_stub_to_tpp_does_not_include_header():
    """The .tpp is included from the .hpp, so it must NOT include the .hpp itself."""
    from pysilicon.build.hwcodegen import extract_kernel
    from pysilicon.build.hwgen import (
        _collect_hooks_with_params,
        impl_stub_to_tpp,
    )

    comp = _TmplComp(name="tcomp", sim=Simulation())
    tree = extract_kernel(comp)
    hook, tparams = _collect_hooks_with_params(tree)[0]
    stub = impl_stub_to_tpp(comp, hook, tparams)
    assert '#include "tcomp.hpp"' not in stub


def test_kernel_files_to_str_uses_tpp_for_templated_hook():
    from pysilicon.build.hwgen import kernel_files_to_str

    comp = _TmplComp(name="tcomp", sim=Simulation())
    files = kernel_files_to_str(comp)
    assert "tcomp.hpp" in files
    assert "tcomp.cpp" in files
    # Templated hook → .tpp, NOT .cpp.
    assert "tcomp_process_impl.tpp" in files
    assert "tcomp_process_impl.cpp" not in files


def test_kernel_files_to_str_uses_cpp_for_non_templated_hook():
    """DemoComponent's process hook has no stream args, so still .cpp."""
    from pysilicon.build.hwgen import kernel_files_to_str
    from tests.hw.test_resolve import DemoComponent

    comp = DemoComponent(name="demo", sim=Simulation())
    files = kernel_files_to_str(comp)
    assert "demo_process_impl.cpp" in files
    assert "demo_process_impl.tpp" not in files

