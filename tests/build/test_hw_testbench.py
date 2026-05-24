"""Tests for the ``HwTestbench`` class and testbench-mode extractor.

Phase 14 of the HwComponent codegen project introduces a separate codegen
source for testbench C++.  Phase 1 (this file) covers the wiring: the new
``HwTestbench`` class, its ``main()`` placeholder, and the
``extract_kernel`` routing that dispatches testbench subclasses through
``extract_testbench`` / the ``is_testbench=True`` extractor mode.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from pysilicon.build.hwcodegen import (
    HwStmtExtractor,
    extract_kernel,
    extract_testbench,
)
from pysilicon.hw.hw_testbench import HwTestbench
from pysilicon.hw.hwstmt import SeqStmt
from pysilicon.simulation.simulation import Simulation


pytestmark = pytest.mark.phase1


# ---------------------------------------------------------------------------
# Phase 1 — class + routing
# ---------------------------------------------------------------------------

def test_hw_testbench_is_a_hwcomponent():
    """``HwTestbench`` inherits from ``HwComponent`` so it picks up the
    ``HwParam`` / ``HwConst`` machinery and the simulation lifecycle."""
    from pysilicon.hw.hw_component import HwComponent
    assert issubclass(HwTestbench, HwComponent)


def test_hw_testbench_marker_is_set():
    """The codegen routing dispatches on the ``_is_testbench`` class
    marker.  Subclasses inherit ``True``; ``HwComponent`` proper does not
    have the marker set."""
    from pysilicon.hw.hw_component import HwComponent
    assert getattr(HwTestbench, '_is_testbench', False) is True
    assert getattr(HwComponent, '_is_testbench', False) is False


def test_base_main_raises_not_implemented():
    """The base-class ``main()`` is a placeholder that fails fast when a
    subclass forgets to override it."""
    tb = HwTestbench(name='unused', sim=Simulation())
    with pytest.raises(NotImplementedError, match='main'):
        tb.main()


@dataclass
class _EmptyTB(HwTestbench):
    """Trivial subclass — body is docstring-only, no real testbench logic
    yet.  Phase 3+ exercises real extraction; Phase 1 just confirms the
    routing through the extractor doesn't crash on a minimal body."""

    def main(self) -> None:
        """Phase 1 placeholder body."""


def test_extract_testbench_routes_through_main():
    """``extract_testbench`` reads ``comp.main`` (not ``run_proc``) and
    produces a tree without raising on the trivial body."""
    tb = _EmptyTB(name='tb', sim=Simulation())
    tree = extract_testbench(tb)
    assert isinstance(tree, SeqStmt)
    assert tree.stmts == []


def test_extract_kernel_dispatches_testbench_subclasses():
    """The legacy ``extract_kernel`` entry point auto-routes testbench
    subclasses through ``extract_testbench`` — callers don't need to
    branch on the marker."""
    tb = _EmptyTB(name='tb', sim=Simulation())
    tree = extract_kernel(tb)
    assert isinstance(tree, SeqStmt)


def test_extractor_carries_is_testbench_flag():
    """The mode flag is plumbed through; ``HwStmtExtractor`` stashes it
    so Phase 3/4 emitter logic can branch on the extractor's mode."""
    tb = _EmptyTB(name='tb', sim=Simulation())
    ext = HwStmtExtractor(tb, method_name='main', is_testbench=True)
    assert ext._is_testbench is True
    # Default is False — preserves backwards compat for kernel-mode callers.
    kernel_ext = HwStmtExtractor(tb, method_name='main')
    assert kernel_ext._is_testbench is False


# ---------------------------------------------------------------------------
# Phase 2 — HlsCodegenStep testbench mode
# ---------------------------------------------------------------------------

from typing import ClassVar


@dataclass
class _PolyTbStub(HwTestbench):
    """Minimal testbench class with ``cpp_kernel_name = "poly"`` so the
    Phase-2 emitter writes ``gen/poly_tb.cpp`` (matching what Phase 6
    will plug into ``poly_build.py``).  The body stays a docstring
    placeholder — real extraction lands in Phase 3+."""

    cpp_kernel_name: ClassVar[str | None] = "poly"

    def main(self) -> None:
        """Phase-2 placeholder body."""


@pytest.mark.phase2
def test_hls_codegen_step_auto_detects_testbench_mode():
    from pysilicon.build.hwcodegen_steps import HlsCodegenStep
    step = HlsCodegenStep(
        comp_class=_PolyTbStub,
        source_artifact="poly_source",
        output_dir="gen",
    )
    assert step._is_testbench is True
    # Kernel-mode component stays in kernel mode.
    from tests.hw.test_resolve import DemoComponent
    kernel_step = HlsCodegenStep(
        comp_class=DemoComponent,
        source_artifact="demo_src",
        output_dir="gen",
    )
    assert kernel_step._is_testbench is False


@pytest.mark.phase2
def test_hls_codegen_step_explicit_is_testbench_override():
    """``is_testbench=True`` forces TB mode even on a non-marker class."""
    from pysilicon.build.hwcodegen_steps import HlsCodegenStep
    from tests.hw.test_resolve import DemoComponent
    step = HlsCodegenStep(
        comp_class=DemoComponent,
        source_artifact="x",
        output_dir="gen",
        is_testbench=True,
    )
    assert step._is_testbench is True


@pytest.mark.phase2
def test_hls_codegen_step_testbench_produces_single_tb_file():
    """In TB mode, ``produces`` is just ``{<kernel>_tb: <kernel>_tb.cpp}``."""
    from pathlib import Path
    from pysilicon.build.hwcodegen_steps import HlsCodegenStep
    step = HlsCodegenStep(
        comp_class=_PolyTbStub,
        source_artifact="poly_source",
        output_dir="gen",
    )
    assert step.produces == {"poly_tb": Path("gen/poly_tb.cpp")}


@pytest.mark.phase2
def test_hls_codegen_step_run_emits_skeleton_tb_cpp(tmp_path):
    """``run()`` writes a compilable skeleton file in TB mode."""
    from pysilicon.build.build import BuildConfig
    from pysilicon.build.hwcodegen_steps import HlsCodegenStep
    step = HlsCodegenStep(
        comp_class=_PolyTbStub,
        source_artifact="poly_source",
        output_dir="gen",
    )
    artifacts = step.run(BuildConfig(root_dir=tmp_path))
    tb_path = tmp_path / "gen" / "poly_tb.cpp"
    assert artifacts == {"poly_tb": tb_path}
    body = tb_path.read_text(encoding="utf-8")
    # Skeleton must compile and reference the kernel header.
    assert '#include "poly.hpp"' in body
    assert "int main(int argc, char** argv)" in body
    assert "return 0;" in body


@pytest.mark.phase2
def test_tb_files_to_str_returns_single_file():
    from pysilicon.build.hwgen import tb_files_to_str
    files = tb_files_to_str(_PolyTbStub, output_dir="gen")
    assert set(files) == {"poly_tb.cpp"}
    assert "int main(" in files["poly_tb.cpp"]


# ---------------------------------------------------------------------------
# Phase 3 — DUT binding + dut.run() lowering
# ---------------------------------------------------------------------------

from examples.poly.poly import PolyAccelComponent


@dataclass
class _PolyTBPhase3(HwTestbench):
    """Minimal Phase-3 fixture: bind a PolyAccelComponent DUT and call run().

    Exercises the two IR nodes added in Phase 3 — ``DutBindStmt`` and
    ``KernelCallStmt`` — and the corresponding emitter logic in
    ``hwgen.tb_to_cpp``.  Subsequent phases extend the body with stream
    push/pop and file I/O against the same DUT binding.
    """

    cpp_kernel_name: ClassVar[str | None] = "poly"

    def main(self) -> None:
        dut = PolyAccelComponent()
        dut.run()


@pytest.mark.phase3
def test_phase3_extractor_produces_dut_bind_and_kernel_call():
    """The TB-mode extractor turns ``dut = PolyAccelComponent()`` + ``dut.run()``
    into a SeqStmt of [DutBindStmt, KernelCallStmt]."""
    from pysilicon.build.hwcodegen import extract_testbench
    from pysilicon.hw.hwstmt import DutBindStmt, KernelCallStmt
    tb = _PolyTBPhase3(name='tb', sim=Simulation())
    tree = extract_testbench(tb)
    assert isinstance(tree, SeqStmt)
    assert len(tree.stmts) == 2
    bind, call = tree.stmts
    assert isinstance(bind, DutBindStmt)
    assert bind.local_name == 'dut'
    assert bind.comp_class is PolyAccelComponent
    assert bind.kwargs == {}
    assert isinstance(call, KernelCallStmt)
    assert call.local_name == 'dut'


@pytest.mark.phase3
def test_phase3_emits_stream_and_regmap_locals_and_kernel_call():
    """The TB emitter produces stream local decls, regmap field decls
    (scalars and the raw-array ``coeffs``), and the kernel-call line."""
    from pysilicon.build.hwgen import tb_files_to_str
    files = tb_files_to_str(_PolyTBPhase3, output_dir="gen")
    body = files["poly_tb.cpp"]
    # Stream endpoints
    assert "hls::stream<streamutils::axi4s_word<32>> s_in;" in body
    assert "hls::stream<streamutils::axi4s_word<32>> m_out;" in body
    # Regmap scalars
    assert "ap_uint<1> halted = 0;" in body
    assert "ap_uint<8> error = 0;" in body
    assert "ap_uint<16> tx_id = 0;" in body
    # Raw-array regmap field
    assert "float coeffs[4] = {};" in body
    # Kernel call: arg order matches kernel_signature
    assert "poly(s_in, m_out, halted, error, tx_id, coeffs);" in body


@pytest.mark.phase3
def test_phase3_rejects_positional_dut_args():
    """DUT construction must use keyword arguments only — positional
    args are rejected at extraction time so the failure is surfaced
    before downstream emitter logic runs."""
    from pysilicon.build.hwcodegen import SynthesisError, extract_testbench

    @dataclass
    class _BadPositionalTB(HwTestbench):
        cpp_kernel_name: ClassVar[str | None] = "poly"

        def main(self) -> None:
            dut = PolyAccelComponent("bad")  # noqa: F841
            dut.run()

    tb = _BadPositionalTB(name='tb', sim=Simulation())
    with pytest.raises(SynthesisError, match="keyword arguments only"):
        extract_testbench(tb)


@pytest.mark.phase3
def test_phase3_dut_run_with_args_is_rejected():
    """``dut.run(...)`` with any args is rejected — the run-method
    surface is fixed (no inputs, no outputs) in v1."""
    from pysilicon.build.hwcodegen import SynthesisError, extract_testbench

    @dataclass
    class _BadRunArgsTB(HwTestbench):
        cpp_kernel_name: ClassVar[str | None] = "poly"

        def main(self) -> None:
            dut = PolyAccelComponent()
            dut.run(42)

    tb = _BadRunArgsTB(name='tb', sim=Simulation())
    with pytest.raises(SynthesisError, match="dut.run\\(\\) takes no arguments"):
        extract_testbench(tb)
