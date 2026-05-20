"""End-to-end extractor tests targeting the PolyAccelComponent kernel."""
from __future__ import annotations

import pytest

from pysilicon.build.hwcodegen import HwStmtExtractor
from pysilicon.hw.hw_component import HwComponent
from pysilicon.hw.hwstmt import (
    CaseStmt,
    ReturnStmt,
    SeqStmt,
    WhileStmt,
)
from pysilicon.hw.synth import synthesizable
from pysilicon.simulation.simulation import Simulation


# ---------------------------------------------------------------------------
# Minimal synthesizable endpoint for testing
# ---------------------------------------------------------------------------

class _MockEndpoint:
    """Stand-in for a stream endpoint with synthesizable get/write."""

    @synthesizable(synth_fn=lambda ctx, i, o: "")
    def get(self):
        pass

    @synthesizable(synth_fn=lambda ctx, i, o: "")
    def write(self, data):
        pass


def _make_comp(comp_cls):
    sim = Simulation()
    comp = comp_cls(sim=sim)
    comp.ep = _MockEndpoint()
    return comp


# ---------------------------------------------------------------------------
# Phase 1: ReturnStmt
# ---------------------------------------------------------------------------

class _ReturnInIfComp(HwComponent):
    def run_proc(self):
        while True:
            x = yield from self.ep.get()
            if x.f == 1:
                return


def test_return_inside_if_body():
    comp = _make_comp(_ReturnInIfComp)
    tree = HwStmtExtractor(comp).extract()
    assert isinstance(tree, WhileStmt)
    case_stmt = tree.body.stmts[1]
    assert isinstance(case_stmt, CaseStmt)
    first_in_branch = case_stmt.if_true.stmts[0]
    assert isinstance(first_in_branch, ReturnStmt)
    assert first_in_branch.value is None


class _ReturnAtTopComp(HwComponent):
    def run_proc(self):
        while True:
            x = yield from self.ep.get()
            return


def test_return_at_top_of_while():
    comp = _make_comp(_ReturnAtTopComp)
    tree = HwStmtExtractor(comp).extract()
    assert isinstance(tree, WhileStmt)
    assert isinstance(tree.body, SeqStmt)
    assert any(isinstance(s, ReturnStmt) for s in tree.body.stmts)


# ---------------------------------------------------------------------------
# Phase 2: != in CaseStmt
# ---------------------------------------------------------------------------

class _NotEqIfComp(HwComponent):
    def run_proc(self):
        while True:
            x = yield from self.ep.get()
            if x.f != 0:
                return


def test_case_stmt_not_eq_op():
    comp = _make_comp(_NotEqIfComp)
    tree = HwStmtExtractor(comp).extract()
    case_stmt = tree.body.stmts[1]
    assert isinstance(case_stmt, CaseStmt)
    assert case_stmt.op == '!='


class _EqIfComp(HwComponent):
    def run_proc(self):
        while True:
            x = yield from self.ep.get()
            if x.f == 0:
                return


def test_case_stmt_eq_op_default():
    comp = _make_comp(_EqIfComp)
    tree = HwStmtExtractor(comp).extract()
    case_stmt = tree.body.stmts[1]
    assert isinstance(case_stmt, CaseStmt)
    assert case_stmt.op == '=='


# ---------------------------------------------------------------------------
# Phase 3: extract_kernel policy helper
# ---------------------------------------------------------------------------

def test_extract_kernel_no_regmap_uses_run_proc():
    from pysilicon.build.hwcodegen import extract_kernel
    comp = _make_comp(_EqIfComp)
    tree = extract_kernel(comp)
    assert isinstance(tree, WhileStmt)


def test_extract_kernel_with_regmap_uses_on_start():
    """A component with a VitisRegMapMMIFSlave endpoint extracts on_start."""
    import sys
    from pathlib import Path
    POLY_DIR = Path(__file__).resolve().parents[2] / "examples" / "poly"
    if str(POLY_DIR) not in sys.path:
        sys.path.insert(0, str(POLY_DIR))

    from pysilicon.build.hwcodegen import HwStmtExtractor, extract_kernel
    from pysilicon.hw.regmap import VitisRegMapMMIFSlave

    # Construct a minimal HwComponent with a VitisRegMapMMIFSlave endpoint and
    # both run_proc and on_start methods. The kernel selector should pick on_start.
    from pysilicon.hw.regmap import RegAccess, RegField, VitisRegMap, Bit

    class _RegMapComp(HwComponent):
        def __post_init__(self):
            super().__post_init__()
            self.regmap = VitisRegMap({
                "halted": RegField(Bit, RegAccess.R),
            })
            self.s_lite = VitisRegMapMMIFSlave(
                name=f'{self.name}_s_lite', sim=self.sim, bitwidth=32,
                regmap=self.regmap, on_start=self.on_start,
            )
            self.ep_mock = _MockEndpoint()
            self.add_endpoint(self.s_lite)

        def run_proc(self):
            while True:
                yield self.timeout(0)

        def on_start(self):
            while True:
                x = yield from self.ep_mock.get()
                return

    sim = Simulation()
    comp = _RegMapComp(sim=sim)
    tree = extract_kernel(comp)
    assert isinstance(tree, WhileStmt)
    # on_start ends with a return; run_proc body would have produced a
    # yield self.timeout(0) which is not synthesizable.
    assert any(isinstance(s, ReturnStmt) for s in tree.body.stmts)
