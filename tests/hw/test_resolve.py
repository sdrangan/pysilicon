"""Tests for the IR resolution pass (pysilicon/build/hwresolve.py)."""
from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import IntEnum

import pytest

from pysilicon.build.hwcodegen import HwStmtExtractor
from pysilicon.hw.dataschema import DataList, EnumField, IntField
from pysilicon.hw.hw_component import HwComponent, HwParam
from pysilicon.hw.hwstmt import (
    CaseStmt,
    FieldRef,
    HwVar,
    ReturnStmt,
    SeqStmt,
    WhileStmt,
)
from pysilicon.hw.interface import StreamIFMaster, StreamIFSlave
from pysilicon.hw.regmap import (
    Bit,
    RegAccess,
    RegField,
    VitisRegMap,
    VitisRegMapMMIFSlave,
)
from pysilicon.hw.synth import sim_only, synthesizable
from pysilicon.simulation.simobj import ProcessGen
from pysilicon.simulation.simulation import Simulation


# ---------------------------------------------------------------------------
# Module-level schemas / enums used by the demo component
# ---------------------------------------------------------------------------

class DemoCmdType(IntEnum):
    DATA = 0
    END  = 1


class DemoError(IntEnum):
    OK         = 0
    BAD_INPUT  = 1
    OVERFLOW   = 2


DemoCmdTypeField = EnumField.specialize(enum_type=DemoCmdType)
DemoErrorField   = EnumField.specialize(enum_type=DemoError)
TxIdField        = IntField.specialize(bitwidth=16, signed=False)


class DemoCmdHdr(DataList):
    elements = {
        "cmd_type": {"schema": DemoCmdTypeField, "description": "DATA or END"},
        "tx_id":    {"schema": TxIdField,        "description": "Transaction ID"},
    }


# ---------------------------------------------------------------------------
# Demo component (mirrors experiment/extract_demo.py)
# ---------------------------------------------------------------------------

@dataclass
class DemoComponent(HwComponent):
    in_bw:  HwParam[int] = 32
    out_bw: HwParam[int] = 32

    def __post_init__(self) -> None:
        super().__post_init__()
        self.s_in  = StreamIFSlave (name=f'{self.name}_s_in',  sim=self.sim, bitwidth=self.in_bw)
        self.m_out = StreamIFMaster(name=f'{self.name}_m_out', sim=self.sim, bitwidth=self.out_bw)
        self.regmap = VitisRegMap({
            "halted": RegField(Bit,            RegAccess.R),
            "error":  RegField(DemoErrorField, RegAccess.R),
            "tx_id":  RegField(TxIdField,      RegAccess.R),
        })
        self.s_lite = VitisRegMapMMIFSlave(
            name=f'{self.name}_s_lite', sim=self.sim, bitwidth=32,
            regmap=self.regmap, on_start=self.on_start,
        )
        for ep in (self.s_in, self.m_out, self.s_lite):
            self.add_endpoint(ep)

    @sim_only
    def log(self, msg: str) -> None:
        pass

    def on_start(self) -> ProcessGen[None]:
        while True:
            self.log("waiting")
            cmd: DemoCmdHdr = yield from self.s_in.get(DemoCmdHdr)
            if cmd.cmd_type == DemoCmdType.END:
                return
            err = yield from self.process(cmd)
            if err != DemoError.OK:
                self.regmap.set("error",  err)
                self.regmap.set("tx_id",  cmd.tx_id)
                self.regmap.set("halted", 1)
                return

    @synthesizable
    def process(self, cmd: DemoCmdHdr) -> ProcessGen[DemoError]:
        yield self.env.timeout(0)
        return DemoError.OK


def _make_demo() -> DemoComponent:
    return DemoComponent(name="demo", sim=Simulation())


# ---------------------------------------------------------------------------
# Phase 1: CaseStmt.field becomes Optional[str]
# ---------------------------------------------------------------------------

def test_bare_var_case_stmt_has_none_field():
    """`if err == DemoError.OK:` — left side is a bare HwVar, no field."""
    comp = _make_demo()
    tree = HwStmtExtractor(comp, method_name='on_start').extract()
    assert isinstance(tree, WhileStmt)
    body = tree.body.stmts
    # body[3] is `if err != DemoError.OK:` in the demo on_start.
    case = body[3]
    assert isinstance(case, CaseStmt)
    assert case.field is None
    assert case.op == '!='


def test_field_case_stmt_has_field_name():
    """`if cmd.cmd_type == DemoCmdType.END:` — left side is var.field."""
    comp = _make_demo()
    tree = HwStmtExtractor(comp, method_name='on_start').extract()
    body = tree.body.stmts
    case = body[1]
    assert isinstance(case, CaseStmt)
    assert case.field == 'cmd_type'
    assert case.op == '=='
