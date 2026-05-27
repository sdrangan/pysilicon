"""Teaching example: AXI-Lite VitisRegMap kernel for y = relu(a*x + b)."""
from __future__ import annotations

from dataclasses import dataclass

from pysilicon.hw.regmap import RegAccess, RegField, VitisRegMap, VitisRegMapMMIFSlave
from pysilicon.hw.dataschema import IntField
from pysilicon.simulation.simobj import ProcessGen, SimObj

S32 = IntField.specialize(bitwidth=32, signed=True)


def _wrap_int32(value: int) -> int:
    wrapped = int(value) & 0xFFFFFFFF
    if wrapped & 0x80000000:
        wrapped -= 1 << 32
    return wrapped


def relu_ax_plus_b_int32(a: int, x: int, b: int) -> int:
    """Compute y = relu(a*x + b) with int32 wraparound semantics."""
    ax = _wrap_int32(_wrap_int32(a) * _wrap_int32(x))
    ax_plus_b = _wrap_int32(ax + _wrap_int32(b))
    return max(ax_plus_b, 0)


@dataclass
class SimpFunAccel(SimObj):
    """Minimal VitisRegMap accelerator with inputs a,x,b and output y."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.regmap = VitisRegMap({
            "a": RegField(S32, RegAccess.RW, description="int32 scale"),
            "x": RegField(S32, RegAccess.RW, description="int32 input"),
            "b": RegField(S32, RegAccess.RW, description="int32 bias"),
            "y": RegField(S32, RegAccess.R, description="int32 relu(a*x+b) output"),
        })
        self.s_lite = VitisRegMapMMIFSlave(
            name=f"{self.name}_s_lite",
            sim=self.sim,
            bitwidth=32,
            regmap=self.regmap,
            on_start=self.on_start,
        )

    def on_start(self) -> ProcessGen[None]:
        y = relu_ax_plus_b_int32(
            int(self.regmap.get("a").val),
            int(self.regmap.get("x").val),
            int(self.regmap.get("b").val),
        )
        self.regmap.set("y", y)
        yield from ()
