from __future__ import annotations

from pysilicon.hw.aximm import DirectMMIF, MMIFMaster
from pysilicon.hw.clock import Clock
from pysilicon.simulation.simulation import Simulation

from examples.interface.vitis_regmap_simp_fun.simp_fun import S32, SimpFunAccel, relu_ax_plus_b_int32


def _run_and_read_y(*, a: int, x: int, b: int, name: str = "simp") -> int:
    sim = Simulation()
    accel = SimpFunAccel(name=name, sim=sim)

    master = MMIFMaster(sim=sim, bitwidth=32)
    direct = DirectMMIF(sim=sim, clk=Clock(freq=1.0))
    direct.bind("master", master)
    direct.bind("slave", accel.s_lite)

    done = sim.env.event()
    result: dict[str, int] = {}

    def proc():
        yield from master.write_schema(S32(a), addr=accel.regmap.offset_of("a"))
        yield from master.write_schema(S32(x), addr=accel.regmap.offset_of("x"))
        yield from master.write_schema(S32(b), addr=accel.regmap.offset_of("b"))
        yield from accel.regmap.start(master)
        y = yield from master.read_schema(S32, addr=accel.regmap.offset_of("y"))
        result["y"] = int(y.val)

    def wrap():
        yield from proc()
        done.succeed()

    sim.env.process(wrap())
    sim.env.run(until=done)
    return result["y"]


def test_relu_ax_plus_b_int32_positive_case() -> None:
    assert relu_ax_plus_b_int32(2, 4, 3) == 11


def test_relu_ax_plus_b_int32_negative_clamps_to_zero() -> None:
    assert relu_ax_plus_b_int32(-2, 4, -1) == 0


def test_relu_ax_plus_b_int32_no_saturation_wraps_int32() -> None:
    # int32 max + 1 wraps to int32 min, then relu applies.
    assert relu_ax_plus_b_int32(2_147_483_647, 1, 1) == 0


def test_simp_fun_accel_on_start_computes_relu_ax_plus_b() -> None:
    assert _run_and_read_y(a=3, x=5, b=-7) == 8


def test_simp_fun_accel_on_start_uses_int32_wrap_no_saturation() -> None:
    assert _run_and_read_y(a=2_147_483_647, x=1, b=1, name="simp_wrap") == 0
