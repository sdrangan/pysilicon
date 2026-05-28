from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from pysilicon.hw.aximm import DirectMMIF, MMIFMaster
from pysilicon.hw.clock import Clock
from pysilicon.hw.dataschema import IntField
from pysilicon.hw.hw_component import HwComponent
from pysilicon.hw.hw_testbench import HwTestbench
from pysilicon.hw.regmap import RegAccess, RegField, VitisRegMap, VitisRegMapMMIFSlave
from pysilicon.hw.synth import sim_only, synthesizable
from pysilicon.simulation.logger import Logger, NullLogger
from pysilicon.simulation.simobj import ProcessGen, SimObj
from pysilicon.simulation.simulation import Simulation


S32 = IntField.specialize(bitwidth=32, signed=True)
U32 = IntField.specialize(bitwidth=32, signed=False)

STATUS_IDLE = 0
STATUS_BUSY = 1
STATUS_DONE = 2

DEFAULT_VECTOR = {"x": 5, "a": 3, "b": -4}
DEFAULT_CASES = [
    {"x": 4, "a": 3, "b": -20, "y": 0, "label": "negative clamps to zero"},
    {"x": 5, "a": 3, "b": -4, "y": 11, "label": "positive output"},
    {"x": -2, "a": -7, "b": 3, "y": 17, "label": "signed multiply-add"},
]


def relu_affine(x: int, a: int, b: int) -> int:
    return max(0, a * x + b)


@dataclass(frozen=True)
class SimpFunCase:
    x: int
    a: int
    b: int
    y: int | None = None
    label: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, int | str]) -> "SimpFunCase":
        return cls(
            x=int(data["x"]),
            a=int(data["a"]),
            b=int(data["b"]),
            y=None if data.get("y") is None else int(data["y"]),
            label=str(data.get("label", "")),
        )

    @property
    def expected_y(self) -> int:
        return relu_affine(self.x, self.a, self.b) if self.y is None else int(self.y)


def default_cases() -> list[SimpFunCase]:
    return [SimpFunCase.from_dict(item) for item in DEFAULT_CASES]


@dataclass
class SimpFunComponent(HwComponent):
    cpp_kernel_name: ClassVar[str | None] = "simp_fun"
    cpp_namespace: ClassVar[str | None] = "simp_fun_impl"

    clk: Clock = field(default_factory=lambda: Clock(freq=100e6))
    latency_cycles: int = 4
    logger: Logger | NullLogger = field(default_factory=NullLogger)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.regmap = VitisRegMap({
            "status": RegField(U32, RegAccess.R, description="0=idle, 1=busy, 2=done"),
            "x": RegField(S32, RegAccess.RW, description="Input operand"),
            "a": RegField(S32, RegAccess.RW, description="Multiply coefficient"),
            "b": RegField(S32, RegAccess.RW, description="Bias term"),
            "y": RegField(S32, RegAccess.R, description="relu(a*x + b)"),
        })
        self.regmap.set("status", STATUS_IDLE)
        self.regmap.set("y", 0)
        self.s_lite = VitisRegMapMMIFSlave(
            name=f"{self.name}_s_lite",
            sim=self.sim,
            bitwidth=32,
            regmap=self.regmap,
            on_start=self.on_start,
        )
        self.add_endpoint(self.s_lite)

    @sim_only
    def _log(self, event: str, value: int) -> None:
        self.logger.log(event=event, value=value)

    def on_start(self) -> ProcessGen[None]:
        self.regmap.set("status", STATUS_BUSY)
        self._log("status_busy", STATUS_BUSY)
        y = self.compute(
            self.regmap.get("x"),
            self.regmap.get("a"),
            self.regmap.get("b"),
        )
        self.regmap.set("y", y)
        self.regmap.set("status", STATUS_DONE)
        self._log("status_done", STATUS_DONE)

    @synthesizable
    def compute(self, x: S32, a: S32, b: S32) -> S32:
        return S32(relu_affine(int(x.val), int(a.val), int(b.val)))


@dataclass(kw_only=True)
class SimpFunHost(SimObj):
    case: SimpFunCase
    clk: Clock
    latency_cycles: int = 4
    logger: Logger | NullLogger = field(default_factory=NullLogger)
    base_addr: int = 0x0
    max_poll_cycles: int = 32

    def __post_init__(self) -> None:
        super().__post_init__()
        self.master = MMIFMaster(name=f"{self.name}_m_lite", sim=self.sim, bitwidth=32)
        self.y: int | None = None
        self.status: int | None = None
        self.poll_cycles: int = 0
        self.passed: bool = False
        self._regmap_ref: VitisRegMap | None = None

    @sim_only
    def _log(self, event: str, value: int) -> None:
        self.logger.log(event=event, value=value)

    def run_proc(self) -> ProcessGen[None]:
        rm = self._regmap().bind_master(self.master, base_addr=self.base_addr)
        yield from rm.set("x", self.case.x)
        yield from rm.set("a", self.case.a)
        yield from rm.set("b", self.case.b)
        self._log("ap_start_host", 1)
        yield from rm.start()
        yield self.timeout(self.latency_cycles * self.clk.period)

        status = STATUS_IDLE
        for poll in range(self.max_poll_cycles + 1):
            status = yield from rm.get("status")
            self.poll_cycles = poll
            if status == STATUS_DONE:
                break
            yield self.timeout(self.clk.period)
        else:
            raise RuntimeError("Timed out waiting for status == STATUS_DONE.")

        self._log("host_done", int(status))
        self.y = yield from rm.get("y")
        self.status = int(status)
        self.passed = self.y == self.case.expected_y and self.status == STATUS_DONE

    def _regmap(self) -> VitisRegMap:
        if self._regmap_ref is None:
            raise RuntimeError("SimpFunHost._regmap_ref is unset; call connect() first.")
        return self._regmap_ref


@dataclass(frozen=True)
class SimpFunSimResult:
    case: SimpFunCase
    y: int
    status: int
    poll_cycles: int
    passed: bool

    def to_dict(self) -> dict[str, int | bool | str]:
        return {
            "x": self.case.x,
            "a": self.case.a,
            "b": self.case.b,
            "expected_y": self.case.expected_y,
            "y": self.y,
            "status": self.status,
            "poll_cycles": self.poll_cycles,
            "passed": self.passed,
            "label": self.case.label,
        }


class SimpFunTBHls(HwTestbench):
    cpp_kernel_name: ClassVar[str | None] = "simp_fun"

    def main(self) -> None:
        dut = SimpFunComponent()
        dut.regmap.read_uint32_file("x", self.data_dir + "/x.bin")
        dut.regmap.read_uint32_file("a", self.data_dir + "/a.bin")
        dut.regmap.read_uint32_file("b", self.data_dir + "/b.bin")
        dut.run()

        dut.regmap.write_status_json(
            self.data_dir + "/regmap_status.json",
            fields=["status", "y"],
        )


def connect(sim: Simulation, host: SimpFunHost, accel: SimpFunComponent, clk: Clock) -> None:
    lite_link = DirectMMIF(sim=sim, clk=clk, byte_addressable=True)
    lite_link.bind("master", host.master)
    lite_link.bind("slave", accel.s_lite)
    host._regmap_ref = accel.regmap


def simulate_case(
    case: SimpFunCase,
    *,
    clk_freq: float = 100e6,
    latency_cycles: int = 4,
    log_file: str | Path | None = None,
) -> SimpFunSimResult:
    sim = Simulation()
    clk = Clock(freq=clk_freq)
    logger: Logger | NullLogger = NullLogger()
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger = Logger(name="simp_fun_log", sim=sim, file_path=log_path, fields=["event", "value"])
    accel = SimpFunComponent(name="simp_fun", sim=sim, clk=clk,
                             latency_cycles=latency_cycles, logger=logger)
    host = SimpFunHost(
        name="host",
        sim=sim,
        case=case,
        clk=clk,
        latency_cycles=latency_cycles,
        logger=logger,
    )
    connect(sim, host, accel, clk)
    sim.run_sim()
    if host.y is None or host.status is None:
        raise RuntimeError("Simulation completed without producing y/status.")
    return SimpFunSimResult(
        case=case,
        y=int(host.y),
        status=int(host.status),
        poll_cycles=int(host.poll_cycles),
        passed=bool(host.passed),
    )


def run_functional_cases(
    *,
    clk_freq: float = 100e6,
    latency_cycles: int = 4,
) -> list[dict[str, int | bool | str]]:
    return [
        simulate_case(case, clk_freq=clk_freq, latency_cycles=latency_cycles).to_dict()
        for case in default_cases()
    ]


def write_sim_summary(path: str | Path, result: SimpFunSimResult) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return path
