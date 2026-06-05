"""
incr.py — increment-a-buffer accelerator (the AXI-MM codegen toy).

This is the smallest example that fully stresses the m_axi path: the kernel
reads ``n`` words from external memory at ``cmd.addr``, adds 1 to each, and
writes them back in place, then emits a one-word response.  It exercises an
m_axi **read** + **write**, byte→word address conversion, a local buffer sized
from a compile-time bound (``max_n``), a stream command, and a stream response —
everything the AXI-MM queue and a real histogram will later need.

Control is via AXI-Stream + ``ap_ctrl_hs`` (the histogram model), not a regmap:
the command rides the input stream and the response rides the output stream.
Because there is no ``VitisRegMapMMIFSlave``, the codegen kernel root is
``run_proc`` (see ``hwcodegen.extract_kernel``), so the synthesizable kernel
body lives there.

Topology (Python sim)
---------------------
  IncrController.m_cmd ──StreamIF──▶ IncrAccel.s_in
  IncrAccel.m_out      ──StreamIF──▶ IncrController.s_resp
  IncrAccel.m_mem      ──DirectMMIF─▶ MemComponent.s_mm ──▶ Memory

In Vitis codegen the master / interconnect / slave collapse: ``m_mem`` becomes
an ``m_axi`` pointer param, ``MemComponent`` + ``Memory`` become a flat C array,
and the interconnect is just passing the pointer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from pysilicon.hw.arrayutils import get_nwords, write_uint32_file
from pysilicon.hw.clock import Clock
from pysilicon.hw.dataschema import DataArray, DataList, EnumField, IntField, MemAddr
from pysilicon.hw.hw_component import HwComponent, HwParam
from pysilicon.hw.hw_testbench import HwTestbench
from pysilicon.hw.interface import StreamIF, StreamIFMaster, StreamIFSlave
from pysilicon.hw.memif import DirectMMIF
from pysilicon.hw.memory import MemComponent
from pysilicon.hw.synth import synthesizable
from pysilicon.simulation.simobj import ProcessGen, SimObj
from pysilicon.simulation.simulation import Simulation


INCLUDE_DIR = "include"
WORD_BW_SUPPORTED = [32, 64]
MAX_N = 1024            # compile-time buffer bound (decision 5)
MEM_AWIDTH = 64

Uint32Field = IntField.specialize(bitwidth=32, signed=False, include_dir=INCLUDE_DIR)
AddrField = MemAddr.specialize(bitwidth=MEM_AWIDTH)


class IncrError(IntEnum):
    NO_ERROR = 0
    INVALID_N = 1


IncrErrorField = EnumField.specialize(enum_type=IncrError)


class IncrCmd(DataList):
    """Command descriptor: base byte address + element count."""

    elements = {
        "addr": {"schema": AddrField, "description": "Base byte address of the buffer"},
        "n":    {"schema": Uint32Field, "description": "Number of words to increment"},
    }


class IncrResp(DataList):
    """Response descriptor: execution status."""

    elements = {
        "status": {"schema": IncrErrorField, "description": "Execution status code"},
    }


class IncrArray(DataArray):
    """Uint32 buffer for the codegen-source testbench.

    Sized to ``max_shape=(MAX_N,)`` so the generated C++ testbench can declare
    ``ap_uint<32> buf[MAX_N];`` statically; the runtime element count is passed
    per call via ``count=``.
    """

    element_type = Uint32Field
    static = True
    max_shape = (MAX_N,)
    cpp_storage = "raw"


SCHEMA_CLASSES = [
    IncrErrorField,
    IncrCmd,
    IncrResp,
    IncrArray,
]


# ---------------------------------------------------------------------------
# Golden model
# ---------------------------------------------------------------------------

def golden(input_buf: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
    """Reference transform: each word incremented by one."""
    return (np.asarray(input_buf, dtype=np.uint32) + 1).astype(np.uint32)


# ---------------------------------------------------------------------------
# Accelerator (SimPy model + codegen source)
# ---------------------------------------------------------------------------

@dataclass
class IncrAccel(HwComponent):
    """SimPy model of the increment-buffer kernel.

    ``run_proc`` is the synthesizable kernel body (stream-controlled, so the
    codegen root is ``run_proc``).  It reads one :class:`IncrCmd`, loads the
    buffer from memory over ``m_mem``, increments it, writes it back, and emits
    one :class:`IncrResp`.
    """

    cpp_kernel_name: ClassVar[str | None] = "incr"
    cpp_namespace:   ClassVar[str | None] = "incr_impl"

    in_bw:   HwParam[int] = 32
    out_bw:  HwParam[int] = 32
    mem_bw:  HwParam[int] = 32
    max_n:   HwParam[int] = MAX_N
    clk:     Clock = field(default_factory=lambda: Clock(freq=1e9))

    def __post_init__(self) -> None:
        super().__post_init__()
        self.s_in  = StreamIFSlave( name=f'{self.name}_s_in',  sim=self.sim, bitwidth=self.in_bw)
        self.m_out = StreamIFMaster(name=f'{self.name}_m_out', sim=self.sim, bitwidth=self.out_bw)
        from pysilicon.hw.memif import MMIFMaster
        self.m_mem = MMIFMaster(name=f'{self.name}_m_mem', sim=self.sim, bitwidth=self.mem_bw)
        for ep in (self.s_in, self.m_out, self.m_mem):
            self.add_endpoint(ep)

    def run_proc(self) -> ProcessGen[None]:
        """Kernel body (single ap_ctrl_hs invocation).

        Written in the synthesizable restricted subset: each statement is a
        ``yield from`` call to a synthesizable endpoint method or hook.  The
        ``+1`` transform and the response write live in :meth:`transform` /
        :meth:`respond` hooks (FunctionStmt), mirroring poly's ``evaluate``.
        """
        cmd: IncrCmd = yield from self.s_in.get(IncrCmd)
        buf = yield from self.m_mem.read_array(Uint32Field, cmd.n, cmd.addr)
        yield from self.transform(buf, cmd.n)
        yield from self.m_mem.write_array(buf, Uint32Field, cmd.addr, cmd.n)
        yield from self.respond(self.m_out)

    @synthesizable
    def transform(self, buf: IncrArray, n: Uint32Field) -> ProcessGen[None]:
        """The increment transform hook: each of the first ``n`` words gets +1,
        in place (C++ can't return an array by value, so the kernel writes the
        same buffer back)."""
        np.asarray(buf)[:int(n)] += 1
        return
        yield  # unreachable — makes this a generator

    @synthesizable
    def respond(self, m_out: StreamIFMaster) -> ProcessGen[None]:
        """Emit the (always NO_ERROR) response on the output stream."""
        resp = IncrResp()
        resp.status = IncrError.NO_ERROR
        yield from m_out.write(resp)


# ---------------------------------------------------------------------------
# SimPy controller (timing-accurate testbench, mirrors PolyTB)
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class IncrController(SimObj):
    """Drives one increment transaction against the accelerator.

    Allocates a buffer in the shared memory, writes the input data into it
    (TB-side, via the memory's owner master), pushes the command, waits for the
    response, and reads the kernel-produced buffer back.
    """

    mem: MemComponent
    input_buf: npt.NDArray[np.uint32]
    word_bw: int = 32

    def __post_init__(self) -> None:
        super().__post_init__()
        self.m_cmd  = StreamIFMaster(name=f'{self.name}_m_cmd',  sim=self.sim, bitwidth=self.word_bw)
        self.s_resp = StreamIFSlave( name=f'{self.name}_s_resp', sim=self.sim, bitwidth=self.word_bw)
        self.addr: int | None = None
        self.resp: IncrResp | None = None
        self.result: npt.NDArray[np.uint32] | None = None

    def run_proc(self) -> ProcessGen[None]:
        bw = self.word_bw
        n = len(self.input_buf)

        # Allocate + populate the input buffer (TB-side memory access).
        nwords = get_nwords(Uint32Field, word_bw=self.mem.word_size, shape=n)
        self.addr = self.mem.alloc(nwords)
        yield from self.mem.m_mm.write_array(
            np.asarray(self.input_buf, dtype=np.uint32), Uint32Field, self.addr, word_bw=bw)

        # Issue the command and await the response.
        cmd = IncrCmd()
        cmd.addr = self.addr
        cmd.n = n
        yield from self.m_cmd.write(cmd)

        resp_words = yield from self.s_resp.get()
        self.resp = IncrResp().deserialize(resp_words, word_bw=bw)

        # Read the kernel-produced buffer back.
        out = yield from self.mem.m_mm.read_array(Uint32Field, n, self.addr, word_bw=bw)
        self.result = np.asarray(out, dtype=np.uint32)


def connect(sim: Simulation, ctrl: IncrController, accel: IncrAccel,
            mem: MemComponent, clk: Clock) -> None:
    """Wire controller ↔ accelerator (two StreamIFs) and accelerator → memory."""
    in_stream  = StreamIF(sim=sim, clk=clk)
    out_stream = StreamIF(sim=sim, clk=clk)
    mem_link   = DirectMMIF(sim=sim, clk=clk, byte_addressable=True)
    in_stream.bind( "master", ctrl.m_cmd)
    in_stream.bind( "slave",  accel.s_in)
    out_stream.bind("master", accel.m_out)
    out_stream.bind("slave",  ctrl.s_resp)
    mem_link.bind(  "master", accel.m_mem)
    mem_link.bind(  "slave",  mem.s_mm)


# ---------------------------------------------------------------------------
# Codegen-source testbench (consumed by Phase 5 TB codegen; mirrors PolyTBHls)
# ---------------------------------------------------------------------------

@dataclass
class IncrTBHls(HwTestbench):
    """Sequential codegen-source testbench for the increment kernel.

    ``main()`` lowers to ``gen/incr_tb.cpp``: read the command + input buffer
    from disk, allocate a region (preserving alloc order — decision 8), populate
    it, run the DUT with the ``mem`` pointer, drain the response, and write the
    kernel-produced buffer back out for the functional-verify step.
    """

    cpp_kernel_name: ClassVar[str | None] = "incr"

    def main(self) -> None:
        dut = IncrAccel()
        mem = MemComponent(name="mem", sim=None, inline=False, nwords_tot=MAX_N)

        cmd = IncrCmd()
        cmd.read_uint32_file(self.data_dir + "/cmd.bin")

        buf = IncrArray()
        buf.read_uint32_file_array(self.data_dir + "/in.bin", count=cmd.n)

        # Address comes from alloc, not from the file (decision 8).
        cmd.addr = mem.alloc_array(buf, Uint32Field, count=cmd.n)

        dut.s_in.push(cmd)
        dut.run(mem=mem)

        resp = IncrResp()
        dut.m_out.pop(resp)

        out = mem.read_array(cmd.addr, Uint32Field, count=cmd.n)

        # Outputs for FunctionalVerifyStep (actual side, in the data dir).
        resp.write_uint32_file(self.data_dir + "/resp_data.bin")
        out.write_uint32_file_array(self.data_dir + "/out_data.bin", count=cmd.n)


# ---------------------------------------------------------------------------
# Harness + inputs
# ---------------------------------------------------------------------------

def build_inputs(data_dir: str | Path, input_buf: npt.NDArray[np.uint32]) -> Path:
    """Write ``in.bin`` + ``cmd.bin`` + ``params.json`` for the Vitis testbench."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    n = len(input_buf)
    write_uint32_file(np.asarray(input_buf, dtype=np.uint32), elem_type=Uint32Field,
                      file_path=data_dir / "in.bin", nwrite=n)
    # cmd.bin carries n; addr is a placeholder (the TB allocates and overwrites).
    cmd = IncrCmd()
    cmd.addr = 0
    cmd.n = n
    cmd.write_uint32_file(data_dir / "cmd.bin")
    (data_dir / "params.json").write_text(json.dumps({"n": int(n)}, indent=2),
                                          encoding="utf-8")
    return data_dir


@dataclass
class IncrResult:
    """Result bundle from an increment simulation run."""

    input_buf: npt.NDArray[np.uint32]
    result: npt.NDArray[np.uint32]
    expected: npt.NDArray[np.uint32]
    status: IncrError

    @property
    def passed(self) -> bool:
        return self.status == IncrError.NO_ERROR and np.array_equal(self.result, self.expected)


def run_sim(input_buf: npt.NDArray[np.uint32], *, clk_freq: float = 1e9) -> IncrResult:
    """Run the SimPy increment sim and return observed vs golden results."""
    sim = Simulation()
    clk = Clock(freq=clk_freq)
    mem = MemComponent(name="mem", sim=sim, inline=False, clk=clk)
    accel = IncrAccel(name="incr_accel", sim=sim, clk=clk)
    ctrl = IncrController(name="incr_ctrl", sim=sim, mem=mem, input_buf=input_buf)
    connect(sim, ctrl, accel, mem, clk)
    sim.run_sim()
    return IncrResult(
        input_buf=np.asarray(input_buf, dtype=np.uint32),
        result=ctrl.result,
        expected=golden(input_buf),
        status=ctrl.resp.status if ctrl.resp is not None else IncrError.INVALID_N,
    )


def main() -> None:
    rng = np.random.default_rng(7)
    input_buf = rng.integers(0, 1000, size=37, dtype=np.uint32)
    res = run_sim(input_buf)
    print(f"increment sim: n={len(input_buf)}, status={res.status.name}, passed={res.passed}")
    if not res.passed:
        print(f"  expected={res.expected}")
        print(f"  got     ={res.result}")


if __name__ == "__main__":
    main()
