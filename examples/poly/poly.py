from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pysilicon.hw.arrayutils import SchemaArray, read_array, read_uint32_file, write_array
from pysilicon.hw.clock import Clock
from pysilicon.hw.dataschema import DataArray, DataList, EnumField, FloatField, IntField
from pysilicon.hw.hw_component import HwComponent, HwConst, HwParam
from pysilicon.hw.interface import StreamIF, StreamIFMaster, StreamIFSlave
from pysilicon.hw.memif import DirectMMIF, MMIFMaster
from pysilicon.hw.regmap import (
    Bit,
    RegAccess,
    RegField,
    VitisRegMap,
    VitisRegMapMMIFSlave,
)
from pysilicon.hw.synth import sim_only, synthesizable
from pysilicon.simulation.logger import Logger, NullLogger
from pysilicon.simulation.simobj import ProcessGen, SimObj
from pysilicon.simulation.simulation import Simulation


INCLUDE_DIR = "include"
WORD_BW_SUPPORTED = [32, 64]
TxIdField = IntField.specialize(bitwidth=16, signed=False)
NsampField = IntField.specialize(bitwidth=16, signed=False)
Float32 = FloatField.specialize(bitwidth=32, include_dir=INCLUDE_DIR)


class PolyError(IntEnum):
    NO_ERROR = 0
    TLAST_EARLY_CMD_HDR = 1
    NO_TLAST_CMD_HDR = 2
    TLAST_EARLY_SAMP_IN = 3
    NO_TLAST_SAMP_IN = 4
    WRONG_NSAMP = 5

PolyErrorField = EnumField.specialize(enum_type=PolyError)


class PolyCmdType(IntEnum):
    DATA = 0
    END = 1

PolyCmdTypeField = EnumField.specialize(enum_type=PolyCmdType)


class CoeffArray(DataArray):
    """Array of polynomial coefficients in ascending order (constant term first)."""
    ncoeff: HwConst[int] = 4
    element_type = Float32
    static = True
    max_shape = (ncoeff,)


class PolyCmdHdr(DataList):
    """Command header: type, transaction ID, and sample count.

    Coefficients are configured separately via the AXI-Lite register map.
    The END variant carries cmd_type=END and nsamp=0; it signals the kernel
    to break the persistent processing loop and return cleanly.
    """
    elements = {
        "cmd_type": {"schema": PolyCmdTypeField, "description": "DATA or END"},
        "tx_id":    {"schema": TxIdField,        "description": "Transaction ID"},
        "nsamp":    {"schema": NsampField,       "description": "Sample count (0 for END)"},
    }


class PolyRespHdr(DataList):
    """Response header: echo of the transaction ID."""
    elements = {
        "tx_id": {"schema": TxIdField, "description": "Echo of the transaction ID"},
    }


SCHEMA_CLASSES = [
    PolyErrorField,
    PolyCmdTypeField,
    CoeffArray,
    PolyCmdHdr,
    PolyRespHdr,
]


@dataclass(slots=True)
class PolySimResult:
    """Result bundle from a polynomial accelerator simulation run.

    The per-transaction footer is gone; halt/error status comes from the
    regmap and is serialized to ``regmap_status.json`` alongside the
    sample-data outputs.
    """

    cmd_hdr:  PolyCmdHdr
    samp_in:  npt.NDArray[np.float32]
    resp_hdr: PolyRespHdr
    samp_out: npt.NDArray[np.float32]
    halted:   int
    error:    PolyError
    tx_id:    int

    @property
    def passed(self) -> bool:
        return self.error == PolyError.NO_ERROR and self.halted == 0

    @classmethod
    def from_paths(
        cls,
        cmd_hdr_path: Path,
        samp_in_path: Path,
        resp_dir: Path,
    ) -> PolySimResult:
        """Reconstruct a PolySimResult by reading its component files from disk.

        ``cmd_hdr_path`` / ``samp_in_path`` point at the input test vectors
        (written by BuildInputsStep); ``resp_dir`` is the directory holding
        ``resp_hdr.bin``, ``samp_out.bin`` and ``regmap_status.json``.
        """
        cmd_hdr = PolyCmdHdr().read_uint32_file(cmd_hdr_path)
        samp_in = np.array(
            read_uint32_file(samp_in_path, elem_type=Float32, shape=int(cmd_hdr.nsamp)),
            dtype=np.float32,
        )
        resp_hdr = PolyRespHdr().read_uint32_file(resp_dir / "resp_hdr.bin")
        status = json.loads((resp_dir / "regmap_status.json").read_text(encoding="utf-8"))
        samp_out_len = int(cmd_hdr.nsamp)
        samp_out = np.array(
            read_uint32_file(resp_dir / "samp_out.bin", elem_type=Float32,
                             shape=samp_out_len),
            dtype=np.float32,
        )
        return cls(
            cmd_hdr=cmd_hdr, samp_in=samp_in,
            resp_hdr=resp_hdr, samp_out=samp_out,
            halted=int(status["halted"]),
            error=PolyError(int(status["error"])),
            tx_id=int(status["tx_id"]),
        )


@dataclass
class PolyAccelComponent(HwComponent):
    """SimPy model of the polynomial accelerator kernel.

    Control/status is exposed via an AXI-Lite VitisRegMap; the host writes
    ``ap_start`` to launch ``on_start``, which loops reading commands from
    the input stream until it sees an END header or hits an error.
    Coefficients live in the regmap and must be configured before launch
    (they default to zeros, which produces an all-zero output stream).
    """

    in_bw:        HwParam[int] = 32
    out_bw:       HwParam[int] = 32
    aximm_bw:     HwParam[int] = 32
    clk:          Clock = field(default_factory=lambda: Clock(freq=1e9))
    proc_ii:      int = 1
    proc_latency: int = 10
    logger:       Logger | NullLogger = field(default_factory=NullLogger)
    unroll_factor: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        self.s_in  = StreamIFSlave( name=f'{self.name}_s_in',  sim=self.sim, bitwidth=self.in_bw)
        self.m_out = StreamIFMaster(name=f'{self.name}_m_out', sim=self.sim, bitwidth=self.out_bw)
        self.regmap = VitisRegMap({
            "halted": RegField(Bit,            RegAccess.R,  description="1 = halted on error"),
            "error":  RegField(PolyErrorField, RegAccess.R,  description="Last error code"),
            "tx_id":  RegField(TxIdField,      RegAccess.R,  description="TX id of halted txn"),
            "coeffs": RegField(CoeffArray,     RegAccess.RW, description="Polynomial coefficients"),
        }, bitwidth=self.aximm_bw)
        self.s_lite = VitisRegMapMMIFSlave(
            name=f'{self.name}_s_lite', sim=self.sim, bitwidth=self.aximm_bw,
            regmap=self.regmap, on_start=self.on_start,
        )
        for ep in (self.s_in, self.m_out, self.s_lite):
            self.add_endpoint(ep)
        self._job: int = 0

    @sim_only
    def _inc_job(self) -> None:
        self._job += 1

    def on_start(self) -> ProcessGen[None]:
        """Kernel body — invoked by VitisRegMapMMIFSlave on host ap_start write."""
        while True:
            self.logger.log(event='proc_begin', job=self._job)
            cmd_hdr: PolyCmdHdr = yield from self.s_in.get(PolyCmdHdr)
            if cmd_hdr.cmd_type == PolyCmdType.END:
                self.logger.log(event='proc_end', job=self._job)
                return
            err = yield from self.evaluate(cmd_hdr, self.s_in, self.m_out)
            self._inc_job()
            if err != PolyError.NO_ERROR:
                self.regmap.set("error",  err)
                self.regmap.set("tx_id",  cmd_hdr.tx_id)
                self.regmap.set("halted", 1)
                return

    @synthesizable
    def evaluate(
        self,
        cmd_hdr: PolyCmdHdr,
        s_in: StreamIFSlave,
        m_out: StreamIFMaster,
    ) -> ProcessGen[PolyError]:
        """Process one DATA transaction. Returns NO_ERROR or an error code."""
        resp_hdr = PolyRespHdr()
        resp_hdr.tx_id = cmd_hdr.tx_id
        self.logger.log(event='resp_hdr_write_begin', job=self._job)
        yield from m_out.write(resp_hdr)

        self.logger.log(event='samp_read_begin', job=self._job)
        samp_in, tstart = yield from s_in.get_pipelined(Float32, count=cmd_hdr.nsamp)

        coeffs = self.regmap.get("coeffs").val
        y = np.zeros_like(samp_in, dtype=np.float32)
        power = np.ones_like(samp_in, dtype=np.float32)
        for coeff in coeffs:
            y += coeff * power
            power *= samp_in

        t_out_start = tstart + self.proc_latency * self.clk.period
        proc_time = cmd_hdr.nsamp / self.unroll_factor * self.proc_ii * self.clk.period
        proc_time = max(0.0, proc_time + (t_out_start - self.env.now))
        yield self.timeout(proc_time)

        yield from m_out.write_pipelined(SchemaArray(data=y, elem_type=Float32), t_out_start)
        self.logger.log(event='samp_out_write_end', job=self._job)

        if len(samp_in) != cmd_hdr.nsamp:
            return PolyError.WRONG_NSAMP
        return PolyError.NO_ERROR


@dataclass(kw_only=True)
class PolyTB(SimObj):
    """Drives one polynomial transaction and captures the response.

    Writes ``coeffs`` to the regmap, asserts ap_start, sends one DATA
    cmd_hdr + samples, reads back the resp_hdr + samp_out pair, sends an
    END cmd_hdr to terminate the kernel loop, then reads halted/error/tx_id
    from the regmap.
    """

    cmd_hdr:   PolyCmdHdr
    samp_in:   npt.NDArray[np.float32]
    coeffs:    npt.NDArray[np.float32]
    word_bw:   int = 32
    base_addr: int = 0x0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.m_in   = StreamIFMaster(name=f'{self.name}_m_in',   sim=self.sim, bitwidth=self.word_bw)
        self.s_out  = StreamIFSlave( name=f'{self.name}_s_out',  sim=self.sim, bitwidth=self.word_bw)
        self.m_lite = MMIFMaster(    name=f'{self.name}_m_lite', sim=self.sim, bitwidth=32)
        self.resp_hdr:     PolyRespHdr | None              = None
        self.samp_out:     npt.NDArray[np.float32] | None  = None
        self.halted:       int | None                      = None
        self.error:        PolyError | None                = None
        self.tx_id_status: int | None                      = None
        self._regmap_ref:  VitisRegMap | None              = None

    def run_proc(self) -> ProcessGen[None]:
        bw = self.word_bw
        regmap = self._regmap()

        yield from self.m_lite.write_schema(
            CoeffArray(self.coeffs),
            addr=self.base_addr + regmap.offset_of("coeffs"),
        )

        yield from regmap.start(self.m_lite, base_addr=self.base_addr)

        yield from self.m_in.write(self.cmd_hdr.serialize(word_bw=bw))
        yield from self.m_in.write(write_array(self.samp_in, elem_type=Float32, word_bw=bw))

        resp_words = yield from self.s_out.get()
        samp_words = yield from self.s_out.get()
        self.resp_hdr = PolyRespHdr().deserialize(resp_words, word_bw=bw)
        self.samp_out = read_array(samp_words, elem_type=Float32, word_bw=bw,
                                   shape=int(self.cmd_hdr.nsamp))

        end_hdr = PolyCmdHdr()
        end_hdr.cmd_type = PolyCmdType.END
        end_hdr.tx_id    = 0
        end_hdr.nsamp    = 0
        yield from self.m_in.write(end_hdr.serialize(word_bw=bw))

        yield self.timeout(0)
        halted_field = yield from self.m_lite.read_schema(
            Bit, addr=self.base_addr + regmap.offset_of("halted"))
        error_field = yield from self.m_lite.read_schema(
            PolyErrorField, addr=self.base_addr + regmap.offset_of("error"))
        tx_id_field = yield from self.m_lite.read_schema(
            TxIdField, addr=self.base_addr + regmap.offset_of("tx_id"))
        self.halted       = int(halted_field.val)
        self.error        = PolyError(int(error_field.val))
        self.tx_id_status = int(tx_id_field.val)

    def _regmap(self) -> VitisRegMap:
        if self._regmap_ref is None:
            raise RuntimeError(
                "PolyTB._regmap_ref is unset; call connect() before run_sim()."
            )
        return self._regmap_ref


def connect(sim: Simulation, tb: PolyTB, accel: PolyAccelComponent, clk: Clock) -> None:
    """Wire a testbench's master/slave ports to the accelerator via two StreamIFs and a DirectMMIF."""
    in_stream  = StreamIF(sim=sim, clk=clk)
    out_stream = StreamIF(sim=sim, clk=clk)
    lite_link  = DirectMMIF(sim=sim, clk=clk, byte_addressable=True)
    in_stream.bind( "master", tb.m_in)
    in_stream.bind( "slave",  accel.s_in)
    out_stream.bind("master", accel.m_out)
    out_stream.bind("slave",  tb.s_out)
    lite_link.bind( "master", tb.m_lite)
    lite_link.bind( "slave",  accel.s_lite)
    tb._regmap_ref = accel.regmap
