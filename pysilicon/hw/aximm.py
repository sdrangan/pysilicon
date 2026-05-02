"""
aximm.py: AXI Memory-Mapped (AXI-MM) crossbar interface for simulation.

Supports AXI-MM Full (burst) and AXI-MM Lite (register-style, single-beat)
slave endpoints on the same crossbar. Address-based routing and local-address
remapping are performed by the crossbar at transfer time.

Transaction API (on master endpoints)
--------------------------------------
write(words, global_addr) -> ProcessGen
    Transfer a burst of words to the slave covering global_addr.
    FULL slaves: one burst call. LITE slaves: silently split into nwords
    single-word transactions, each at an auto-incremented local address.

read(nwords, global_addr) -> ProcessGen  [return value via proc.value]
    Read nwords from the slave at global_addr. Caller pattern::

        proc = env.process(master_ep.read(nwords, addr))
        yield proc
        data = proc.value   # numpy array of shape (nwords,)

Latency model
-------------
FULL write : (latency_init + nwords) / clk.freq  seconds total
FULL read  : latency_init/clk.freq (request wire)
           + slave rx_read_proc duration
           + (latency_read_return + nwords) / clk.freq  (return wire)
LITE write : nwords × latency_per_word / clk.freq  (one txn per word)
LITE read  : nwords × latency_per_word / clk.freq  (one txn per word)

Address assignment utility
--------------------------
assign_address_ranges(slaves, [(base_addr, size), ...]) -> list[AXIMMAddressRange]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np
import simpy

from pysilicon.hw.interface import InterfaceEndpoint, QueuedTransferIF, Words
from pysilicon.simulation.simobj import ProcessGen


class AXIMMProtocol(Enum):
    """AXI-MM protocol variant for a slave endpoint."""
    FULL = "full"   # burst-capable; amortised latency over nwords
    LITE = "lite"   # register-style; one transaction per word


@dataclass(frozen=True)
class AXIMMAddressRange:
    """Half-open byte-address range [base_addr, base_addr + size)."""

    base_addr: int
    size: int

    def contains(self, addr: int) -> bool:
        return self.base_addr <= addr < self.base_addr + self.size

    def to_local(self, addr: int) -> int:
        return addr - self.base_addr


# Callable type aliases for slave callbacks.
RxWriteProc = Callable[[Words, int], ProcessGen]  # (words, local_addr) -> ProcessGen
RxReadProc  = Callable[[int,   int], ProcessGen]  # (nwords, local_addr) -> ProcessGen


# ---------------------------------------------------------------------------
# Slave endpoint
# ---------------------------------------------------------------------------

@dataclass
class AXIMMCrossBarIFSlave(InterfaceEndpoint):
    """
    Slave (target) endpoint for an :class:`AXIMMCrossBarIF`.

    Parameters
    ----------
    protocol : AXIMMProtocol
        FULL for burst-capable peripherals; LITE for register-map peripherals.
    bitwidth : int
        Data word width in bits.
    rx_write_proc : RxWriteProc | None
        Called as ``rx_write_proc(words, local_addr)`` for each incoming write
        transaction. For LITE slaves this is called once per word with a
        one-element array and the word's individual local byte address.
    rx_read_proc : RxReadProc | None
        Called as ``rx_read_proc(nwords, local_addr)``; the started process
        must set its return value (``proc.value``) to an ``(nwords,)`` array.
    latency_per_word : float
        Total cycles per single-word LITE transaction (ignored for FULL).
        Must be >= 2 to represent the AXI-Lite address + data phases.
    """

    protocol: AXIMMProtocol = AXIMMProtocol.FULL
    bitwidth: int = 32
    rx_write_proc: RxWriteProc | None = None
    rx_read_proc:  RxReadProc  | None = None
    latency_per_word: float = 2.0

    addr_range: AXIMMAddressRange | None = field(init=False)
    bus: simpy.Resource = field(init=False)

    type_name = 'aximm_crossbar_if_slave'

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.addr_range = None
        self.bus = simpy.Resource(self.env, capacity=1)


# ---------------------------------------------------------------------------
# Master endpoint
# ---------------------------------------------------------------------------

@dataclass
class AXIMMCrossBarIFMaster(InterfaceEndpoint):
    """
    Master (initiator) endpoint for an :class:`AXIMMCrossBarIF`.

    Exposes :meth:`write` and :meth:`read` using global byte addresses.
    """

    bitwidth: int = 32
    master_port: int = field(init=False)

    type_name = 'aximm_crossbar_if_master'

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.master_port = -1

    def _check_bound(self) -> None:
        if self.interface is None:
            raise RuntimeError(
                f"Cannot transact: {type(self).__name__} is not bound to an interface"
            )

    def write(self, words: Words, global_addr: int) -> ProcessGen:
        """
        Write a burst of words to the slave at *global_addr*.

        Example::

            yield self.process(master_ep.write(words, 0x0000))
        """
        self._check_bound()
        yield self.process(self.interface.write(words, global_addr, self.master_port))

    def read(self, nwords: int, global_addr: int) -> ProcessGen:
        """
        Read *nwords* from the slave at *global_addr*.

        Example::

            proc = env.process(master_ep.read(4, 0x0000))
            yield proc
            data = proc.value   # numpy array of shape (nwords,)
        """
        self._check_bound()
        proc = self.process(self.interface.read(nwords, global_addr, self.master_port))
        yield proc
        return proc.value


# ---------------------------------------------------------------------------
# Crossbar interface
# ---------------------------------------------------------------------------

@dataclass
class AXIMMCrossBarIF(QueuedTransferIF):
    """
    AXI-MM crossbar connecting ``nports_master`` master ports to
    ``nports_slave`` slave ports.

    Routing is address-based: the crossbar decodes ``global_addr`` against each
    slave's :attr:`~AXIMMCrossBarIFSlave.addr_range` and computes
    ``local_addr = global_addr - slave.addr_range.base_addr`` before invoking
    the slave callbacks.  A :class:`RuntimeError` is raised if no slave covers
    the address.

    Endpoint names
    --------------
    ``'master_0'`` … ``'master_{nports_master-1}'`` — bind
    :class:`AXIMMCrossBarIFMaster`

    ``'slave_0'`` … ``'slave_{nports_slave-1}'`` — bind
    :class:`AXIMMCrossBarIFSlave`

    Parameters
    ----------
    nports_master : int
        Number of master (initiator) ports.
    nports_slave : int
        Number of slave (target) ports.
    latency_init : float
        Fixed wire-latency cycles added to every FULL forward transfer.
    latency_read_return : float
        Fixed wire-latency cycles added to every FULL read return transfer.
    clk : Clock
        Reference clock; all cycle counts are divided by ``clk.freq``.
    bitwidth : int | None
        Data word width in bits; inferred from the first bound endpoint when None.
    """

    nports_master: int = 1
    nports_slave: int = 1
    latency_read_return: float = 0.

    type_name = 'aximm_crossbar_if'

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        if self.nports_master < 1:
            raise ValueError("nports_master must be at least 1")
        if self.nports_slave < 1:
            raise ValueError("nports_slave must be at least 1")
        self.endpoint_names = tuple(
            [f'master_{i}' for i in range(self.nports_master)] +
            [f'slave_{j}'  for j in range(self.nports_slave)]
        )
        super().__post_init__()

    # ------------------------------------------------------------------
    # Bind
    # ------------------------------------------------------------------

    def bind(self, ep_name: str, endpoint: InterfaceEndpoint) -> None:
        if ep_name.startswith('master_'):
            try:
                idx = int(ep_name[7:])
            except ValueError:
                raise KeyError(f"Invalid endpoint name '{ep_name}' for AXIMMCrossBarIF")
            if idx >= self.nports_master:
                raise KeyError(
                    f"Master port {idx} is out of range for crossbar with "
                    f"{self.nports_master} master port(s)"
                )
            if not isinstance(endpoint, AXIMMCrossBarIFMaster):
                raise TypeError(
                    "Master sides of AXIMMCrossBarIF must bind to AXIMMCrossBarIFMaster"
                )
            self._validate_and_set_bitwidth(endpoint)
            super().bind(ep_name, endpoint)
            endpoint.master_port = idx

        elif ep_name.startswith('slave_'):
            try:
                idx = int(ep_name[6:])
            except ValueError:
                raise KeyError(f"Invalid endpoint name '{ep_name}' for AXIMMCrossBarIF")
            if idx >= self.nports_slave:
                raise KeyError(
                    f"Slave port {idx} is out of range for crossbar with "
                    f"{self.nports_slave} slave port(s)"
                )
            if not isinstance(endpoint, AXIMMCrossBarIFSlave):
                raise TypeError(
                    "Slave sides of AXIMMCrossBarIF must bind to AXIMMCrossBarIFSlave"
                )
            self._validate_and_set_bitwidth(endpoint)
            super().bind(ep_name, endpoint)

        else:
            raise KeyError(f"Invalid endpoint name '{ep_name}' for AXIMMCrossBarIF")

    # ------------------------------------------------------------------
    # Address decode
    # ------------------------------------------------------------------

    def _decode_address(self, global_addr: int) -> tuple[AXIMMCrossBarIFSlave, int]:
        """Return (slave_ep, local_addr) for *global_addr*, or raise RuntimeError."""
        for ep_name, ep in self.endpoints.items():
            if ep_name.startswith('slave_') and ep is not None:
                if ep.addr_range is not None and ep.addr_range.contains(global_addr):
                    return ep, ep.addr_range.to_local(global_addr)
        raise RuntimeError(
            f"Global address 0x{global_addr:08x} is not within any slave address "
            f"range on crossbar '{self.name}'"
        )

    def _dtype(self) -> np.dtype:
        bw = self.bitwidth if self.bitwidth is not None else 32
        return np.dtype(np.uint32) if bw <= 32 else np.dtype(np.uint64)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def write(self, words: Words, global_addr: int, master_port: int) -> ProcessGen:
        """
        Route a write transaction to the appropriate slave.

        FULL: one burst call after ``(latency_init + nwords)`` cycle delay.
        LITE: ``nwords`` sequential single-word calls each after
              ``latency_per_word`` cycles, with auto-incremented local address.

        Raises
        ------
        RuntimeError
            If *global_addr* is not covered by any slave address range.
        """
        slave_ep, local_addr = self._decode_address(global_addr)

        if slave_ep.protocol == AXIMMProtocol.FULL:
            cycles = self.latency_init + words.shape[0]
            dly = cycles / self.clk.freq
            if dly > 0:
                yield self.timeout(dly)
            with slave_ep.bus.request() as req:
                yield req
                if slave_ep.rx_write_proc is not None:
                    yield self.env.process(slave_ep.rx_write_proc(words, local_addr))

        else:  # LITE — one word per transaction
            nwords = words.shape[0]
            word_bytes = (self.bitwidth if self.bitwidth is not None else 32) // 8
            for i in range(nwords):
                dly = slave_ep.latency_per_word / self.clk.freq
                if dly > 0:
                    yield self.timeout(dly)
                with slave_ep.bus.request() as req:
                    yield req
                    if slave_ep.rx_write_proc is not None:
                        yield self.env.process(
                            slave_ep.rx_write_proc(
                                words[i : i + 1],
                                local_addr + i * word_bytes,
                            )
                        )

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def read(self, nwords: int, global_addr: int, master_port: int) -> ProcessGen:
        """
        Route a read transaction to the appropriate slave and return data.

        The generator return value (``proc.value`` after the caller yields the
        started process) is a numpy array of shape ``(nwords,)``.

        FULL: request wire latency → slave access → return wire latency.
        LITE: ``nwords`` sequential per-word transactions each costing
              ``latency_per_word`` cycles.

        Raises
        ------
        RuntimeError
            If *global_addr* is not covered by any slave address range.
        """
        slave_ep, local_addr = self._decode_address(global_addr)
        dtype = self._dtype()

        if slave_ep.protocol == AXIMMProtocol.FULL:
            # Request leg
            req_dly = self.latency_init / self.clk.freq
            if req_dly > 0:
                yield self.timeout(req_dly)

            with slave_ep.bus.request() as req:
                yield req
                if slave_ep.rx_read_proc is not None:
                    proc = self.env.process(slave_ep.rx_read_proc(nwords, local_addr))
                    yield proc
                    words = proc.value
                else:
                    words = np.zeros(nwords, dtype=dtype)

            # Return leg
            ret_dly = (self.latency_read_return + nwords) / self.clk.freq
            if ret_dly > 0:
                yield self.timeout(ret_dly)

            return words

        else:  # LITE — one word per transaction
            word_bytes = (self.bitwidth if self.bitwidth is not None else 32) // 8
            result = np.zeros(nwords, dtype=dtype)

            for i in range(nwords):
                dly = slave_ep.latency_per_word / self.clk.freq
                if dly > 0:
                    yield self.timeout(dly)
                with slave_ep.bus.request() as req:
                    yield req
                    if slave_ep.rx_read_proc is not None:
                        proc = self.env.process(
                            slave_ep.rx_read_proc(1, local_addr + i * word_bytes)
                        )
                        yield proc
                        result[i] = proc.value[0]

            return result


# ---------------------------------------------------------------------------
# Address assignment utility
# ---------------------------------------------------------------------------

def assign_address_ranges(
    slaves: list[AXIMMCrossBarIFSlave],
    ranges: list[tuple[int, int]],
) -> list[AXIMMAddressRange]:
    """
    Assign byte-address ranges to slave endpoints before simulation starts.

    Parameters
    ----------
    slaves : list[AXIMMCrossBarIFSlave]
        Slave endpoints in port order.
    ranges : list[tuple[int, int]]
        ``(base_addr, size)`` pairs, one per slave.

    Returns
    -------
    list[AXIMMAddressRange]
        The created :class:`AXIMMAddressRange` objects, indexed by slave port.

    Raises
    ------
    ValueError
        If ``len(slaves) != len(ranges)`` or if any two ranges overlap.
    """
    if len(slaves) != len(ranges):
        raise ValueError(
            f"assign_address_ranges: {len(slaves)} slaves but {len(ranges)} ranges"
        )

    addr_ranges = [AXIMMAddressRange(base_addr=b, size=s) for b, s in ranges]

    for i in range(len(addr_ranges)):
        for j in range(i + 1, len(addr_ranges)):
            a, b = addr_ranges[i], addr_ranges[j]
            if a.base_addr < b.base_addr + b.size and b.base_addr < a.base_addr + a.size:
                raise ValueError(
                    f"Address ranges overlap: "
                    f"[0x{a.base_addr:08x}, 0x{a.base_addr + a.size:08x}) and "
                    f"[0x{b.base_addr:08x}, 0x{b.base_addr + b.size:08x})"
                )

    for slave, ar in zip(slaves, addr_ranges):
        slave.addr_range = ar

    return addr_ranges
