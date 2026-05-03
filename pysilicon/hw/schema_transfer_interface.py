"""Logical interface for transmitting serializable objects over physical transports."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import simpy

from pysilicon.hw.dataschema import Words
from pysilicon.hw.interface import Interface, InterfaceEndpoint, StreamIFMaster, StreamIFSlave
from pysilicon.simulation.simobj import ProcessGen


class PhysicalTransport(ABC):
    """Abstract physical transport: send a word burst and register a receive callback."""

    @abstractmethod
    def write_words(self, words: Words) -> ProcessGen:
        """Transmit a word burst through the physical endpoint."""
        ...

    @abstractmethod
    def set_rx_callback(self, callback: Callable[[Words], ProcessGen]) -> None:
        """Register the callback invoked when a word burst arrives."""
        ...


@dataclass
class StreamTransport(PhysicalTransport):
    """Physical transport backed by StreamIFMaster / StreamIFSlave."""

    master_ep: StreamIFMaster | None = None
    slave_ep: StreamIFSlave | None = None

    def write_words(self, words: Words) -> ProcessGen:
        yield from self.master_ep.write(words)

    def set_rx_callback(self, callback: Callable[[Words], ProcessGen]) -> None:
        self.slave_ep.rx_proc = callback


@dataclass
class SchemaTransferIFMaster(InterfaceEndpoint):
    """Master endpoint: serializes any object and forwards words to the transport."""

    transport: PhysicalTransport | None = None
    bitwidth: int = 32

    def write(self, obj: Any) -> ProcessGen:
        """Serialize *obj* and transmit through the transport."""
        yield from self.transport.write_words(obj.serialize(word_bw=self.bitwidth))


@dataclass
class SchemaTransferIFSlave(InterfaceEndpoint):
    """Slave endpoint: receives word bursts and deserializes them via *schema_type*."""

    transport: PhysicalTransport | None = None
    schema_type: type | None = None
    bitwidth: int = 32
    rx_proc: Callable[[Any], ProcessGen] | None = None
    queue: simpy.Store = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.queue = simpy.Store(self.env)

    def pre_sim(self) -> None:
        self.transport.set_rx_callback(self._on_words_received)

    def _on_words_received(self, words: Words) -> ProcessGen:
        obj = self.schema_type().deserialize(words, word_bw=self.bitwidth)
        yield self.queue.put(obj)
        if self.rx_proc is not None:
            yield self.env.process(self.rx_proc(obj))


@dataclass
class SchemaTransferIF(Interface):
    """Logical interface container for SchemaTransferIFMaster / SchemaTransferIFSlave."""

    def __post_init__(self) -> None:
        self.endpoint_names = ('master', 'slave')
        super().__post_init__()

    def bind(self, ep_name: str, endpoint: InterfaceEndpoint) -> None:
        if ep_name not in ('master', 'slave'):
            raise KeyError(
                f"SchemaTransferIF only has 'master' and 'slave' sides, "
                f"but got '{ep_name}'"
            )
        if ep_name == 'master' and not isinstance(endpoint, SchemaTransferIFMaster):
            raise TypeError(
                "master side of SchemaTransferIF must bind to SchemaTransferIFMaster"
            )
        if ep_name == 'slave' and not isinstance(endpoint, SchemaTransferIFSlave):
            raise TypeError(
                "slave side of SchemaTransferIF must bind to SchemaTransferIFSlave"
            )
        other_name = 'slave' if ep_name == 'master' else 'master'
        other = self.endpoints[other_name]
        if other is not None and other.bitwidth != endpoint.bitwidth:
            raise ValueError(
                f"Endpoint bitwidth {endpoint.bitwidth} does not match "
                f"existing {other_name} bitwidth {other.bitwidth}"
            )
        super().bind(ep_name, endpoint)
