"""Logical interfaces for transmitting serializable objects over physical transports."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import simpy

from pysilicon.hw.dataschema import FloatField, IntField, Words
from pysilicon.hw.interface import Interface, InterfaceEndpoint, StreamIFMaster, StreamIFSlave
from pysilicon.simulation.simobj import ProcessGen


# ---------------------------------------------------------------------------
# Numpy fast-path helpers for ArrayTransferIF
# ---------------------------------------------------------------------------

def _scalar_dtype_for_fast_path(
    element_type: type, word_bw: int
) -> tuple[np.dtype, np.dtype] | None:
    """Return ``(elem_dtype, word_dtype)`` for the numpy fast path, or ``None``.

    Fast path is available for scalar ``FloatField`` / ``IntField`` subclasses
    whose value fits in exactly one word (``nwords_per_inst == 1``).
    """
    if not issubclass(element_type, (FloatField, IntField)):
        return None
    if element_type.nwords_per_inst(word_bw) != 1:
        return None

    word_dtype = np.dtype(np.uint32 if word_bw <= 32 else np.uint64)

    if issubclass(element_type, FloatField):
        bw = element_type.get_bitwidth()
        elem_dtype = np.dtype(np.float32 if bw == 32 else np.float64)
    else:  # IntField
        bw = element_type.get_bitwidth()  # type: ignore[attr-defined]
        signed: bool = element_type.signed  # type: ignore[attr-defined]
        if bw <= 8:
            elem_dtype = np.dtype(np.int8 if signed else np.uint8)
        elif bw <= 16:
            elem_dtype = np.dtype(np.int16 if signed else np.uint16)
        elif bw <= 32:
            elem_dtype = np.dtype(np.int32 if signed else np.uint32)
        else:
            elem_dtype = np.dtype(np.int64 if signed else np.uint64)

    return elem_dtype, word_dtype


def _to_words(arr: np.ndarray, elem_dtype: np.dtype, word_dtype: np.dtype) -> np.ndarray:
    """Convert a numpy element array to a word array for transmission."""
    arr = np.asarray(arr, dtype=elem_dtype)
    if arr.dtype.itemsize == word_dtype.itemsize:
        return arr.view(word_dtype)
    unsigned_same = np.dtype(f'u{arr.dtype.itemsize}')
    return arr.view(unsigned_same).astype(word_dtype)


def _from_words(words: np.ndarray, elem_dtype: np.dtype) -> np.ndarray:
    """Convert a word array back to a numpy element array after reception."""
    if words.dtype.itemsize == elem_dtype.itemsize:
        return words.view(elem_dtype)
    unsigned_same = np.dtype(f'u{elem_dtype.itemsize}')
    return words.astype(unsigned_same).view(elem_dtype)


class PhysicalTransport(ABC):
    """Abstract physical transport: send a word burst and register a receive callback."""

    @abstractmethod
    def write_words(self, words: Words) -> ProcessGen[None]:
        """Transmit a word burst through the physical endpoint."""
        ...

    @abstractmethod
    def set_rx_callback(self, callback: Callable[[Words], ProcessGen[None]]) -> None:
        """Register the callback invoked when a word burst arrives."""
        ...

    @abstractmethod
    def get_words(self) -> ProcessGen[Words]:
        """Pull the next word burst from the physical endpoint (pull model)."""
        ...


@dataclass
class StreamTransport(PhysicalTransport):
    """Physical transport backed by StreamIFMaster / StreamIFSlave."""

    master_ep: StreamIFMaster | None = None
    slave_ep: StreamIFSlave | None = None

    def write_words(self, words: Words) -> ProcessGen[None]:
        yield from self.master_ep.write(words)

    def set_rx_callback(self, callback: Callable[[Words], ProcessGen[None]]) -> None:
        self.slave_ep.rx_proc = callback

    def get_words(self) -> ProcessGen[Words]:
        words = yield from self.slave_ep.get()
        return words


# ---------------------------------------------------------------------------
# SchemaTransferIF
# ---------------------------------------------------------------------------

@dataclass
class SchemaTransferIFMaster(InterfaceEndpoint):
    """Master endpoint: serializes any object and forwards words to the transport."""

    transport: PhysicalTransport | None = None
    bitwidth: int = 32

    def write(self, obj: Any) -> ProcessGen[None]:
        """Serialize *obj* and transmit through the transport."""
        yield from self.transport.write_words(obj.serialize(word_bw=self.bitwidth))


@dataclass
class SchemaTransferIFSlave(InterfaceEndpoint):
    """Slave endpoint: receives word bursts and deserializes them via *schema_type*.

    Two receive modes are supported:

    **Push mode** (default, ``pull_mode=False``):
        Call ``pre_sim()`` to register a callback on the transport.  The
        transport's ``run_proc`` loop delivers deserialized objects via
        :attr:`rx_proc` and/or :attr:`queue`.

    **Pull mode** (``pull_mode=True``):
        Do not call ``pre_sim()``, or call it with ``pull_mode=True`` at
        construction.  The owning component drives sequencing by calling
        ``yield from slave.get()`` directly.
    """

    transport: PhysicalTransport | None = None
    schema_type: type | None = None
    bitwidth: int = 32
    rx_proc: Callable[[Any], ProcessGen[None]] | None = None
    pull_mode: bool = False
    queue: simpy.Store = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.queue = simpy.Store(self.env)

    def pre_sim(self) -> None:
        if not self.pull_mode:
            self.transport.set_rx_callback(self._on_words_received)

    def _on_words_received(self, words: Words) -> ProcessGen[None]:
        obj = self.schema_type().deserialize(words, word_bw=self.bitwidth)
        yield self.queue.put(obj)
        if self.rx_proc is not None:
            yield self.env.process(self.rx_proc(obj))

    def get(self) -> ProcessGen[Any]:
        """Pull and deserialize the next burst (pull model).

        Returns the deserialized schema instance.  Does not touch :attr:`queue`.
        """
        words = yield from self.transport.get_words()
        return self.schema_type().deserialize(words, word_bw=self.bitwidth)


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


# ---------------------------------------------------------------------------
# ArrayTransferIF
# ---------------------------------------------------------------------------

@dataclass
class ArrayTransferIFMaster(InterfaceEndpoint):
    """Master endpoint: serializes an iterable of *element_type* instances and
    transmits them as a single word burst through the transport.

    Each element is serialized individually; the resulting words are
    concatenated into one burst (one AXI-Stream packet, TLAST at end).
    Elements may be schema instances or raw Python values accepted by
    ``element_type(value)``.
    """

    transport: PhysicalTransport | None = None
    element_type: type | None = None
    bitwidth: int = 32

    def write(self, elements) -> ProcessGen[None]:
        """Serialize *elements* and transmit as one burst.

        Parameters
        ----------
        elements:
            ``np.ndarray`` of the element's native numpy dtype (fast path), or
            an iterable of ``element_type`` instances / raw values accepted by
            ``element_type(value)``.
        """
        fast = _scalar_dtype_for_fast_path(self.element_type, self.bitwidth)
        if fast is not None and isinstance(elements, np.ndarray):
            elem_dtype, word_dtype = fast
            all_words = _to_words(elements, elem_dtype, word_dtype)
            yield from self.transport.write_words(all_words)
            return
        nwords_per_elem = self.element_type.nwords_per_inst(self.bitwidth)
        items = list(elements)
        dtype = np.uint32 if self.bitwidth <= 32 else np.uint64
        all_words = np.empty(len(items) * nwords_per_elem, dtype=dtype)
        for i, raw in enumerate(items):
            elem = raw if isinstance(raw, self.element_type) else self.element_type(raw)
            elem_words = elem.serialize(word_bw=self.bitwidth)
            all_words[i * nwords_per_elem:(i + 1) * nwords_per_elem] = elem_words
        yield from self.transport.write_words(all_words)


@dataclass
class ArrayTransferIFSlave(InterfaceEndpoint):
    """Slave endpoint: receives a word burst and deserializes it into a list of
    *element_type* instances.

    Two receive modes are supported (same semantics as :class:`SchemaTransferIFSlave`):

    **Push mode** (``pull_mode=False``, default):
        Register via ``pre_sim()``; bursts are delivered to :attr:`rx_proc`
        with the element count inferred from burst length.

    **Pull mode** (``pull_mode=True``):
        The owning component calls ``yield from slave.get(count=n)``; an exact
        element count is required and TLAST position is validated.
    """

    transport: PhysicalTransport | None = None
    element_type: type | None = None
    bitwidth: int = 32
    rx_proc: Callable[[list], ProcessGen[None]] | None = None
    pull_mode: bool = False

    def pre_sim(self) -> None:
        if not self.pull_mode:
            self.transport.set_rx_callback(self._on_words_received)

    def _deserialize(self, words: Words, count: int) -> np.ndarray | list:
        fast = _scalar_dtype_for_fast_path(self.element_type, self.bitwidth)
        if fast is not None:
            elem_dtype, _ = fast
            return _from_words(words, elem_dtype)
        nwords_per_elem = self.element_type.nwords_per_inst(self.bitwidth)
        return [
            self.element_type().deserialize(
                words[i * nwords_per_elem:(i + 1) * nwords_per_elem],
                word_bw=self.bitwidth,
            )
            for i in range(count)
        ]

    def _on_words_received(self, words: Words) -> ProcessGen[None]:
        nwords_per_elem = self.element_type.nwords_per_inst(self.bitwidth)
        count = words.shape[0] // nwords_per_elem
        elements = self._deserialize(words, count)
        if self.rx_proc is not None:
            yield self.env.process(self.rx_proc(elements))

    def get(self, count: int) -> ProcessGen[np.ndarray | list]:
        """Pull and deserialize exactly *count* elements (pull model).

        For scalar ``FloatField`` / ``IntField`` element types, returns a
        ``np.ndarray`` of the element's native dtype (e.g. ``np.float32``).
        For composite types, returns a ``list`` of deserialized instances.

        Validates that the burst contains precisely ``count *
        element_type.nwords_per_inst(bitwidth)`` words, matching the C++
        TLAST contract (TLAST_EARLY / NO_TLAST errors).
        """
        nwords_per_elem = self.element_type.nwords_per_inst(self.bitwidth)
        expected = count * nwords_per_elem
        words = yield from self.transport.get_words()
        actual = words.shape[0]
        if actual < expected:
            raise RuntimeError(
                f"TLAST early: expected {expected} words for {count} "
                f"{self.element_type.__name__} elements, got {actual}"
            )
        if actual > expected:
            raise RuntimeError(
                f"Missing TLAST: expected {expected} words for {count} "
                f"{self.element_type.__name__} elements, got {actual}"
            )
        return self._deserialize(words, count)


@dataclass
class ArrayTransferIF(Interface):
    """Logical interface container for ArrayTransferIFMaster / ArrayTransferIFSlave."""

    def __post_init__(self) -> None:
        self.endpoint_names = ('master', 'slave')
        super().__post_init__()

    def bind(self, ep_name: str, endpoint: InterfaceEndpoint) -> None:
        if ep_name not in ('master', 'slave'):
            raise KeyError(
                f"ArrayTransferIF only has 'master' and 'slave' sides, "
                f"but got '{ep_name}'"
            )
        if ep_name == 'master' and not isinstance(endpoint, ArrayTransferIFMaster):
            raise TypeError(
                "master side of ArrayTransferIF must bind to ArrayTransferIFMaster"
            )
        if ep_name == 'slave' and not isinstance(endpoint, ArrayTransferIFSlave):
            raise TypeError(
                "slave side of ArrayTransferIF must bind to ArrayTransferIFSlave"
            )
        other_name = 'slave' if ep_name == 'master' else 'master'
        other = self.endpoints[other_name]
        if other is not None:
            if other.element_type != endpoint.element_type:
                raise ValueError(
                    f"element_type mismatch: master has {other.element_type.__name__}, "
                    f"slave has {endpoint.element_type.__name__}"
                )
            if other.bitwidth != endpoint.bitwidth:
                raise ValueError(
                    f"bitwidth mismatch: existing endpoint has {other.bitwidth}, "
                    f"new endpoint has {endpoint.bitwidth}"
                )
        super().bind(ep_name, endpoint)
