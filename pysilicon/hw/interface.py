"""
interface.py: Hardware interfaces and endpoints (graph + simulation hooks).

Type aliases
-----------

Words
    Type alias for representing a burst/block of words over a fixed-bitwidth
    stream interface.

    - If bitwidth <= 32, words should be an (n,) array of uint32.
    - If bitwidth <= 64, words should be an (n,) array of uint64.
    - If bitwidth > 64, words should be an (n, k) array of uint64 where
      k = ceil(bitwidth / 64) and each word is represented in little-endian
      order.

RxProc
    Type alias for a SimPy process function that receives a block of words.
    The callable returns a ``ProcessGen`` (generator yielding SimPy events)
    and is typically started using ``env.process(rx_proc(words))``.

Example
-------
```python
class Adder(SimObj):
    ...
    def proc(self, words: Words) -> ProcessGen:
        n = words.shape[0]
        total = np.sum(words)
        proc_time = 0.1 * n
        yield self.timeout(proc_time)

adder = Adder(...)
adder_proc: RxProc = adder.proc

words = np.array([1, 2, 3], dtype=np.uint32)
env.process(adder_proc(words))
```
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Generator
import simpy

import numpy as np
from numpy.typing import NDArray

from pysilicon.hw.component import Component
from pysilicon.hw.named import NamedObject
from pysilicon.hw.clock import Clock
from pysilicon.simulation.simobj import SimObj, ProcessGen

"""Type aliases for interface definitions."""
Words = TypeAlias = NDArray[np.uint32] | NDArray[np.uint64]
RxProc = TypeAlias = Callable[[Words], ProcessGen]

@dataclass
class InterfaceEndpoint(SimObj):
    """
    Base class for a concrete endpoint owned by a component.
    """

    comp : Component | None = field(init=False)
    """The component that owns this endpoint.
    Set when the endpoint is added to a component.
    """

    interface : Interface | None = field(init=False)
    """
    The interface this endpoint is bound to, if any.
    Set when the endpoint is bound to an interface.
    """
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.comp = None
        self.interface = None


    def bind(self, interface: Interface, ep_name: str) -> None:
        """
        Bind this endpoint to a side of an interface.
        Parameters:
        -----------
        interface : Interface
            The interface to bind to.
        ep_name : str
            The name of the interface side to bind to.
        """
        interface.bind(ep_name, self)


@dataclass
class Interface(SimObj):
    """
    Base class for an interface with a set of named sides.
    """

    endpoint_names: tuple[str, ...] = field(init=False)
    endpoints: dict[str, InterfaceEndpoint | None] = field(init=False)
    """
    Dictionary mapping valid endpoint names to the currently
    bound endpoint (or None if unbound)."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        """
        Initialize the endpoints dictionary from the instance endpoint names.
        """
        super().__post_init__()
        self.endpoints = {}
        if not hasattr(self, "endpoint_names") or not self.endpoint_names:
            raise ValueError("endpoint_names must be defined before Interface.__post_init__")
        for ep_name in self.endpoint_names:
            self.endpoints[ep_name] = None

    def bind(self, ep_name: str, endpoint: InterfaceEndpoint) -> None:
        """
        Binds an endpoint to a side of this interface.
        """
        if ep_name not in self.endpoints:
            raise KeyError(f"Interface side '{ep_name}' is not valid for interface '{self.name}'")
        if self.endpoints[ep_name] is not None:
            raise ValueError(f"Interface side '{ep_name}' is already bound on interface '{self.name}'")

        self.endpoints[ep_name] = endpoint
        endpoint.interface = self

class StreamType(Enum):
    hls = "hls"
    axi4 = "axi4"

class TransferNotifyType(Enum):
    """
    Methods for notifying the receiver side of interface during
    a transfer.

    - end_only:  Only notify at the end of a transfer.
    This mode is ideal when the RX sides only begins processing after the
    full data burst is received.

    - begin_end: Notify at both the beginning and end of a transfer.
    This mode is ideal when the RX side can begin processing as
    soon as the RX starts.

    - per_word: Notify for every beat/word transferred. This mode is
    the most cycle accurate but also the most expensive to simulate.
    The mode will not be initially supported.
    """
    end_only = "end_only"
    begin_end = "begin_end"
    per_word = "per_word"


# ---------------------------------------------------------------------------
# Shared base classes
# ---------------------------------------------------------------------------

@dataclass
class QueuedTransferIFSlave(InterfaceEndpoint):
    """
    Base class for slave/output endpoints that buffer incoming word bursts.

    Provides the SimPy resources (``data_buffer``, ``bus``, ``nrx``, ``ntx``)
    and the ``run_proc`` loop shared by :class:`StreamIFSlave` and
    :class:`CrossBarIFOutput`.  Protocol-specific fields (stream type, notify
    type, etc.) are added in subclasses.
    """

    bitwidth: int = 32
    """Bitwidth of the data words."""

    rx_proc: RxProc | None = None
    """Optional process called with each received burst of words."""

    queue_size: int | None = None
    """RX queue depth in words.  ``None`` means unbounded."""

    data_buffer: simpy.Store = field(init=False)
    """Buffer holding incoming bursts; each entry is one full burst."""

    bus: simpy.Resource = field(init=False)
    """Serialises concurrent writes to this endpoint."""

    nrx: simpy.Container = field(init=False)
    """Number of words pending in the RX queue."""

    ntx: simpy.Container = field(init=False)
    """Number of words pending in the TX (overflow) queue."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.data_buffer = simpy.Store(self.env)
        self.bus = simpy.Resource(self.env, capacity=1)
        capacity = self.queue_size if self.queue_size is not None else float('inf')
        self.nrx = simpy.Container(self.env, init=0, capacity=capacity)
        self.ntx = simpy.Container(self.env, init=0)

    def run_proc(self) -> ProcessGen:
        """Continually processes incoming data bursts and invokes :attr:`rx_proc`."""
        while True:
            words = yield self.data_buffer.get()
            nwords = words.shape[0]

            btx = min(nwords, self.ntx.level)
            if btx > 0:
                yield self.ntx.get(btx)

            brx = nwords - btx
            if brx > 0:
                if self.nrx.level < brx:
                    raise RuntimeError(
                        f"Not enough words in RX queue to read {nwords} words. "
                        f"RX queue level: {self.nrx.level}, TX queue level: {self.ntx.level}"
                    )
                yield self.nrx.get(brx)

            if self.rx_proc is not None:
                yield self.env.process(self.rx_proc(words))


@dataclass
class QueuedTransferIFMaster(InterfaceEndpoint):
    """
    Base class for master/input endpoints that push word bursts into an interface.

    Provides a ``write`` method that dispatches to the bound interface.
    Subclasses override :meth:`_make_write_call` when the interface's
    ``write`` signature differs (e.g. :class:`CrossBarIFInput` passes
    ``port_in``).
    """

    bitwidth: int = 32
    """Bitwidth of the data words."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def _make_write_call(self, words: Words) -> ProcessGen:
        """Return the generator that writes *words* to the bound interface."""
        return self.interface.write(words)

    def write(self, words: Words) -> ProcessGen:
        """
        Write a burst of words through the bound interface.

        Parameters
        ----------
        words : Words
            The block of words to write.

        Example
        -------
        ```
        words = np.array([1, 2, 3], dtype=np.uint32)
        yield env.process( master_ep.write(words) )
        ```
        """
        if self.interface is None:
            raise RuntimeError(
                f"Cannot write: {type(self).__name__} is not bound to an interface"
            )
        yield self.process(self._make_write_call(words))


@dataclass
class QueuedTransferIF(Interface):
    """
    Base class for interfaces that transfer queued word bursts with optional latency.

    Provides shared ``bitwidth``, ``clk``, and ``latency_init`` fields, a
    ``_push_to_endpoint`` generator for the common write path, and
    ``_validate_and_set_bitwidth`` for bind-time checking.  Protocol-specific
    routing and endpoint-type validation are implemented in subclasses.
    """

    bitwidth: int | None = None
    """
    Data bitwidth.  Inferred from the first bound endpoint when ``None``.
    """

    clk: Clock | None = None
    """Clock signal for this interface."""

    latency_init: float = 0.
    """
    Fixed latency in cycles added to every transfer.  Total transfer time is
    ``(latency_init + nwords) / clk.freq`` seconds.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        # endpoint_names must be set by the concrete subclass before calling super()
        super().__post_init__()
        if self.bitwidth is not None and self.bitwidth <= 0:
            raise ValueError("bitwidth must be positive")
        if self.clk is None:
            raise ValueError(f"clock must be provided for {type(self).__name__}")

    def _validate_and_set_bitwidth(self, endpoint: InterfaceEndpoint) -> None:
        """Infer or check that *endpoint* bitwidth matches the interface bitwidth."""
        if self.bitwidth is None:
            self.bitwidth = endpoint.bitwidth
        if self.bitwidth != endpoint.bitwidth:
            raise ValueError(
                f"Endpoint bitwidth {endpoint.bitwidth} does not match "
                f"interface bitwidth {self.bitwidth}"
            )

    def _push_to_endpoint(
        self, ep: QueuedTransferIFSlave, words: Words
    ) -> ProcessGen:
        """
        Model transfer latency then push *words* into *ep*'s buffer.

        This is the shared write path used by all ``QueuedTransferIF`` subclasses.
        """
        cycles = self.latency_init + words.shape[0]
        dly = cycles / self.clk.freq
        if dly > 0:
            yield self.timeout(dly)

        with ep.bus.request() as req:
            yield req

            nwords_rem = words.shape[0]

            if nwords_rem > 0:
                yield ep.nrx.put(1)
                nwords_rem -= 1

            nwords_rx = min(nwords_rem, ep.nrx.capacity - ep.nrx.level)
            if nwords_rx > 0:
                yield ep.nrx.put(nwords_rx)
                nwords_rem -= nwords_rx

            if nwords_rem > 0:
                yield ep.ntx.put(nwords_rem)

            yield ep.data_buffer.put(words)


# ---------------------------------------------------------------------------
# Stream interface
# ---------------------------------------------------------------------------

@dataclass
class StreamIF(QueuedTransferIF):
    """
    A stream interface with 'master' (TX) and 'slave' (RX) sides.
    """

    stream_type: StreamType | None = None
    """
    The type of stream protocol to use for this interface.
    If None, the stream type will be inferred from the types of
    endpoints that are bound to this interface.
    """

    notify_type: TransferNotifyType | None = None
    """
    The method for notifying the receiver side of transfers on
    this interface. If None, the notify type will be inferred
    from the slave endpoint.
    """

    type_name = 'stream_if'

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        self.endpoint_names = ('master', 'slave')
        super().__post_init__()

    def write(self, words: Words) -> ProcessGen:
        """
        Write a burst of words to the master (TX) side of this interface.
        This will trigger the RX process on the slave side to process the
        incoming data.

        Parameters
        ----------
        words : Words
            The block of words to write.
        """
        if self.endpoints['slave'] is None:
            raise RuntimeError(
                f"Cannot write to StreamIF '{self.name}' because the slave side is not bound"
            )
        slave = self.endpoints['slave']
        yield from self._push_to_endpoint(slave, words)

    def bind(self, ep_name: str, endpoint: InterfaceEndpoint) -> None:
        if ep_name not in ('master', 'slave'):
            raise KeyError(
                f"Stream interface only has 'master' and 'slave' sides, but got '{ep_name}'"
            )
        if ep_name == "slave" and not isinstance(endpoint, StreamIFSlave):
            raise TypeError("slave side of StreamIF must bind to StreamIFSlave")
        if ep_name == "master" and not isinstance(endpoint, StreamIFMaster):
            raise TypeError("master side of StreamIF must bind to StreamIFMaster")
        if self.stream_type is None:
            self.stream_type = endpoint.stream_type
        if endpoint.stream_type != self.stream_type:
            raise ValueError(
                f"Endpoint stream type {endpoint.stream_type.value} does not match "
                f"interface stream type {self.stream_type.value}"
            )
        if ep_name == "slave":
            if self.notify_type is None:
                self.notify_type = endpoint.notify_type
            if self.notify_type != endpoint.notify_type:
                raise ValueError(
                    f"Endpoint notify type {endpoint.notify_type.value} does not match "
                    f"interface notify type {self.notify_type.value}"
                )
            if self.notify_type == TransferNotifyType.per_word:
                raise NotImplementedError("per_word notify type is not yet supported")
        self._validate_and_set_bitwidth(endpoint)
        super().bind(ep_name, endpoint)


@dataclass
class StreamIFSlave(QueuedTransferIFSlave):
    """
    A stream slave (RX) endpoint that is realized as a function call.
    """

    stream_type: StreamType = StreamType.axi4
    """The type of stream protocol to use for this interface."""

    notify_type: TransferNotifyType = TransferNotifyType.end_only
    """The method for notifying the receiver side of transfers on this interface."""

    notify_end_proc: Callable[[], ProcessGen] | None = None
    """
    An optional process to call at the end of a transfer when
    `notify_type == begin_end`. This allows the slave to be notified of
    both the beginning and end of a transfer.
    """

    type_name = 'stream_if_slave'

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)


@dataclass
class StreamIFMaster(QueuedTransferIFMaster):
    """
    A stream master (TX) endpoint that provides a write function.
    """

    stream_type: StreamType = StreamType.axi4
    """The type of stream protocol to use for this interface."""

    notify_type: TransferNotifyType = TransferNotifyType.end_only
    """
    The method for notifying the receiver side of transfers on
    this interface.
    """

    type_name = 'stream_if_master'

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)


# ---------------------------------------------------------------------------
# Crossbar interface
# ---------------------------------------------------------------------------

@dataclass
class CrossBarIF(QueuedTransferIF):
    """
    A crossbar interface connecting ``nports_in`` input ports to ``nports_out``
    output ports.  Each transfer arriving at an input port is routed to exactly
    one output port via a configurable ``route_fn``.

    Endpoint names
    --------------
    ``'in_0'``, ``'in_1'``, … ``'in_{nports_in-1}'``  — bind :class:`CrossBarIFInput`
    ``'out_0'``, ``'out_1'``, … ``'out_{nports_out-1}'`` — bind :class:`CrossBarIFOutput`
    """

    nports_in: int = 2
    """Number of input ports."""

    nports_out: int = 2
    """Number of output ports."""

    route_fn: Callable[[Words, int], int] | None = None
    """
    Routing function ``(words, port_in) -> port_out``.
    If ``None``, the default mapping ``port_out = port_in % nports_out`` is used.
    """

    type_name = 'crossbar_if'

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        # Validate port counts before building endpoint_names (used by super().__post_init__)
        if self.nports_in < 1:
            raise ValueError("nports_in must be at least 1")
        if self.nports_out < 1:
            raise ValueError("nports_out must be at least 1")
        self.endpoint_names = tuple(
            [f'in_{i}' for i in range(self.nports_in)] +
            [f'out_{j}' for j in range(self.nports_out)]
        )
        super().__post_init__()

    def write(self, words: Words, port_in: int) -> ProcessGen:
        """
        Route words arriving at ``port_in`` to the appropriate output port.

        Parameters
        ----------
        words : Words
            The block of words to transfer.
        port_in : int
            Index of the input port the words arrived on.
        """
        if port_in < 0 or port_in >= self.nports_in:
            raise ValueError(f"port_in {port_in} out of range [0, {self.nports_in})")

        port_out = (
            self.route_fn(words, port_in)
            if self.route_fn is not None
            else port_in % self.nports_out
        )

        if port_out < 0 or port_out >= self.nports_out:
            raise ValueError(
                f"route_fn returned port_out {port_out} which is out of range "
                f"[0, {self.nports_out})"
            )

        out_ep = self.endpoints[f'out_{port_out}']
        if out_ep is None:
            raise RuntimeError(
                f"Output port out_{port_out} is not bound on crossbar '{self.name}'"
            )

        yield from self._push_to_endpoint(out_ep, words)

    def bind(self, ep_name: str, endpoint: InterfaceEndpoint) -> None:
        if ep_name.startswith('in_'):
            try:
                idx = int(ep_name[3:])
            except ValueError:
                raise KeyError(f"Invalid endpoint name '{ep_name}' for CrossBarIF")
            if idx >= self.nports_in:
                raise KeyError(
                    f"Input port index {idx} is out of range for crossbar with "
                    f"{self.nports_in} input(s)"
                )
            if not isinstance(endpoint, CrossBarIFInput):
                raise TypeError("Input sides of CrossBarIF must bind to CrossBarIFInput")
        elif ep_name.startswith('out_'):
            try:
                idx = int(ep_name[4:])
            except ValueError:
                raise KeyError(f"Invalid endpoint name '{ep_name}' for CrossBarIF")
            if idx >= self.nports_out:
                raise KeyError(
                    f"Output port index {idx} is out of range for crossbar with "
                    f"{self.nports_out} output(s)"
                )
            if not isinstance(endpoint, CrossBarIFOutput):
                raise TypeError("Output sides of CrossBarIF must bind to CrossBarIFOutput")
        else:
            raise KeyError(f"Invalid endpoint name '{ep_name}' for CrossBarIF")

        self._validate_and_set_bitwidth(endpoint)

        if ep_name.startswith('in_'):
            endpoint.port_in = idx

        super().bind(ep_name, endpoint)


@dataclass
class CrossBarIFInput(QueuedTransferIFMaster):
    """
    An input endpoint for a :class:`CrossBarIF`.

    An upstream master calls :meth:`write` to push a burst of words into the
    crossbar; the crossbar's routing function determines which output port
    receives the data.
    """

    port_in: int = field(init=False)
    """Index of the input port this endpoint is bound to. Set by :meth:`CrossBarIF.bind`."""

    type_name = 'crossbar_if_input'

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.port_in = -1

    def _make_write_call(self, words: Words) -> ProcessGen:
        return self.interface.write(words, self.port_in)


@dataclass
class CrossBarIFOutput(QueuedTransferIFSlave):
    """
    An output endpoint for a :class:`CrossBarIF`.

    Words routed to this port are buffered internally and delivered to the
    optional :attr:`rx_proc` callback, mirroring the behaviour of
    :class:`StreamIFSlave`.
    """

    type_name = 'crossbar_if_output'

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
