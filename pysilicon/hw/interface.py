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



@dataclass
class StreamIF(Interface):
    """
    A stream interface with 'master' (TX) and 'slave' (RX) sides.
    """

    stream_type: StreamType | None  =  None
    """
    The type of stream protocol to use for this interface.
    If None, the stream type will be inferred from the types of 
    endpoints that are bound to this interface.
    """

    bitwidth: int | None = None
    """
    The bitwidth of the stream interface. Must be positive.
    If None, the bitwidth will be inferred from the types of endpoints"""

    notify_type: TransferNotifyType | None = None  
    """
    The method for notifying the receiver side of transfers on 
    this interface. If None, the notify type will be inferred 
    from the slave endpoint.   """

    clk : Clock | None = None
    """
    Clock signal for this stream interface.  Note that the 
    clocks on the endpoints, if any, are not used so they may
    be in a difference clock domain.
    """

    latency_init : float = 0.
    """
    Optional fixed latency in cycles to apply to all transfers 
    on this interface.  The total latency for a transfer will be
    
        latency_init + nwords.
    """

    type_name = 'stream_if'

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        self.endpoint_names = ('master', 'slave')
        super().__post_init__()
        if self.bitwidth is not None and self.bitwidth <= 0:
            raise ValueError("bitwidth must be positive")
        if self.clk is None:
            raise ValueError("clock must be provided for StreamIF")

    def write(
            self, 
            words: Words) -> ProcessGen:
        """
        Write a burst of word to the master (TX) side of this interface.
        This will trigger the RX process on the slave side to process the 
        incoming data.

        Parameters
        ----------
        words : Words
            The block of words to write. 
        """
        if self.endpoints['slave'] is None:
            raise RuntimeError(f"Cannot write to AXI4-Stream interface '{self.name}' because the slave side is not bound to an endpoint")
        slave = self.endpoints['slave']

        # Wait until the end of the burst
        cycles = self.latency_init + words.shape[0]
        dly = cycles / self.clk.freq
        if (dly > 0):
            yield self.timeout(dly)

        with slave.bus.request() as req:

            # Wait for the stream to be available
            yield req

            # Get the number of words
            nwords_rem = words.shape[0]

            # Wait for at least one unit of space in the RX queue
            # to be available
            if (nwords_rem > 0):
                yield slave.nrx.put(1)
                nwords_rem -= 1

            # First, we fill the RX queue
            nwords_rx = min(nwords_rem, slave.nrx.capacity - slave.nrx.level)
            if (nwords_rx > 0):
                yield slave.nrx.put(nwords_rx)
                nwords_rem -= nwords_rx
            
            # If there are remaining words, we fill the TX queue
            if (nwords_rem > 0):
                yield slave.ntx.put(nwords_rem)
                nwords_rem = 0

            # Finally, we write the words to the slave's data buffer
            yield slave.data_buffer.put(words)

       
    def bind(self, ep_name: str, endpoint: InterfaceEndpoint) -> None:
        if ep_name not in ['master', 'slave']:
            raise KeyError(f"Stream interface only has 'master' and 'slave' sides, but got '{ep_name}'")
        if ep_name == "slave" and not isinstance(endpoint, StreamIFSlave):
            raise TypeError("slave side of StreamIF must bind to StreamIFSlave")
        if ep_name == "master" and not isinstance(endpoint, StreamIFMaster):
            raise TypeError("master side of StreamIF must bind to StreamIFMaster")
        if self.stream_type is None:
            self.stream_type = endpoint.stream_type
        if endpoint.stream_type != self.stream_type:
            raise ValueError(
                f"Endpoint stream type {endpoint.stream_type.value} does not match interface stream type {self.stream_type.value}"
            )
        if (ep_name == "slave"):
            if self.notify_type is None:
                self.notify_type = endpoint.notify_type
            if (self.notify_type != endpoint.notify_type):
                raise ValueError(
                    f"Endpoint notify type {endpoint.notify_type.value} does not match interface notify type {self.notify_type.value}"
                )
            if self.notify_type == TransferNotifyType.per_word:
                raise NotImplementedError("per_word notify type is not yet supported")
        if (self.bitwidth is None):
            self.bitwidth = endpoint.bitwidth
        if (self.bitwidth != endpoint.bitwidth):
            raise ValueError(f"Endpoint bitwidth {endpoint.bitwidth} does not match interface bitwidth {self.bitwidth}")
        if (self.stream_type is None):
            self.stream_type = endpoint.stream_type
        if (self.stream_type != endpoint.stream_type):
            raise ValueError(f"Endpoint stream type {endpoint.stream_type.value} does not match interface stream type {self.stream_type.value}")  
        super().bind(ep_name, endpoint)


@dataclass
class StreamIFSlave(InterfaceEndpoint):
    """
    A stream slave (RX) endpoint that is realized as a function call.
    """

    stream_type: StreamType = StreamType.axi4
    """The type of stream protocol to use for this interface."""

    bitwidth: int = 32
    """The bitwidth of the stream interface."""

    notify_type: TransferNotifyType = TransferNotifyType.end_only
    """The method for notifying the receiver side of transfers on 
    this interface. """

    rx_proc : RxProc | None = None
    """
    An optional process to call when words are written to the master (TX) 
    side of the interface. 
    """

    notify_end_proc : Callable[[], ProcessGen] | None = None
    """
    An optional process to call at the end of a transfer when
    `notify_type == begin_end`. This allows the slave to be notified of
    both the beginning and end of a transfer. """

    queue_size : int | None = None
    """
    An optional size for an internal transaction queue that 
    buffers incoming transfers.  Size is in words. If None, the 
    queue is unbounded.
    """

    data_buffer : simpy.Store = field(init=False)
    """
    Buffer for storing incoming bursts of words. Each entry is a 
    burst/block of words, not just a single word.  
    """

    bus : simpy.Resource = field(init=False)
    """
    A resource used to indicate whether a transfer is currently being 
    processed.  The write process will request this resource before writing
    """

    nrx : simpy.Container = field(init=False)
    """Number of pending words in the RX queue"""

    ntx : simpy.Container = field(init=False)
    """Number of pending words in the TX queue."""

    type_name = 'stream_if_slave'


    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        super().__post_init__()
   
        self.data_buffer = simpy.Store(self.env)
        self.bus = simpy.Resource(self.env, capacity=1)
        if self.queue_size is not None:
            capacity = self.queue_size
        else:            
            capacity = float('inf')
        self.nrx = simpy.Container(self.env, init=0,capacity=capacity)
        self.ntx = simpy.Container(self.env, init=0)


    def run_proc(self) -> ProcessGen:
        """
        Continual run loop for the slave to processing incoming transfers
        """
        while True:
            # Wait for the block of words to be available
            words = yield self.data_buffer.get()
            nwords = words.shape[0]
                

            # Update the queue state.  Frst, we pull from the TX queue
            btx = min(nwords, self.ntx.level)
            if (btx > 0):
                yield self.ntx.get(btx)
            
            # Pull from the RX queue for the remaining words
            brx = nwords-btx
            if (brx > 0):
                if (self.nrx.level < brx):
                    raise RuntimeError(f"Not enough words in RX queue to read {nwords} words. RX queue level: {self.nrx.level}, TX queue level: {self.ntx.level}")
                yield self.nrx.get(brx)

            if self.rx_proc is not None:
                yield self.env.process(self.rx_proc(words))
        

@dataclass
class StreamIFMaster(InterfaceEndpoint):
    """
    A stream master (TX) endpoint that provides a write function.
    """

    stream_type: StreamType = StreamType.axi4
    """The type of stream protocol to use for this interface."""

    bitwidth: int = 32
    """The bitwidth of the stream interface."""

    notify_type: TransferNotifyType = TransferNotifyType.end_only
    """
    The method for notifying the receiver side of transfers on 
    this interface. """

    type_name = 'stream_if_master'

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        super().__post_init__()
    

    def write(self, words: Words) -> ProcessGen:
        """
        Write a burst of word to the master (TX) side of this interface.
        The write will block until the slave side has sufficient RX
        queue capacity.

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
            raise RuntimeError(f"Cannot write to AXI4-Stream master endpoint because it is not bound to an interface")

        yield self.process( self.interface.write(words) )


@dataclass
class CrossBarIF(Interface):
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

    bitwidth: int | None = None
    """Data bitwidth. Inferred from the first endpoint bound if ``None``."""

    clk: Clock | None = None
    """Clock signal for this crossbar interface."""

    latency_init: float = 0.
    """Fixed latency in cycles added to every transfer (modelled as ``(latency_init + nwords) / clk.freq`` seconds)."""

    route_fn: Callable[[Words, int], int] | None = None
    """
    Routing function ``(words, port_in) -> port_out``.
    If ``None``, the default mapping ``port_out = port_in % nports_out`` is used.
    """

    type_name = 'crossbar_if'

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        self.endpoint_names = tuple(
            [f'in_{i}' for i in range(self.nports_in)] +
            [f'out_{j}' for j in range(self.nports_out)]
        )
        super().__post_init__()
        if self.nports_in < 1:
            raise ValueError("nports_in must be at least 1")
        if self.nports_out < 1:
            raise ValueError("nports_out must be at least 1")
        if self.bitwidth is not None and self.bitwidth <= 0:
            raise ValueError("bitwidth must be positive")
        if self.clk is None:
            raise ValueError("clock must be provided for CrossBarIF")

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

        port_out = self.route_fn(words, port_in) if self.route_fn is not None else port_in % self.nports_out

        if port_out < 0 or port_out >= self.nports_out:
            raise ValueError(
                f"route_fn returned port_out {port_out} which is out of range [0, {self.nports_out})"
            )

        out_ep = self.endpoints[f'out_{port_out}']
        if out_ep is None:
            raise RuntimeError(f"Output port out_{port_out} is not bound on crossbar '{self.name}'")

        cycles = self.latency_init + words.shape[0]
        dly = cycles / self.clk.freq
        if dly > 0:
            yield self.timeout(dly)

        with out_ep.bus.request() as req:
            yield req

            nwords_rem = words.shape[0]

            if nwords_rem > 0:
                yield out_ep.nrx.put(1)
                nwords_rem -= 1

            nwords_rx = min(nwords_rem, out_ep.nrx.capacity - out_ep.nrx.level)
            if nwords_rx > 0:
                yield out_ep.nrx.put(nwords_rx)
                nwords_rem -= nwords_rx

            if nwords_rem > 0:
                yield out_ep.ntx.put(nwords_rem)

            yield out_ep.data_buffer.put(words)

    def bind(self, ep_name: str, endpoint: InterfaceEndpoint) -> None:
        if ep_name.startswith('in_'):
            try:
                idx = int(ep_name[3:])
            except ValueError:
                raise KeyError(f"Invalid endpoint name '{ep_name}' for CrossBarIF")
            if idx >= self.nports_in:
                raise KeyError(
                    f"Input port index {idx} is out of range for crossbar with {self.nports_in} input(s)"
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
                    f"Output port index {idx} is out of range for crossbar with {self.nports_out} output(s)"
                )
            if not isinstance(endpoint, CrossBarIFOutput):
                raise TypeError("Output sides of CrossBarIF must bind to CrossBarIFOutput")
        else:
            raise KeyError(f"Invalid endpoint name '{ep_name}' for CrossBarIF")

        if self.bitwidth is None:
            self.bitwidth = endpoint.bitwidth
        if self.bitwidth != endpoint.bitwidth:
            raise ValueError(
                f"Endpoint bitwidth {endpoint.bitwidth} does not match interface bitwidth {self.bitwidth}"
            )

        if ep_name.startswith('in_'):
            endpoint.port_in = idx

        super().bind(ep_name, endpoint)


@dataclass
class CrossBarIFInput(InterfaceEndpoint):
    """
    An input endpoint for a :class:`CrossBarIF`.

    An upstream master calls :meth:`write` to push a burst of words into the
    crossbar; the crossbar's routing function determines which output port
    receives the data.
    """

    bitwidth: int = 32
    """Bitwidth of the data words."""

    port_in: int = field(init=False)
    """Index of the input port this endpoint is bound to. Set by :meth:`CrossBarIF.bind`."""

    type_name = 'crossbar_if_input'

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.port_in = -1

    def write(self, words: Words) -> ProcessGen:
        """
        Write a burst of words into the crossbar at this input port.

        Parameters
        ----------
        words : Words
            The block of words to write.

        Example
        -------
        ```
        words = np.array([1, 2, 3], dtype=np.uint32)
        yield env.process( input_ep.write(words) )
        ```
        """
        if self.interface is None:
            raise RuntimeError(
                "Cannot write: CrossBarIFInput is not bound to a CrossBarIF"
            )
        yield self.process(self.interface.write(words, self.port_in))


@dataclass
class CrossBarIFOutput(InterfaceEndpoint):
    """
    An output endpoint for a :class:`CrossBarIF`.

    Words routed to this port are buffered internally and delivered to the
    optional :attr:`rx_proc` callback, mirroring the behaviour of
    :class:`StreamIFSlave`.
    """

    bitwidth: int = 32
    """Bitwidth of the data words."""

    rx_proc: RxProc | None = None
    """Optional process called with each received burst of words."""

    queue_size: int | None = None
    """Optional RX queue depth in words. ``None`` means unbounded."""

    data_buffer: simpy.Store = field(init=False)
    """Buffer holding incoming bursts; each entry is one full burst."""

    bus: simpy.Resource = field(init=False)
    """Serialises concurrent writes from different input ports."""

    nrx: simpy.Container = field(init=False)
    """Number of words pending in the RX queue."""

    ntx: simpy.Container = field(init=False)
    """Number of words pending in the TX (overflow) queue."""

    type_name = 'crossbar_if_output'

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
