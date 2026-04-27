from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray

from pysilicon.hw.component import Component
from pysilicon.hw.named import NamedObject


class InterfaceEndpoint:
    """
    Base class for a concrete endpoint owned by a component.

    Attributes:
    -----------
    comp : Component
        The component that owns this endpoint. Set when 
        the endpoint is added to a component.
    interface : Interface
        The interface this endpoint is bound to, if any. 
        Set when the endpoint is bound to an interface.
    """
    def __init__(self):
        self.comp: Component | None = None
        self.interface: Interface | None = None

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


class Interface(NamedObject):
    """
    Base class for an interface with a fixed set of named sides.

    Attributes:
    -----------
    name : str
        The name of this interface, unique within the system.
    endpoints : dict[str, InterfaceEndpoint | None]
        The mapping of interface side names to the endpoints 
        bound to them.
    """
    def __init__(self, name: str | None, ep_names: list[str]):
        NamedObject.__init__(self, name)
        self.endpoints: dict[str, InterfaceEndpoint | None] = {}
        for ep_name in ep_names:
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
    
class StreamIF(Interface):
    """
    A stream interface with 'master' (TX) and 'slave' (RX) sides.
    """

    def __init__(
            self, 
            name: str | None = None,
            stream_type: StreamType = StreamType.axi4,
            bitwidth: int = 32):
        super().__init__(name, ep_names=['master', 'slave'])
        self.stream_type = stream_type
        self.bitwidth = bitwidth

    def write(self, 
              words: NDArray[np.uint32] | NDArray[np.uint64]) -> None:
        """
        Write a burst of word to the master (TX) side of this interface.

        If bitwidth <= 32, words should be an (n,) array of uint32. 
        If bitwidth <= 64, words should be an (n,) array of uint64.
        If bitwidth > 64, words should be an (n,k) array of uint64 
        where k = ceil(bitwidth / 64) and each word is 
        represented in little-endian order.
        """
        if self.endpoints['slave'] is None:
            raise RuntimeError(f"Cannot write to AXI4-Stream interface '{self.name}' because the slave side is not bound to an endpoint")
        self.endpoints['slave'].rx_fun(words)

    def bind(self, ep_name: str, endpoint: InterfaceEndpoint) -> None:
        if ep_name not in ['master', 'slave']:
            raise KeyError(f"Stream interface only has 'master' and 'slave' sides, but got '{ep_name}'")
        if ep_name == "slave" and not isinstance(endpoint, StreamIFSlave):
            raise TypeError("slave side of StreamIF must bind to StreamIFSlave")
        if ep_name == "master" and not isinstance(endpoint, StreamIFMaster):
            raise TypeError("master side of StreamIF must bind to StreamIFMaster")
        if endpoint.stream_type != self.stream_type:
            raise ValueError(
                f"Endpoint stream type {endpoint.stream_type.value} does not match interface stream type {self.stream_type.value}"
            )
        
        if endpoint.bitwidth != self.bitwidth:
            raise ValueError(f"Endpoint bitwidth {endpoint.bitwidth} does not match interface bitwidth {self.bitwidth}")
        super().bind(ep_name, endpoint)

class StreamIFSlave(InterfaceEndpoint):
    """
    A stream slave (RX) endpoint that is realized as a function call.

    Parameters
    ----------
    rx_fun : callable[[NDArray[np.uint64] | NDArray[np.uint32]], None]
        The function to call when words are written to the master (TX) side of the interface. 
        The function should take a single argument which is an (n,) or (n,k) 
        array of uint64 containing the words that were written.
    bitwidth : int
        The bitwidth of the AXI4-Stream interface.
    """
    def __init__(
            self, 
            rx_fun: 
                callable[[NDArray[np.uint64] | NDArray[np.uint32]], None],
            stream_type: StreamType = StreamType.axi4,
            bitwidth: int = 32):
        self.rx_fun = rx_fun
        self.stream_type = stream_type
        self.bitwidth = bitwidth
        super().__init__()

class StreamIFMaster(InterfaceEndpoint):
    """
    A stream master (TX) endpoint that provides a write function.

    Parameters
    ----------
    bitwidth : int
        The bitwidth of the stream interface.
    """
    def __init__(
            self, 
            stream_type: StreamType = StreamType.axi4,
            bitwidth: int = 32):
        self.stream_type = stream_type
        self.bitwidth = bitwidth
        super().__init__()

    def write(self, 
              words: NDArray[np.uint32] | NDArray[np.uint64]) -> None:
        """
        Write a burst of word to the master (TX) side of this interface.

        If bitwidth <= 32, words should be an (n,) array of uint32. 
        If bitwidth <= 64, words should be an (n,) array of uint64.
        If bitwidth > 64, words should be an (n,k) array of uint64 
        where k = ceil(bitwidth / 64) and each word is 
        represented in little-endian order.
        """
        if self.interface is None:
            raise RuntimeError(f"Cannot write to AXI4-Stream master endpoint because it is not bound to an interface")
        self.interface.write(words)