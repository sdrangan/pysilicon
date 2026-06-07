from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from waveflow.simulation.simobj import SimObj

if TYPE_CHECKING:
    from waveflow.hw.interface import InterfaceEndpoint

@dataclass
class Component(SimObj):
    """
    Base class for a software or hardware component.
    """

    endpoints: dict[str, InterfaceEndpoint] = \
        field(default_factory=dict)
    """Endpoints of the component, indexed by name."""

    def add_endpoint(self, endpoint: InterfaceEndpoint) -> None:
        endpoint.comp = self
        self.endpoints[endpoint.name] = endpoint