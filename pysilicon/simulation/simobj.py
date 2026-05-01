from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generator
from pysilicon.hw.named import NamedObject

import simpy


ProcessGen = Generator[simpy.events.Event, Any, Any]
ProcessFactory = Callable[[], ProcessGen]


@dataclass(frozen=True)
class ActionRecord(object):
    """Timing record for one action invocation."""

    name: str
    start: float
    end: float


@dataclass(frozen=True)
class ActionOverlap(object):
    """Represents an overlap between two actions on the same object."""

    previous: ActionRecord
    current: ActionRecord

@dataclass
class SimConfig(NamedObject):
    """Configuration parameters for a simulation run."""

    env : simpy.Environment | None = None
    """Shared simulation environment. If None, a new environment 
    will be created."""

    duration: float = 0
    """Simulation duration in seconds. Must be non-negative. 
       Ignored if zero."""

    def __post_init__(self) -> None:
        if self.env is not None and not isinstance(self.env, simpy.Environment):
            raise ValueError("env must be a simpy.Environment or None.")
        if self.duration < 0:
            raise ValueError("duration must be non-negative.")
        
        # Create a new environment if none was provided
        if self.env is None:
            self.env = simpy.Environment()

    

@dataclass
class SimObj(NamedObject):
    """
    Base class for active simulation entities built on top of ``simpy``.

    A ``SimObj`` owns one or more concurrent processes registered with a shared
    ``simpy.Environment``. Subclasses typically register long-running loops that
    consume and produce transactions.
    """

    sim_config: SimConfig | None = None
    """Configuration parameters for the simulation environment."""

    track_action_overlaps: bool = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        """
        Initialize the the process registry
        """
        super().__post_init__()
        if self.sim_config is None:
            raise ValueError("sim_config must be provided.")
        
        self.processes: list[simpy.events.Process] = []
        self._process_factories: list[tuple[str, ProcessFactory]] = []
        self.action_history: list[ActionRecord] = []
        self.action_overlaps: list[ActionOverlap] = []

    @property
    def env(self) -> simpy.Environment:
        """The shared simulation environment."""
        return self.sim_config.env
    
    @property
    def now(self) -> float:
        """Current simulation timestamp in seconds.  """
        return float(self.env.now)

    def timeout(self, delay: float) -> simpy.events.Timeout:
        """Convenience wrapper around ``env.timeout``. 

        Parameters
        ----------
        delay : float
            Time to wait in seconds. Must be non-negative. 
        
        Example
        --------
        
        ```
        yield self.timeout(5)  # wait for 5 seconds
        ```
        """
        if delay < 0:
            raise ValueError("delay must be non-negative.")
        return self.env.timeout(delay)

    def event(self) -> simpy.events.Event:
        """Create a plain SimPy event in the shared environment."""
        return self.env.event()

    def process(self, generator: ProcessGen) -> simpy.events.Process:
        """
        Register and start a process generator in the environment.

        Parameters
        ----------
        generator : Generator
            A SimPy process generator yielding events.
        """
        proc = self.env.process(generator)
        self.processes.append(proc)
        return proc

    def add_process(self, name: str, factory: ProcessFactory, autostart: bool = True) -> None:
        """
        Register a named process factory.

        Parameters
        ----------
        name : str
            Name for introspection/debugging.
        factory : Callable[[], Generator]
            Zero-argument callable that returns a process generator.
        autostart : bool
            If ``True``, the process is started immediately.
        """
        if not name:
            raise ValueError("name must be non-empty.")
        self._process_factories.append((name, factory))
        if autostart:
            self.process(factory())

    def start_registered_processes(self) -> None:
        """Start all registered process factories."""
        for _, factory in self._process_factories:
            self.process(factory())

    def transaction_queue(self, capacity: int | float = float("inf")) -> simpy.Store:
        """
        Create a transaction queue associated with this simulation environment.

        Parameters
        ----------
        capacity : int | float
            Queue capacity. Defaults to unbounded.
        """
        return simpy.Store(self.env, capacity=capacity)

    def resource(self, capacity: int = 1) -> simpy.Resource:
        """Create a shared resource primitive tied to this environment."""
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        return simpy.Resource(self.env, capacity=capacity)

    def container(self, capacity: float, init: float = 0.0) -> simpy.Container:
        """Create a level-based container tied to this environment."""
        return simpy.Container(self.env, capacity=capacity, init=init)

    def action(
        self,
        name: str,
        processing_delay: float = 0.0,
    ) -> ProcessGen:
        """
        Track one action window and optionally model its latency.

        This method is intended to be yielded from inside SimPy processes:
        ``yield from self.action("decode", processing_delay=3)``.

        Parameters
        ----------
        name : str
            Action name.
        processing_delay : float
            Non-negative action delay.
        """
        if not name:
            raise ValueError("name must be non-empty.")
        if processing_delay < 0:
            raise ValueError("processing_delay must be non-negative.")

        start = self.now
        if processing_delay > 0:
            yield self.timeout(processing_delay)
        end = self.now

        current = ActionRecord(name=name, start=start, end=end)
        self._record_action(current)
        return current

    def _record_action(self, current: ActionRecord) -> None:
        self.action_history.append(current)

        if not self.track_action_overlaps:
            return

        for prev in self.action_history[:-1]:
            if prev.end > current.start and current.end > prev.start:
                self.action_overlaps.append(ActionOverlap(previous=prev, current=current))

    def active_overlap_count(self) -> int:
        """Return the number of detected overlapping action windows."""
        return len(self.action_overlaps)

    def clear_action_logs(self) -> None:
        """Clear collected action history and overlap records."""
        self.action_history.clear()
        self.action_overlaps.clear()
