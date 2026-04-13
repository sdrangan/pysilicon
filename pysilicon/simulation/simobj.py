from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generator

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


class SimObj(object):
    """
    Base class for active simulation entities built on top of ``simpy``.

    A ``SimObj`` owns one or more concurrent processes registered with a shared
    ``simpy.Environment``. Subclasses typically register long-running loops that
    consume and produce transactions.
    """

    def __init__(
        self,
        env: simpy.Environment,
        name: str | None = None,
        track_action_overlaps: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        env : simpy.Environment
            Shared simulation environment.
        name : str | None
            Optional object name. Defaults to class name.
        track_action_overlaps : bool
            If ``True``, overlapping action windows are captured in
            ``action_overlaps`` for race/resource conflict analysis.
        """
        self.env: simpy.Environment = env
        self.name: str = self.__class__.__name__ if name is None else name
        self.track_action_overlaps: bool = track_action_overlaps

        self.processes: list[simpy.events.Process] = []
        self._process_factories: list[tuple[str, ProcessFactory]] = []
        self.action_history: list[ActionRecord] = []
        self.action_overlaps: list[ActionOverlap] = []

    @property
    def now(self) -> float:
        """Current simulation timestamp."""
        return float(self.env.now)

    def timeout(self, delay: float) -> simpy.events.Timeout:
        """Convenience wrapper around ``env.timeout``."""
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
