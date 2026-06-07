"""Simulation event logger.

Writes timestamped CSV rows during a SimPy simulation.  Multiple processes
may call :meth:`Logger.log` concurrently; a SimPy Resource serializes writes
to prevent interleaved rows.  Each entry is flushed immediately so the log
is inspectable even when the simulation terminates with an error.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import simpy

from waveflow.hw.synth import sim_only
from waveflow.simulation.simobj import ProcessGen, SimObj


class NullLogger:
    """Drop-in replacement for Logger that discards all entries.

    Used as the default when a component does not need logging, so call
    sites never need ``if self.logger:`` guards.
    """

    @sim_only
    def log(self, **kwargs) -> None:
        pass


@dataclass(kw_only=True)
class Logger(SimObj):
    """Timestamped CSV event logger for SimPy simulations.

    Usage::

        logger = Logger(name='log', sim=sim,
                        file_path='sim.csv', fields=['event', 'nsamp'])

        # inside any run_proc:
        logger.log(event='start', nsamp=1024)
        logger.log(event='done')          # nsamp left blank

        # after run_sim:
        times, events = logger.get_tv('event')
    """

    file_path: Path | str
    fields: list[str]

    def __post_init__(self) -> None:
        super().__post_init__()
        self._fp = open(self.file_path, 'w', newline='')
        self._writer = csv.writer(self._fp)
        self._writer.writerow(['time'] + self.fields)
        self._fp.flush()
        self._resource = simpy.Resource(self.env)

    @sim_only
    def log(self, **kwargs) -> None:
        """Schedule a log entry at the current simulation time.

        Only fields declared in ``fields`` are accepted.  Unspecified fields
        are written as empty strings.  The write is deferred to a SimPy
        process so concurrent callers are serialized correctly.
        """
        for k in kwargs:
            if k not in self.fields:
                raise ValueError(
                    f"Unknown log field '{k}'. Valid fields: {self.fields}"
                )
        t = self.env.now
        row = [t] + [kwargs.get(f, '') for f in self.fields]
        self.env.process(self._write_proc(row))

    def _write_proc(self, row: list) -> ProcessGen[None]:
        with self._resource.request() as req:
            yield req
            self._writer.writerow(row)
            self._fp.flush()

    def post_sim(self) -> None:
        self._close()

    def error_cleanup(self) -> None:
        self._close()

    def _close(self) -> None:
        if not self._fp.closed:
            self._fp.close()

    def get_tv(self, field: str) -> tuple[list[float], list]:
        """Return ``(times, values)`` for *field* from the log file.

        Only rows where *field* was explicitly logged (non-empty) are
        included.  Values are cast to ``float`` where possible; otherwise
        returned as strings.  Call after :meth:`~Simulation.run_sim`.
        """
        if field not in self.fields:
            raise ValueError(
                f"Unknown field '{field}'. Valid fields: {self.fields}"
            )
        col_idx = self.fields.index(field) + 1  # +1 for leading time column
        times: list[float] = []
        vals: list = []
        with open(self.file_path, newline='') as f:
            reader = csv.reader(f)
            next(reader)  # skip header row
            for row in reader:
                if col_idx < len(row) and row[col_idx] != '':
                    times.append(float(row[0]))
                    v = row[col_idx]
                    try:
                        vals.append(float(v))
                    except ValueError:
                        vals.append(v)
        return times, vals
