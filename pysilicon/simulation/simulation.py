from __future__ import annotations

import simpy


class Simulation:
    """
    Runtime coordination object for a SimPy-based simulation.

    ``Simulation`` owns the ``simpy`` environment and the list of registered
    ``SimObj`` instances.  It orchestrates the standard three-phase lifecycle:

    1. ``pre_sim()``  — called on every registered object before the run.
    2. ``run_proc()`` — optional generator process; scheduled via
       ``env.process()`` when not ``None``.
    3. ``post_sim()`` — called on every registered object after the run.
    """

    def __init__(self) -> None:
        self.env: simpy.Environment = simpy.Environment()
        self._sim_objs: list = []

    def add_obj(self, obj: object) -> None:
        """Register a ``SimObj`` with this simulation.

        Called automatically from ``SimObj.__init__`` so callers rarely need
        to invoke this directly.
        """
        self._sim_objs.append(obj)

    def run_sim(self) -> None:
        """Execute the full simulation lifecycle.

        Steps
        -----
        1. Call ``pre_sim()`` on all registered objects in registration order.
        2. Schedule each object's ``run_proc()`` generator as a SimPy process
           (objects whose ``run_proc()`` returns ``None`` are skipped).
        3. Advance the simulation via ``env.run()``.
        4. Call ``post_sim()`` on all registered objects in registration order.
        """
        for obj in self._sim_objs:
            obj.pre_sim()

        for obj in self._sim_objs:
            proc = obj.run_proc()
            if proc is not None:
                self.env.process(proc)

        self.env.run()

        for obj in self._sim_objs:
            obj.post_sim()
