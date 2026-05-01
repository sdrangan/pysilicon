import pytest

from pysilicon.simulation.simulation import Simulation
from pysilicon.simulation.simobj import SimObj


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class PassiveObj(SimObj):
    """SimObj that only uses pre_sim / post_sim (no active process)."""

    def __init__(self, sim: Simulation) -> None:
        super().__init__(sim=sim)
        self.pre_called = False
        self.post_called = False

    def pre_sim(self) -> None:
        self.pre_called = True

    def post_sim(self) -> None:
        self.post_called = True


class ActiveObj(SimObj):
    """SimObj that also runs a process during the simulation."""

    def __init__(self, sim: Simulation, duration: float = 5.0) -> None:
        super().__init__(sim=sim)
        self.duration = duration
        self.pre_called = False
        self.post_called = False
        self.ran = False

    def pre_sim(self) -> None:
        self.pre_called = True

    def run_proc(self):
        yield self.timeout(self.duration)
        self.ran = True

    def post_sim(self) -> None:
        self.post_called = True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_simulation_has_env_and_empty_registry() -> None:
    """A fresh Simulation exposes a SimPy environment and an empty object list."""
    import simpy

    sim = Simulation()
    assert isinstance(sim.env, simpy.Environment)
    assert sim._sim_objs == []


def test_add_obj_registers_object() -> None:
    """add_obj appends an object to the internal registry."""
    sim = Simulation()
    obj = PassiveObj(sim)   # __init__ calls add_obj automatically
    assert sim._sim_objs == [obj]


def test_simobj_init_auto_registers() -> None:
    """SimObj registers itself with the Simulation during __init__."""
    sim = Simulation()
    a = PassiveObj(sim)
    b = PassiveObj(sim)
    assert sim._sim_objs == [a, b]


def test_run_sim_calls_lifecycle_hooks_on_passive_obj() -> None:
    """run_sim calls pre_sim and post_sim on passive (no process) objects."""
    sim = Simulation()
    obj = PassiveObj(sim)

    sim.run_sim()

    assert obj.pre_called
    assert obj.post_called


def test_run_sim_schedules_and_runs_active_obj() -> None:
    """run_sim schedules run_proc and advances time for active objects."""
    sim = Simulation()
    obj = ActiveObj(sim, duration=10.0)

    sim.run_sim()

    assert obj.pre_called
    assert obj.ran
    assert obj.post_called
    assert sim.env.now == 10.0


def test_run_sim_registration_order() -> None:
    """pre_sim and post_sim are invoked in registration order."""
    sim = Simulation()
    order: list[str] = []

    class Ordered(SimObj):
        def __init__(self, sim, tag):
            super().__init__(sim=sim)
            self.tag = tag

        def pre_sim(self):
            order.append(f"pre_{self.tag}")

        def post_sim(self):
            order.append(f"post_{self.tag}")

    Ordered(sim, "a")
    Ordered(sim, "b")
    Ordered(sim, "c")

    sim.run_sim()

    assert order == ["pre_a", "pre_b", "pre_c", "post_a", "post_b", "post_c"]


def test_default_run_proc_is_none() -> None:
    """The default run_proc implementation returns None (passive object)."""
    sim = Simulation()
    obj = SimObj(sim=sim)
    assert obj.run_proc() is None


def test_default_lifecycle_hooks_are_noop() -> None:
    """Default pre_sim and post_sim do not raise and return None."""
    sim = Simulation()
    obj = SimObj(sim=sim)
    assert obj.pre_sim() is None
    assert obj.post_sim() is None


def test_passive_obj_not_scheduled_as_process() -> None:
    """Objects whose run_proc returns None are not scheduled; env time stays at 0."""
    sim = Simulation()
    PassiveObj(sim)

    sim.run_sim()

    assert sim.env.now == 0.0


def test_run_sim_mixed_active_and_passive() -> None:
    """Active and passive objects coexist; only active objects advance env time."""
    sim = Simulation()
    passive = PassiveObj(sim)
    active = ActiveObj(sim, duration=3.0)

    sim.run_sim()

    assert passive.pre_called and passive.post_called
    assert active.ran
    assert sim.env.now == 3.0


def test_simobj_exposes_sim_reference() -> None:
    """SimObj stores a reference to its owning Simulation."""
    sim = Simulation()
    obj = PassiveObj(sim)
    assert obj.sim is sim
