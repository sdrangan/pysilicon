"""Test that the mem_demo harness runs and self-checks."""
from __future__ import annotations

from examples.memory.mem_demo import MemCrossbarDemo, MemDemo, run_and_check


def test_mem_demo_direct_passes():
    demo = MemDemo()
    assert demo.run_and_check() is True


def test_mem_demo_crossbar_passes():
    demo = MemCrossbarDemo()
    assert demo.run_and_check() is True


def test_run_and_check_passes():
    assert run_and_check() is True
