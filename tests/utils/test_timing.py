"""
tests/utils/test_timing.py - unit tests for pysilicon.utils.timing and the
canonical example in examples/timing/basic_timing_diagram.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / CI-safe backend – must be set before pyplot import

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_example_module():
    """Dynamically import examples/timing/basic_timing_diagram.py."""
    example_path = (
        Path(__file__).parent.parent.parent
        / "examples" / "timing" / "basic_timing_diagram.py"
    )
    spec = importlib.util.spec_from_file_location(
        "basic_timing_diagram", example_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Tests for core timing classes
# ---------------------------------------------------------------------------

class TestSigTimingInfo:
    def test_two_level_binary(self):
        from pysilicon.utils.timing import SigTimingInfo

        sig = SigTimingInfo("clk", [0, 5, 10, 15], ['0', '1', '0', '1'])
        assert sig.two_level is True

    def test_two_level_false_when_unknown(self):
        from pysilicon.utils.timing import SigTimingInfo

        sig = SigTimingInfo("bus", [0, 5, 10], ['x', '4', 'x'])
        assert sig.two_level is False

    def test_attributes_stored(self):
        from pysilicon.utils.timing import SigTimingInfo

        sig = SigTimingInfo("a", [0, 10], ['0', '1'], is_clock=True)
        assert sig.name == "a"
        assert list(sig.times) == [0, 10]
        assert sig.values == ['0', '1']
        assert sig.is_clock is True


class TestClkSig:
    def test_period_and_ncycles(self):
        from pysilicon.utils.timing import ClkSig

        clk = ClkSig(period=10, ncycles=4)
        # 4 cycles -> 8 transitions
        assert len(clk.times) == 8
        assert len(clk.values) == 8

    def test_start_rising(self):
        from pysilicon.utils.timing import ClkSig

        clk = ClkSig(period=10, ncycles=2, start_rising=True)
        assert clk.values[0] == '1'

    def test_start_falling(self):
        from pysilicon.utils.timing import ClkSig

        clk = ClkSig(period=10, ncycles=2, start_rising=False)
        assert clk.values[0] == '0'

    def test_clk_periods_rising_edges(self):
        from pysilicon.utils.timing import ClkSig

        clk = ClkSig(period=10, ncycles=4, start_rising=True)
        edges = clk.clk_periods()
        # Rising edges at t = 0, 10, 20, 30
        assert edges == pytest.approx([0, 10, 20, 30])

    def test_is_clock_flag(self):
        from pysilicon.utils.timing import ClkSig

        clk = ClkSig()
        assert clk.is_clock is True


class TestTimingDiagram:
    def _build_diagram(self):
        from pysilicon.utils.timing import ClkSig, SigTimingInfo, TimingDiagram

        clk = ClkSig(period=10, ncycles=4)
        sig = SigTimingInfo("x", [0, 5, 15, 25], ['x', '1', '0', 'x'])
        td = TimingDiagram()
        td.add_signal(clk)
        td.add_signal(sig)
        return td

    def test_add_signal_stores_by_name(self):
        td = self._build_diagram()
        assert 'clk' in td.sig_info
        assert 'x' in td.sig_info

    def test_add_signals_multiple(self):
        from pysilicon.utils.timing import SigTimingInfo, TimingDiagram

        td = TimingDiagram()
        sigs = [
            SigTimingInfo("a", [0, 5], ['0', '1']),
            SigTimingInfo("b", [0, 5], ['1', '0']),
        ]
        td.add_signals(sigs)
        assert set(td.sig_info.keys()) == {'a', 'b'}

    def test_plot_signals_returns_axes(self):
        import matplotlib.pyplot as plt
        from pysilicon.utils.timing import ClkSig, TimingDiagram

        td = TimingDiagram()
        td.add_signal(ClkSig(period=10, ncycles=4))
        ax = td.plot_signals()
        assert ax is not None
        plt.close("all")

    def test_plot_signals_no_crash_with_trange(self):
        import matplotlib.pyplot as plt

        td = self._build_diagram()
        ax = td.plot_signals(trange=[0, 40])
        assert ax is not None
        plt.close("all")


# ---------------------------------------------------------------------------
# Smoke test: example figure generation
# ---------------------------------------------------------------------------

def test_save_timing_figures_creates_png(tmp_path):
    """Import save_timing_figures from the example script and verify output."""
    mod = _load_example_module()
    saved = mod.save_timing_figures(tmp_path)

    assert len(saved) >= 1, "Expected at least one output file"
    for p in saved:
        p = Path(p)
        assert p.exists(), f"Expected output file not found: {p}"
        assert p.stat().st_size > 0, f"Output file is empty: {p}"
        assert p.suffix == ".png", f"Expected PNG, got: {p.suffix}"