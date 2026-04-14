"""
tests/poly/test_timing_analysis_plot.py — plot tests for poly timing analysis.

These tests verify that :func:`plot_poly_timing` runs without error in a
non-interactive (headless) environment and returns valid matplotlib objects.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before importing pyplot
import matplotlib.pyplot as plt
import pytest

# ---- path setup for examples/poly sibling module ----
_POLY_DIR = Path(__file__).resolve().parents[2] / "examples" / "poly"
if str(_POLY_DIR) not in sys.path:
    sys.path.insert(0, str(_POLY_DIR))

from timing_analysis import PolyTimingResult, analyze_poly_vcd, plot_poly_timing

FIXTURE_VCD = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "poly"
    / "timing"
    / "poly_timing_fixture.vcd"
)


@pytest.fixture(scope="module")
def result() -> PolyTimingResult:
    return analyze_poly_vcd(FIXTURE_VCD)


class TestPlotPolyTiming:
    def test_returns_axes(self, result: PolyTimingResult) -> None:
        ax = plot_poly_timing(result, show=False)
        assert ax is not None
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_xlabel_set(self, result: PolyTimingResult) -> None:
        ax = plot_poly_timing(result, show=False)
        assert "ns" in ax.get_xlabel().lower() or "time" in ax.get_xlabel().lower()
        plt.close("all")

    def test_legend_present(self, result: PolyTimingResult) -> None:
        ax = plot_poly_timing(result, show=False)
        legend = ax.get_legend()
        assert legend is not None
        plt.close("all")
