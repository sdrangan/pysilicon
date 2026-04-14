"""
basic_timing_diagram.py - canonical runnable example for pysilicon timing utilities.

Generates a minimal timing diagram that illustrates ``ClkSig``, ``SigTimingInfo``,
and ``TimingDiagram``, then saves the figure(s) to an output directory.

Usage
-----
Run from the repository root to regenerate the docs assets::

    python examples/timing/basic_timing_diagram.py

Or specify a custom output directory::

    python examples/timing/basic_timing_diagram.py --output /path/to/dir
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Default destination keeps generated figures next to the docs page that
# references them.
_DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "docs" / "guide" / "_static" / "timing"


def save_timing_figures(output_dir: str | Path) -> list[Path]:
    """Build a basic timing diagram and save figure(s) to *output_dir*.

    The example models a simple registered pipeline stage::

        clk  - 10 ns period, 4 cycles
        x    - input data, changes a few ns before each rising edge
        xreg - registered copy of x (available one cycle later)
        y    - combinational output y = xreg * xreg

    Parameters
    ----------
    output_dir:
        Directory where the PNG file(s) will be written.  Created if it does
        not already exist.

    Returns
    -------
    list[Path]
        List of paths to the saved figure files.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend – safe for scripts and CI
    import matplotlib.pyplot as plt

    from pysilicon.utils.timing import ClkSig, SigTimingInfo, TimingDiagram

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build signals
    # ------------------------------------------------------------------
    ncycles = 4
    clk_period = 10
    clk = ClkSig(clk_name="clk", period=clk_period, ncycles=ncycles)

    # Input x – arrives a few ns before each rising edge
    xvals  = ['x', '4',  'x', '5',  'x', '8',  'x']
    xtimes = [0,    3,   10,  13,   20,  23,   30]
    xsig = SigTimingInfo("x", xtimes, xvals)

    # Registered value: xreg <= x  (updated on rising edge)
    xregvals  = ['x', '4',  'x', '5',  'x', '8',  'x']
    xregtimes = [0,   11,   20,  21,   30,  31,   40]
    xregsig = SigTimingInfo("xreg", xregtimes, xregvals)

    # Combinational output y = xreg * xreg
    yvals  = ['x', '16', 'x', '25', 'x', '64', 'x']
    ytimes = [0,   14,   20,  24,   30,  34,   40]
    ysig = SigTimingInfo("y", ytimes, yvals)

    # ------------------------------------------------------------------
    # Assemble and plot the timing diagram
    # ------------------------------------------------------------------
    td = TimingDiagram()
    td.add_signal(clk)
    td.add_signals([xsig, xregsig, ysig])

    ax = td.plot_signals(trange=[0, 40])
    ax.set_xlabel("Time [ns]")

    # Highlight the two pipeline stages with translucent color patches
    colors = ['blue', 'green']
    sigs = ['x', 'xreg', 'y']
    for i, color in enumerate(colors):
        for s in sigs:
            td.add_patch(s, ind=i * 2 + 1, color=color, alpha=0.2)

    fig = ax.get_figure()
    fig.tight_layout()

    out_path = output_dir / "basic_timing_diagram.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return [out_path]


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate basic timing diagram figure(s) into an output directory."
    )
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT_DIR),
        help=(
            "Directory to write figure(s) into "
            f"(default: {_DEFAULT_OUTPUT_DIR})"
        ),
    )
    args = parser.parse_args()

    saved = save_timing_figures(args.output)
    for p in saved:
        print(f"Saved: {p}")
