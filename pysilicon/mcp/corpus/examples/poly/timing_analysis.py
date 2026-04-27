"""
timing_analysis.py — Poly AXI4-Stream timing analysis

Provides a reusable API for analyzing an existing VCD file captured from
the ``poly`` Vitis HLS kernel.  The analysis can run from a pre-captured
VCD without rerunning RTL co-simulation.

Typical usage
-------------
>>> from timing_analysis import analyze_poly_vcd, plot_poly_timing
>>> result = analyze_poly_vcd("vcd/dump.vcd")
>>> print(result.cmd_hdr.val)
>>> plot_poly_timing(result)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from vcdvcd import VCDVCD

from pysilicon.hw.arrayutils import read_array
from pysilicon.utils.vcd import VcdParser
from pysilicon.utils.timing import TimingDiagram

# Import poly schemas (sibling module)
from poly_demo import PolyCmdHdr, PolyRespHdr, PolyRespFtr, Float32

_WORD_BW = 32
_TOP_NAME = "AESL_inst_poly"


class PolyTimingResult:
    """
    Container for the decoded results of a poly VCD timing analysis.

    Attributes
    ----------
    clk_name : str
        Full VCD signal name of the clock.
    clk_period : float
        Estimated clock period in nanoseconds.
    in_signals : dict[str, str]
        AXI4-Stream signal name mapping for the input stream.
    out_signals : dict[str, str]
        AXI4-Stream signal name mapping for the output stream.
    bursts_in : list[dict]
        Raw burst dictionaries extracted from the input stream.
    bursts_out : list[dict]
        Raw burst dictionaries extracted from the output stream.
    cmd_hdr : PolyCmdHdr
        Decoded command header (tx_id, coeffs, nsamp).
    x : numpy.ndarray
        Input sample array decoded from the input data burst.
    resp_hdr : PolyRespHdr
        Decoded response header (tx_id echo).
    y : numpy.ndarray
        Output sample array decoded from the output data burst.
    resp_ftr : PolyRespFtr
        Decoded response footer (nsamp_read, error).
    vp : VcdParser
        The underlying VCD parser instance (for advanced use).
    """

    def __init__(self) -> None:
        self.clk_name: str | None = None
        self.clk_period: float | None = None
        self.in_signals: dict = {}
        self.out_signals: dict = {}
        self.bursts_in: list = []
        self.bursts_out: list = []
        self.cmd_hdr: PolyCmdHdr | None = None
        self.x: np.ndarray | None = None
        self.resp_hdr: PolyRespHdr | None = None
        self.y: np.ndarray | None = None
        self.resp_ftr: PolyRespFtr | None = None
        self.vp: VcdParser | None = None


def analyze_poly_vcd(vcd_path: str | Path) -> PolyTimingResult:
    """
    Analyze a VCD file captured from the poly Vitis HLS kernel.

    Loads the VCD, extracts the AXI4-Stream input and output signals,
    extracts bursts, decodes the command header, input samples, response
    header, output samples, and response footer, and returns all results
    in a :class:`PolyTimingResult`.

    Parameters
    ----------
    vcd_path : str | Path
        Path to the VCD file to analyze.

    Returns
    -------
    PolyTimingResult
        Structured result containing decoded headers, samples, and timing info.

    Raises
    ------
    FileNotFoundError
        If *vcd_path* does not exist.
    ValueError
        If the expected signals or burst structure are not found in the VCD.
    """
    vcd_path = Path(vcd_path)
    if not vcd_path.exists():
        raise FileNotFoundError(f"VCD file not found: {vcd_path}")

    result = PolyTimingResult()

    # Parse VCD
    vcd = VCDVCD(str(vcd_path), signals=None, store_tvs=True)

    # Set up VCD parser
    vp = VcdParser(vcd)
    result.vp = vp

    # Discover clock signal
    result.clk_name = vp.add_clock_signal()

    # Discover AXI4-Stream signals
    in_stream_name = f"{_TOP_NAME}.in_stream_"
    out_stream_name = f"{_TOP_NAME}.out_stream_"

    result.in_signals, _in_bw = vp.add_axiss_signals(
        name=in_stream_name,
        short_name_prefix="in_stream",
        ignore_multiple=True,
    )
    result.out_signals, _out_bw = vp.add_axiss_signals(
        name=out_stream_name,
        short_name_prefix="out_stream",
        ignore_multiple=True,
    )

    # Extract bursts (get_values() is called internally)
    result.bursts_in, result.clk_period = vp.extract_axis_bursts(
        result.clk_name, result.in_signals
    )
    result.bursts_out, _ = vp.extract_axis_bursts(
        result.clk_name, result.out_signals
    )

    if len(result.bursts_in) < 2:
        raise ValueError(
            f"Expected at least 2 input bursts (cmd_hdr + data), got {len(result.bursts_in)}"
        )
    if len(result.bursts_out) < 3:
        raise ValueError(
            f"Expected at least 3 output bursts (resp_hdr + data + resp_ftr), got {len(result.bursts_out)}"
        )

    # Decode command header from burst 0
    result.cmd_hdr = PolyCmdHdr()
    result.cmd_hdr.deserialize(word_bw=_WORD_BW, packed=result.bursts_in[0]["data"])

    # Decode input samples from burst 1
    nsamp = int(result.cmd_hdr.val["nsamp"])
    result.x = read_array(
        packed=result.bursts_in[1]["data"],
        word_bw=_WORD_BW,
        elem_type=Float32,
        shape=(nsamp,),
    )

    # Decode response header from output burst 0
    result.resp_hdr = PolyRespHdr()
    result.resp_hdr.deserialize(word_bw=_WORD_BW, packed=result.bursts_out[0]["data"])

    # Decode output samples from output burst 1
    result.y = read_array(
        packed=result.bursts_out[1]["data"],
        word_bw=_WORD_BW,
        elem_type=Float32,
        shape=(nsamp,),
    )

    # Decode response footer from output burst 2
    result.resp_ftr = PolyRespFtr()
    result.resp_ftr.deserialize(word_bw=_WORD_BW, packed=result.bursts_out[2]["data"])

    return result


def plot_poly_timing(
    result: PolyTimingResult,
    trange: tuple[float, float] | None = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot the AXI4-Stream timing diagram from a :class:`PolyTimingResult`,
    with input and output bursts color-coded by type.

    Parameters
    ----------
    result : PolyTimingResult
        Result from :func:`analyze_poly_vcd`.
    trange : tuple[float, float] | None
        Optional ``(t_start, t_end)`` in nanoseconds to zoom the plot.
    show : bool
        If ``True`` (default), call ``plt.show()`` after plotting.
        Set to ``False`` in non-interactive or test environments.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object of the timing diagram.
    """
    from matplotlib.patches import Patch

    vp = result.vp
    sig_list = vp.get_td_signals()

    td = TimingDiagram()
    td.add_signals(sig_list)

    ax = td.plot_signals(
        add_clk_grid=True,
        trange=trange,
        text_scale_factor=1e4,
        text_mode="never",
    )
    ax.set_xlabel("Time [ns]")

    def _color_burst(sig_name, burst, color, clk_period):
        t0 = burst["tstart"]
        t1 = t0 + len(burst["beat_type"]) * clk_period
        td.add_patch(sig_name=sig_name, time=[t0, t1], color=color, alpha=0.3)

    hdr_color = "orange"
    data_color = "green"
    footer_color = "blue"
    cp = result.clk_period

    _color_burst("in_stream_TDATA", result.bursts_in[0], hdr_color, cp)
    _color_burst("in_stream_TDATA", result.bursts_in[1], data_color, cp)
    _color_burst("out_stream_TDATA", result.bursts_out[0], hdr_color, cp)
    _color_burst("out_stream_TDATA", result.bursts_out[1], data_color, cp)
    _color_burst("out_stream_TDATA", result.bursts_out[2], footer_color, cp)

    legend_elements = [
        Patch(facecolor="orange", edgecolor="black", alpha=0.3, label="header"),
        Patch(facecolor="green", edgecolor="black", alpha=0.3, label="data"),
        Patch(facecolor="blue", edgecolor="black", alpha=0.3, label="footer"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )

    if show:
        plt.show()

    return ax
