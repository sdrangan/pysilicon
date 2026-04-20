
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vcdvcd import VCDVCD

from pysilicon.hw.arrayutils import read_array
from pysilicon.utils.vcd import VcdParser
from pysilicon.utils.timing import TimingDiagram

# Import conv2d schemas (sibling module)
from conv2d_demo import Conv2DCmd, Conv2DResp, Conv2DDebug, Conv2DEvent

class Conv2DTimingResult:
    """
    Container for the decoded results of a conv2d VCD timing analysis.

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
        self.dbg_signals: dict = {}
        self.cmd_time : float | None = None
        self.resp_time : float | None = None
        self.dbg_times : List[float] = []
        self.cmd: Conv2DCmd | None = None
        self.resp: Conv2DResp | None = None
        self.dbg: List[Conv2DDebug] = []
        self.vp: VcdParser | None = None    

def analyze_vcd(vcd_path: str | Path) -> Conv2DTimingResult:
    """
    Analyze a VCD file captured from the conv2d Vitis HLS kernel.

    Loads the VCD, extracts the AXI4-Stream input and output signals,
    extracts bursts, decodes the command header, input samples, response
    header, output samples, and response footer, and returns all results
    in a :class:`Conv2DTimingResult`.

    Parameters
    ----------
    vcd_path : str | Path
        Path to the VCD file to analyze.

    Returns
    -------
    Conv2DTimingResult
        Structured result containing stream data and timing

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

    result = Conv2DTimingResult()

    # Parse VCD
    vcd = VCDVCD(str(vcd_path), signals=None, store_tvs=True)

    # Set up VCD parser
    vp = VcdParser(vcd)
    result.vp = vp

    # Discover clock signal
    result.clk_name = vp.add_clock_signal()

    # Discover AXI4-Stream signals
    in_stream_name = f"in_stream_"
    out_stream_name = f"out_stream_"
    dbg_stream_name = f"debug_stream_"

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
    result.dbg_signals, _dbg_bw = vp.add_axiss_signals(
        name=dbg_stream_name,
        short_name_prefix="debug_stream",
        ignore_multiple=True,
    )

    # Initialize the burst list
    bursts = []

    # Extract input bursts 
    bursts_in, clk_period = vp.extract_axis_bursts(
        result.clk_name, result.in_signals
    )
    for i, burst in enumerate(bursts_in):
        cmd = Conv2DCmd()
        words = burst['data']
        cmd.deserialize(words, word_bw=_in_bw)
        b = {'tstart': burst['tstart'], 'type' : 'Conv2DCmd', 'data' : cmd}
        bursts.append(b)
    print(f'Input bursts: {len(bursts_in)}')

    
    # Extract the debug event bursts
    bursts_dbg, _ = vp.extract_axis_bursts(
        result.clk_name, result.dbg_signals
    ) 
    for i, burst in enumerate(bursts_dbg):
        dbg = Conv2DDebug()
        words = burst['data']
        dbg.deserialize(words, word_bw=_dbg_bw)
        b = {'tstart': burst['tstart'], 'type': 'Conv2DDebug', 'data' : dbg}
        bursts.append(b)
    print(f'Debug bursts: {len(bursts_dbg)}')

    # Extract the output burst
    bursts_out, _ = vp.extract_axis_bursts(
        result.clk_name, result.out_signals
    )
    for i, burst in enumerate(bursts_out):
        resp = Conv2DResp()
        words = burst['data']
        resp.deserialize(words, word_bw=_out_bw)
        b = {'tstart': burst['tstart'], 'type': 'Conv2DResp', 'data' : resp}
        bursts.append(b)
    print(f'Output bursts: {len(bursts_out)}')

 
    return bursts, clk_period

if __name__ == "__main__":
    vcd_path = Path(__file__).parent / "vcd" / "dump.vcd"
    bursts = analyze_vcd(vcd_path)