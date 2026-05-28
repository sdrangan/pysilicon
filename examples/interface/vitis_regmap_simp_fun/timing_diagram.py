from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pysilicon.utils.timing import ClkSig, SigTimingInfo, TimingDiagram


def _busy_signal(name: str, cycles: int, max_cycles: int) -> SigTimingInfo:
    return SigTimingInfo(name, [0, cycles, max_cycles], ["1", "0", "0"])


def write_timing_diagram(
    py_timing_path: str | Path,
    cosim_timing_path: str | Path,
    verdict_path: str | Path,
    svg_path: str | Path,
    json_path: str | Path,
) -> dict:
    py_timing_path = Path(py_timing_path)
    cosim_timing_path = Path(cosim_timing_path)
    verdict_path = Path(verdict_path)
    svg_path = Path(svg_path)
    json_path = Path(json_path)

    py = json.loads(py_timing_path.read_text(encoding="utf-8"))
    cs = json.loads(cosim_timing_path.read_text(encoding="utf-8"))
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))

    py_cycles = int(py["transaction_cycles"])
    cosim_cycles = int(cs["transaction_cycles"])
    max_cycles = max(py_cycles, cosim_cycles) + 2

    td = TimingDiagram(time_unit="cycles")
    td.add_signal(ClkSig(clk_name="clk", period=1, ncycles=max_cycles, start_rising=True))
    td.add_signals([
        _busy_signal("py_busy", py_cycles, max_cycles),
        _busy_signal("cosim_busy", cosim_cycles, max_cycles),
    ])

    ax = td.plot_signals(add_clk_grid=True, trange=[0, max_cycles], text_mode="never")
    ax.set_xlabel("Cycle")
    td.add_patch("py_busy", time=[0, py_cycles], color="tab:blue", alpha=0.25)
    td.add_patch("cosim_busy", time=[0, cosim_cycles], color="tab:orange", alpha=0.25)
    ax.text(py_cycles, td.ytop["py_busy"] + 0.1, f"py done @ {py_cycles}", color="tab:blue")
    ax.text(cosim_cycles, td.ytop["cosim_busy"] + 0.1,
            f"cosim done @ {cosim_cycles}", color="tab:orange")
    ax.set_title("vitis_regmap_simp_fun timing comparison")

    fig = ax.get_figure()
    fig.tight_layout()
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg_path, format="svg")
    plt.close(fig)

    metadata = {
        "py_cycles": py_cycles,
        "cosim_cycles": cosim_cycles,
        "delta": int(verdict["delta"]),
        "tolerance": int(verdict["tolerance"]),
        "pass": bool(verdict["pass"]),
        "events": [
            {"signal": "py_busy", "event": "ap_start", "cycle": 0},
            {"signal": "py_busy", "event": "done", "cycle": py_cycles},
            {"signal": "cosim_busy", "event": "ap_start", "cycle": 0},
            {"signal": "cosim_busy", "event": "done", "cycle": cosim_cycles},
        ],
        "sources": {
            "py_timing": str(py_timing_path),
            "cosim_timing": str(cosim_timing_path),
            "timing_verdict": str(verdict_path),
            "svg": str(svg_path),
        },
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
