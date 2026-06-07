"""Committed-figure workflow for the ``shared_mem`` (histogram) example.

Two figures document the multi-buffer m_axi story:

* **burst_diagram.svg** — the AXI-MM burst *layout*: the ``data`` and ``bin_edges``
  read bursts and the ``counts`` write burst at their byte addresses.
* **timing_diagram.svg** — one transaction's *timeline*: the read phase, the
  compute gap, and the write phase, cycle-annotated.

Both are rendered deterministically from the cosim burst report
(``vcd/burst_info.json``), which is itself regenerable from the **committed**
``vcd/dump.vcd`` with no Vitis run.  The generated SVGs land in ``results/``
(gitignored); :class:`SyncDocsFiguresStep` promotes them — on demand, via an
explicit manifest — into ``docs/examples/shared_mem/images/`` as committed assets,
so the docs figure only changes when you intend it to and the change is a
reviewable ``git diff``.

Run it with::

    python hist_build.py --through sync_docs_figures
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("svg")
# Deterministic SVG: a stable hashsalt fixes the element ids matplotlib would
# otherwise randomize, so a re-render only diffs when the figure truly changed.
matplotlib.rcParams["svg.hashsalt"] = "shared_mem_figures"
import matplotlib.pyplot as plt  # noqa: E402

from waveflow.build.build import BuildConfig, BuildStep  # noqa: E402

BYTES_PER_WORD = 4  # 32-bit memory words

# Source-in-code -> dest-in-docs.  The proxy never copies by guesswork; this is
# the explicit, reviewable mapping a sync consults.
FIGURE_MANIFEST = [
    {"name": "burst_diagram",
     "source": "results/burst_diagram.svg",
     "dest": "docs/examples/shared_mem/images/burst_diagram.svg"},
    {"name": "timing_diagram",
     "source": "results/timing_diagram.svg",
     "dest": "docs/examples/shared_mem/images/timing_diagram.svg"},
]

_REGION_COLORS = {"data": "#4C78A8", "bin_edges": "#54A24B", "counts": "#E45756"}


def _save_svg(fig, path: Path) -> None:
    """Write a deterministic SVG (no embedded timestamp)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # metadata Date=None omits the <dc:date> matplotlib embeds by default.
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ensure_burst_info(example_dir: Path, *, ndata: int = 37, nbins: int = 6) -> Path:
    """Return ``vcd/burst_info.json``, regenerating it from the committed
    ``vcd/dump.vcd`` if absent (cheap — no Vitis, just a VCD parse + the numpy
    golden for the expected layout)."""
    example_dir = Path(example_dir)
    burst_info = example_dir / "vcd" / "burst_info.json"
    if burst_info.exists():
        return burst_info
    dump_vcd = example_dir / "vcd" / "dump.vcd"
    if not dump_vcd.exists():
        raise FileNotFoundError(
            f"Neither {burst_info} nor {dump_vcd} exists; run the cosim flow "
            "(python hist_build.py --through extract_bursts --trace-level port) first."
        )
    # Lazy import (function-level) of the burst-extraction harness from hist_build,
    # which avoids a module-level cycle (hist_build imports the figure step classes
    # from this module).
    try:
        from examples.shared_mem.hist_build import HistTest
    except ModuleNotFoundError:
        from hist_build import HistTest  # type: ignore[no-redef]
    ht = HistTest(example_dir=example_dir, ndata=ndata, nbins=nbins)
    ht.simulate()
    ht.extract_bursts(vcd_path=dump_vcd)
    return burst_info


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------

def render_burst_diagram(burst_info_path: Path, out_svg: Path) -> Path:
    """The multi-buffer burst layout: read/write bursts on a byte-address axis."""
    report = json.loads(Path(burst_info_path).read_text(encoding="utf-8"))
    exp = report["expected"]
    rows = [("m_axi read", exp["read_bursts"]), ("m_axi write", exp["write_bursts"])]

    fig, ax = plt.subplots(figsize=(8.5, 2.6))
    max_addr = 0
    for row_idx, (_label, bursts) in enumerate(rows):
        y = len(rows) - 1 - row_idx
        for b in bursts:
            addr, nwords, name = int(b["addr"]), int(b["nwords"]), str(b["name"])
            width = nwords * BYTES_PER_WORD
            ax.barh(y, width, left=addr, height=0.6,
                    color=_REGION_COLORS.get(name, "#9D9D9D"),
                    edgecolor="white", linewidth=1.2, zorder=3)
            ax.text(addr + width / 2, y, f"{name}\n{nwords}w",
                    ha="center", va="center", fontsize=8, color="white", zorder=4)
            max_addr = max(max_addr, addr + width)

    ax.set_yticks([len(rows) - 1 - i for i in range(len(rows))])
    ax.set_yticklabels([label for label, _ in rows])
    ax.set_xlim(-4, max_addr + 8)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("byte address into the gmem bundle")
    ax.set_title("Multi-buffer AXI-MM burst layout (ndata=37, nbins=6)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#DDDDDD", zorder=0)
    _save_svg(fig, Path(out_svg))
    return Path(out_svg)


def render_timing_diagram(burst_info_path: Path, out_svg: Path) -> Path:
    """One transaction's timeline: read phase, compute gap, write phase."""
    report = json.loads(Path(burst_info_path).read_text(encoding="utf-8"))
    clk_ns = float(report["clk_period_ns"])
    reads = report["actual"]["read_bursts"]
    writes = report["actual"]["write_bursts"]

    def _span(bursts):
        starts = [float(b["tstart"]) for b in bursts]
        ends = [float(b.get("data_tend", b["tstart"])) for b in bursts]
        return min(starts), max(ends)

    read_t0, read_t1 = _span(reads)
    write_t0, write_t1 = _span(writes)
    t_end = write_t1

    # Phases as (label, t_start, t_end, color).
    phases = [
        ("command in", read_t0 - clk_ns, read_t0, "#B279A2"),
        ("read data + edges", read_t0, read_t1, _REGION_COLORS["data"]),
        ("compute", read_t1, write_t0, "#9D9D9D"),
        ("write counts", write_t0, write_t1, _REGION_COLORS["counts"]),
        ("response out", t_end, t_end + clk_ns, "#B279A2"),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 2.6))
    for i, (label, t0, t1, color) in enumerate(phases):
        y = len(phases) - 1 - i
        ax.barh(y, max(t1 - t0, clk_ns), left=t0, height=0.6, color=color,
                edgecolor="white", linewidth=1.0, zorder=3)
        cyc = (t1 - t0) / clk_ns
        ann = f"{label}  ({cyc:.0f} cyc)" if cyc >= 1 else label
        ax.text(t1 + clk_ns, y, ann, ha="left", va="center", fontsize=8.5, zorder=4)

    ax.set_yticks([])
    ax.set_xlim(read_t0 - 3 * clk_ns, t_end + 22 * clk_ns)
    ax.set_ylim(-0.6, len(phases) - 0.4)
    ax.set_xlabel(f"time [ns]   (clk period = {clk_ns:.0f} ns)")
    ax.set_title("Histogram transaction timeline (ndata=37, nbins=6, from RTL cosim)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#DDDDDD", zorder=0)
    _save_svg(fig, Path(out_svg))
    return Path(out_svg)


# ---------------------------------------------------------------------------
# Build steps
# ---------------------------------------------------------------------------

class GenerateBurstDiagramStep(BuildStep):
    description = "Render the AXI-MM burst-layout SVG from the cosim burst report."
    consumes: list = []
    produces = {"burst_diagram_svg": Path("results/burst_diagram.svg")}

    def run(self, config: BuildConfig, **_) -> dict[str, Any]:
        burst_info = ensure_burst_info(config.root_dir)
        out = config.root_dir / "results" / "burst_diagram.svg"
        render_burst_diagram(burst_info, out)
        return {"burst_diagram_svg": out}


class GenerateTimingDiagramStep(BuildStep):
    description = "Render the transaction-timeline SVG from the cosim burst report."
    consumes: list = []
    produces = {"timing_diagram_svg": Path("results/timing_diagram.svg")}

    def run(self, config: BuildConfig, **_) -> dict[str, Any]:
        burst_info = ensure_burst_info(config.root_dir)
        out = config.root_dir / "results" / "timing_diagram.svg"
        render_timing_diagram(burst_info, out)
        return {"timing_diagram_svg": out}


class SyncDocsFiguresStep(BuildStep):
    """Promote the generated SVGs into the committed docs assets, on demand.

    Copies each manifest entry ``results/*.svg -> docs/.../images/*.svg`` and
    writes a ``sync_status.json`` provenance record (per figure: source path,
    content hash) next to the committed assets — the cheap staleness signal a
    docs lint can check without re-running Vitis."""

    description = "Copy generated figures into docs/images and record provenance."
    consumes = ["burst_diagram_svg", "timing_diagram_svg"]
    produces = {"docs_figures_sync": Path("docs/examples/shared_mem/images/sync_status.json")}

    def run(self, config: BuildConfig, **_) -> dict[str, Any]:
        # docs/ lives at the repo root; the example dir is examples/shared_mem.
        repo_root = config.root_dir.parents[1]
        records = []
        for entry in FIGURE_MANIFEST:
            src = config.root_dir / entry["source"]
            dst = repo_root / entry["dest"]
            if not src.exists():
                raise RuntimeError(f"Manifest source missing: {src}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
            records.append({
                "name": entry["name"],
                "source": entry["source"],
                "dest": entry["dest"],
                "source_sha256": _sha256(src),
            })
        sync_path = repo_root / "docs" / "examples" / "shared_mem" / "images" / "sync_status.json"
        sync_path.parent.mkdir(parents=True, exist_ok=True)
        sync_path.write_text(
            json.dumps({"figures": records}, indent=2) + "\n", encoding="utf-8")
        return {"docs_figures_sync": sync_path}
