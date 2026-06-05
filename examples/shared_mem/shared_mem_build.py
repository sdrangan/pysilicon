"""Build helpers for the codegen-driven ``shared_mem`` (histogram) example.

Generates the Vitis HLS include headers (reusing ``HistTest.gen_vitis_code``) and
the **generated** kernel (``gen/hist.cpp`` / ``gen/hist.hpp``) from the Python
``HistAccel``, writes per-case input vectors, and exposes the golden expectation
(status + counts) — the pieces the C-sim test drives.  The generated kernel and
generated testbench (``gen/hist_tb.cpp`` from ``HistTBHls``) are compiled by
``run.tcl`` against the hand-written datapath hooks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pysilicon.build.hwgen import header_to_cpp, kernel_to_cpp, tb_files_to_str
from pysilicon.hw.arrayutils import read_uint32_file, write_uint32_file

try:
    from examples.shared_mem.hist import HistAccel, HistTBHls, golden_counts
    from examples.shared_mem.hist_demo import (
        Float32, HistCmd, HistError, HistResp, HistTest, MAX_NBINS, MAX_NDATA,
        Uint32Field,
    )
except ModuleNotFoundError:  # direct execution from the example dir
    from hist import HistAccel, HistTBHls, golden_counts  # type: ignore[no-redef]
    from hist_demo import (  # type: ignore[no-redef]
        Float32, HistCmd, HistError, HistResp, HistTest, MAX_NBINS, MAX_NDATA,
        Uint32Field,
    )


def generate_vitis_sources(work_dir: str | Path) -> Path:
    """Generate ``include/`` headers and the generated ``gen/hist.{cpp,hpp}``.

    Returns the ``gen`` directory.  The include headers come from
    ``HistTest.gen_vitis_code`` (the proven path); the kernel + header are
    generated from :class:`HistAccel`.
    """
    from pysilicon.build.build import BuildConfig, BuildDag
    from pysilicon.build.streamutils import MemMgrStep

    work_dir = Path(work_dir)
    HistTest(example_dir=work_dir).gen_vitis_code()   # streams + schemas + array-utils
    # gen_vitis_code omits the memory manager; the generated header + TB include
    # include/memmgr.hpp / memmgr_tb.hpp, so generate those too.
    mm_dag = BuildDag()
    mm_dag.add(MemMgrStep(output_dir="include"))
    mm_dag.run(BuildConfig(root_dir=work_dir))
    gen = work_dir / "gen"
    gen.mkdir(parents=True, exist_ok=True)
    (gen / "hist.cpp").write_text(kernel_to_cpp(HistAccel), encoding="utf-8")
    (gen / "hist.hpp").write_text(header_to_cpp(HistAccel), encoding="utf-8")
    # The generated testbench (Phase 5) — drives the generated kernel.
    for fname, content in tb_files_to_str(HistTBHls).items():
        (gen / fname).write_text(content, encoding="utf-8")
    return gen


def _write_array_file(path: Path, arr, elem_type, count: int) -> None:
    """Write exactly ``count`` elements as the raw word stream the C++
    ``read_uint32_file_array`` expects.  ``write_uint32_file`` emits a spurious
    1-word file for an empty array, which the TB's ``count==0`` read flags as
    trailing bytes — so write a true 0-byte file when there is nothing to emit
    (the nbins==1 zero-edge case, or a validation-failure 0-data case)."""
    if count > 0:
        write_uint32_file(arr, elem_type=elem_type, file_path=path, nwrite=count)
    else:
        Path(path).write_bytes(b"")


@dataclass
class HistCase:
    """One C-sim coverage case + its golden expectation."""

    ndata: int
    nbins: int
    seed: int = 3

    def gen_data(self) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        rng = np.random.default_rng(self.seed)
        data = (rng.normal(0.0, 1.25, size=max(self.ndata, 0)).astype(np.float32)
                if self.ndata > 0 else np.zeros(0, np.float32))
        edges = np.sort(
            rng.uniform(-2.5, 2.5, size=max(self.nbins - 1, 0)).astype(np.float32)
        )
        return data, edges

    @property
    def expected_status(self) -> HistError:
        # Mirrors HistAccel.validate (the kernel's validation logic).
        if self.ndata <= 0 or self.ndata > MAX_NDATA:
            return HistError.INVALID_NDATA
        if self.nbins <= 0 or self.nbins > MAX_NBINS:
            return HistError.INVALID_NBINS
        return HistError.NO_ERROR

    def write_inputs(self, data_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        data, edges = self.gen_data()
        # The generated TB reads the full HistCmd from cmd.bin (addresses are
        # placeholders it overwrites via alloc) and reads the input files
        # unconditionally, so write them always — even 0-element (the TB's read
        # of a 0/negative count is a no-op).
        HistCmd(tx_id=self.seed, data_addr=0, bin_edges_addr=0,
                ndata=self.ndata, nbins=self.nbins, cnt_addr=0
                ).write_uint32_file(str(data_dir / "cmd.bin"))
        _write_array_file(data_dir / "data_array.bin", data, Float32, max(self.ndata, 0))
        _write_array_file(data_dir / "edges_array.bin", edges, Float32, max(self.nbins - 1, 0))
        return data, edges

    def check_outputs(self, data_dir: str | Path, data, edges) -> tuple[bool, str]:
        """Compare the C-sim outputs against the golden; returns (passed, detail)."""
        data_dir = Path(data_dir)
        resp = HistResp().read_uint32_file(str(data_dir / "resp_data.bin"))
        if int(resp.status) != int(self.expected_status):
            return False, (f"status {int(resp.status)} != expected "
                           f"{int(self.expected_status)}")
        if self.expected_status != HistError.NO_ERROR:
            return True, f"status={int(resp.status)} (expected error)"
        counts = np.asarray(
            read_uint32_file(str(data_dir / "counts_array.bin"),
                             elem_type=Uint32Field, shape=self.nbins),
            dtype=np.uint32,
        )
        gold = golden_counts(data, edges, self.nbins)
        if not np.array_equal(counts, gold):
            return False, f"counts {counts.tolist()} != golden {gold.tolist()}"
        return True, f"counts={counts.tolist()} match golden"


# Coverage set: nbins==1 (the unconditional zero-count edges read), nbins>1 with
# several bins (normal binning), and a validation-failure case.
#
# The validation case is ndata==0 (INVALID_NDATA), NOT nbins==0.  The generated
# TB reads `count = nbins - 1` edges unconditionally (it mirrors the kernel's
# guard-free read; the extractor can't lower an `if nbins>1` guard).  For
# nbins==0 that count is -1, which the file read rejects, and nbins>max_nbins
# would overrun the fixed edges[max_nbins] buffer — so an nbins-based failure
# isn't drivable through the generated TB.  ndata==0 still exercises the >=1
# alloc clamp (data count is 0); the counts-alloc clamp is the same emitted
# expression (see CODEGEN_NOTES.md).
CSIM_CASES = [
    HistCase(ndata=37, nbins=1),
    HistCase(ndata=37, nbins=6),
    HistCase(ndata=200, nbins=12),
    HistCase(ndata=0, nbins=6),    # INVALID_NDATA
]


# ---------------------------------------------------------------------------
# Committed-figure workflow (the docs timing/burst diagrams)
# ---------------------------------------------------------------------------

EXAMPLE_DIR = Path(__file__).resolve().parent


def build_figures_dag():
    """DAG that renders the two docs figures and syncs them into docs/images."""
    from pysilicon.build.build import BuildDag
    try:
        from examples.shared_mem.shared_mem_figures import (
            GenerateBurstDiagramStep, GenerateTimingDiagramStep, SyncDocsFiguresStep,
        )
    except ModuleNotFoundError:  # direct execution from the example dir
        from shared_mem_figures import (  # type: ignore[no-redef]
            GenerateBurstDiagramStep, GenerateTimingDiagramStep, SyncDocsFiguresStep,
        )
    dag = BuildDag()
    dag.add(GenerateBurstDiagramStep(name="generate_burst_diagram"))
    dag.add(GenerateTimingDiagramStep(name="generate_timing_diagram"))
    dag.add(SyncDocsFiguresStep(name="sync_docs_figures"))
    return dag


def main() -> None:
    import argparse

    from pysilicon.build.build import BuildConfig

    parser = argparse.ArgumentParser(
        description="shared_mem figure workflow: render + sync the docs diagrams.")
    parser.add_argument("--through", metavar="STEP", default="sync_docs_figures",
                        help="Run the figure DAG up to and including this step.")
    parser.add_argument("--list-steps", action="store_true",
                        help="Print the figure step names and exit.")
    parser.add_argument("--force", action="store_true", help="Force all steps to rebuild.")
    args = parser.parse_args()

    dag = build_figures_dag()
    if args.list_steps:
        for name in dag.step_names():
            print(name)
        return
    dag.run(BuildConfig(root_dir=EXAMPLE_DIR), through=args.through, force=args.force)


if __name__ == "__main__":
    main()
