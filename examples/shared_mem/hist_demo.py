from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import numpy as np

from pysilicon.build.build import BuildConfig, BuildDag
from pysilicon.build.streamutils import StreamUtilsStep
from pysilicon.hw.arrayutils import (
    ArrayUtilsStep,
    get_nwords,
    read_array,
    read_uint32_file,
    write_array,
    write_uint32_file,
)
from pysilicon.hw.dataschema import DataSchemaStep
from pysilicon.hw.memory import AddrUnit, Memory
from pysilicon.toolchain import toolchain
from pysilicon.toolchain.stagetest import StageTest
from pysilicon.utils.vcd import AximmBeatType

# Data model moved to hist.py (the data-model home); re-exported here so the
# existing `from hist_demo import <schema>` consumers keep working until the
# monolith is retired.
try:
    from examples.shared_mem.hist import (  # noqa: F401
        AddrField, Float32, HistCmd, HistError, HistErrorField, HistResp,
        HistogramAccel, HistSimResult, INCLUDE_DIR, MAX_NBINS, MAX_NDATA,
        MEM_AUNIT, MEM_AWIDTH, MEM_DWIDTH, NbinField, NdataField,
        SCHEMA_CLASSES, STREAM_DWIDTH, TxIdField, Uint32Field,
    )
except ModuleNotFoundError:  # direct execution from the example dir
    from hist import (  # type: ignore[no-redef]  # noqa: F401
        AddrField, Float32, HistCmd, HistError, HistErrorField, HistResp,
        HistogramAccel, HistSimResult, INCLUDE_DIR, MAX_NBINS, MAX_NDATA,
        MEM_AUNIT, MEM_AWIDTH, MEM_DWIDTH, NbinField, NdataField,
        SCHEMA_CLASSES, STREAM_DWIDTH, TxIdField, Uint32Field,
    )


EXAMPLE_DIR = Path(__file__).resolve().parent
WORD_BW_SUPPORTED = [32, 64]
TRACE_LEVELS = ("none", "port", "all")
STAGES = ("csim", "csynth", "cosim", "generate_vcd", "extract_bursts")
VITIS_STAGES = ("csim", "csynth", "cosim")
STAGE_FLOW = StageTest.from_names(STAGES)
VITIS_STAGE_FLOW = StageTest.from_names(VITIS_STAGES)
DEFAULT_MAX_READ_BURST_LENGTH = 16
DEFAULT_MAX_WRITE_BURST_LENGTH = 16


def _stage_index(stage: str) -> int:
    return STAGE_FLOW.stage_index(stage)


def _vcd_trace_level(trace_level: str) -> str:
    return trace_level if trace_level in {"all", "port"} else "*"


def _remove_path_if_exists(path: Path) -> None:
    def _handle_remove_readonly(func, target, excinfo):
        _ = excinfo
        os.chmod(target, stat.S_IWRITE)
        func(target)

    if not path.exists():
        return
    if path.is_file() or path.is_symlink():
        try:
            path.unlink()
        except PermissionError:
            os.chmod(path, stat.S_IWRITE)
            path.unlink()
        return
    shutil.rmtree(path, onexc=_handle_remove_readonly)


def _json_scalar(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _json_hex_word(value, bitwidth: int) -> str:
    if bitwidth <= 0:
        raise ValueError("bitwidth must be positive.")

    int_value = int(_json_scalar(value))
    mask = (1 << bitwidth) - 1
    hex_width = (bitwidth + 3) // 4
    return f"0x{(int_value & mask):0{hex_width}x}"

class HistTest(object):
    """Stateful test/demo harness for the histogram accelerator flow."""

    def __init__(
            self,
            seed: int = 7,
            ndata: int = 37,
            nbins: int = 6,
            example_dir: Path = EXAMPLE_DIR,
            include_dir: str = INCLUDE_DIR,
            mem_dwidth: int = MEM_DWIDTH,
            mem_awidth: int = MEM_AWIDTH
    ):
        self.seed = int(seed)
        self.ndata = min(max(1, int(ndata)), MAX_NDATA)
        self.nbins = min(max(1, int(nbins)), MAX_NBINS)
        self.example_dir = Path(example_dir)
        self.include_dir = include_dir
        self.mem_dwidth = mem_dwidth
        self.mem_awidth = mem_awidth

        self.mem: Memory | None = None
        self.hist_accel: HistogramAccel | None = None
        self.data: np.ndarray | None = None
        self.bin_edges: np.ndarray | None = None
        self.expected: np.ndarray | None = None
        self.counts: np.ndarray | None = None
        self.cmd: HistCmd | None = None
        self.resp: HistResp | None = None
        self.data_addr: int | None = None
        self.edge_addr: int | None = None
        self.count_addr: int | None = None

    def gen_test_data(self) -> None:
        """
        Generate randomized input data and bin edges for a simulation run.
        """
        rng = np.random.default_rng(self.seed)
        self.data = rng.normal(loc=0.0, scale=1.25, size=self.ndata).astype(np.float32)
        self.bin_edges = np.sort(
            rng.uniform(-2.5, 2.5, size=max(self.nbins - 1, 0)).astype(np.float32)
        )


    def simulate(self) -> HistSimResult:
        """
        Run the Python model and store both observed and expected counts.
        """
        if self.data is None or self.bin_edges is None or self.cmd is None:
            self.gen_test_data()

        assert self.data is not None
        assert self.bin_edges is not None

        # Initialize memory.  We use byte addressing since we are emulating AXI4-style
        # interfaces which use byte addresses.  
        self.mem = Memory(
            word_size=self.mem_dwidth,
            addr_size=self.mem_awidth,
            addr_unit=AddrUnit.byte,
        )

        # Instantiate an accelerator connected to the memory
        self.hist_accel = HistogramAccel(self.mem)

        # Allocate memory for the input data, bin edges, and output counts
        # Note that the addresses are in bytes.
        nwords_data = get_nwords(elem_type=Float32, word_bw=self.mem.word_size, 
                                 shape=self.data.shape)
        nwords_edges = get_nwords(elem_type=Float32, word_bw=self.mem.word_size, 
                                  shape=self.bin_edges.shape)
        nwords_counts = get_nwords(elem_type=Uint32Field, word_bw=self.mem.word_size, 
                                   shape=self.nbins)
        self.data_addr = self.mem.alloc(nwords_data)
        self.edge_addr = self.mem.alloc(nwords_edges)
        self.count_addr = self.mem.alloc(nwords_counts)

        # Write the input data and bin edges to the allocated memory
        self.mem.write(
            self.data_addr,
            write_array(self.data, elem_type=Float32, word_bw=self.mem.word_size),
        )
        self.mem.write(
            self.edge_addr,
            write_array(self.bin_edges, elem_type=Float32, word_bw=self.mem.word_size),
        )


        # Create the command descriptor with the appropriate fields
        self.cmd = HistCmd(
            tx_id=self.seed,
            data_addr=self.data_addr,
            bin_edges_addr=self.edge_addr,
            ndata=self.ndata,
            nbins=self.nbins,
            cnt_addr=self.count_addr,
        )
 
        # Simulate the accelator and read back the results
        self.resp = self.hist_accel.compute_hist(self.cmd)

        # Read the histogram counts back from memory and compute the expected counts using numpy for comparison
        count_words = self.mem.read(
            self.count_addr,
            nwords=get_nwords(elem_type=Uint32Field, word_bw=self.mem.word_size, shape=self.nbins),
        )
        self.counts = read_array(
            count_words,
            elem_type=Uint32Field,
            word_bw=self.mem.word_size,
            shape=self.nbins,
        )

        # Compute expected counts using numpy for comparison
        self.expected = np.bincount(
            np.searchsorted(self.bin_edges, self.data, side="right"),
            minlength=self.nbins,
        ).astype(np.uint32)

        return HistSimResult(
            cmd=self.cmd,
            resp=self.resp,
            counts=self.counts,
            expected=self.expected,
        )


        
    def gen_vitis_code(self) -> list[Path]:
        """Generate schema and utility headers needed for the Vitis flow."""
        cfg = BuildConfig(root_dir=self.example_dir)
        dag = BuildDag()
        dag.add(StreamUtilsStep(output_dir=self.include_dir))
        schema_steps = [
            dag.add(DataSchemaStep(cls, word_bw_supported=WORD_BW_SUPPORTED, include_dir=self.include_dir))
            for cls in SCHEMA_CLASSES
        ]
        dag.add(ArrayUtilsStep(Float32, WORD_BW_SUPPORTED))
        dag.add(ArrayUtilsStep(Uint32Field, WORD_BW_SUPPORTED))
        results = dag.run(cfg)
        return [results[step.name].artifacts["include"] for step in schema_steps]

    def write_input_files(self, data_dir: Path | None = None) -> Path:
        """Write test data files for the Vitis testbench.

        Writes ``params.json`` (``tx_id``, ``ndata``, ``nbins``),
        ``data_array.bin``, and ``edges_array.bin`` (when ``nbins > 1``) to the
        data directory.  The testbench reads these files, performs memory
        allocations via ``MemMgr``, and constructs the ``HistCmd`` itself.

        The Python model (``simulate()``) must have been run first so that the
        input arrays and transaction parameters are known.

        Parameters
        ----------
        data_dir : Path | None
            Directory in which to write the files.  Defaults to
            ``<example_dir>/data``.

        Returns
        -------
        Path
            The directory containing the written files.
        """
        if self.cmd is None or self.data is None:
            self.simulate()

        if data_dir is None:
            data_dir = self.example_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Write scalar parameters so the testbench can reconstruct the HistCmd.
        params_path = data_dir / "params.json"
        params_path.write_text(
            json.dumps({"tx_id": int(self.cmd.tx_id), "ndata": self.ndata, "nbins": self.nbins},
                       indent=2),
            encoding="utf-8",
        )
        write_uint32_file(self.data, elem_type=Float32, file_path=data_dir / "data_array.bin",
                          nwrite=self.ndata)
        if len(self.bin_edges) > 0:
            write_uint32_file(self.bin_edges, elem_type=Float32, file_path=data_dir / "edges_array.bin",
                              nwrite=len(self.bin_edges))
        return data_dir

    def read_vitis_outputs(self, data_dir: Path) -> HistSimResult:
        """Read Vitis testbench output files and compare against the Python model.

        Parameters
        ----------
        data_dir : Path
            Directory containing the ``resp_data.bin`` and ``counts_array.bin``
            files written by the Vitis testbench.

        Returns
        -------
        HistSimResult
            Result bundle with counts and response read from the Vitis output files.

        Raises
        ------
        RuntimeError
            If the response status or histogram counts do not match the Python model.
        """
        resp = HistResp().read_uint32_file(data_dir / "resp_data.bin")
        counts = np.asarray(
            read_uint32_file(data_dir / "counts_array.bin", elem_type=Uint32Field, shape=self.nbins),
            dtype=np.uint32,
        )

        if not resp.is_close(self.resp):
            raise RuntimeError(
                f"Response mismatch after Vitis simulation.\n"
                f"  got:      tx_id={resp.tx_id}, status={resp.status}\n"
                f"  expected: tx_id={self.resp.tx_id}, status={self.resp.status}"
            )
        if not np.array_equal(counts, self.expected):
            raise RuntimeError(
                f"Count mismatch after Vitis simulation:\n"
                f"  got:      {counts}\n"
                f"  expected: {self.expected}"
            )

        return HistSimResult(cmd=self.cmd, resp=resp, counts=counts, expected=self.expected)

    def _expected_burst_summary(self) -> dict[str, object]:
        if self.cmd is None:
            self.simulate()

        assert self.cmd is not None

        max_read_burst_length, max_write_burst_length = self._detect_axi_max_burst_lengths()
        bytes_per_word = self.mem_dwidth // 8

        read_regions = [
            {
                "name": "data",
                "addr": int(self.cmd.data_addr),
                "nwords": int(get_nwords(Float32, word_bw=self.mem_dwidth, shape=self.ndata)),
            },
        ]
        if self.nbins > 1:
            read_regions.append(
                {
                    "name": "bin_edges",
                    "addr": int(self.cmd.bin_edges_addr),
                    "nwords": int(get_nwords(Float32, word_bw=self.mem_dwidth, shape=self.nbins - 1)),
                }
            )

        write_regions = [
            {
                "name": "counts",
                "addr": int(self.cmd.cnt_addr),
                "nwords": int(get_nwords(Uint32Field, word_bw=self.mem_dwidth, shape=self.nbins)),
            }
        ]

        read_bursts = [
            burst
            for region in read_regions
            for burst in self._split_region_into_bursts(region, max_read_burst_length, bytes_per_word)
        ]
        write_bursts = [
            burst
            for region in write_regions
            for burst in self._split_region_into_bursts(region, max_write_burst_length, bytes_per_word)
        ]

        return {
            "max_read_burst_length": max_read_burst_length,
            "max_write_burst_length": max_write_burst_length,
            "read_burst_count": len(read_bursts),
            "write_burst_count": len(write_bursts),
            "read_regions": read_regions,
            "write_regions": write_regions,
            "read_bursts": read_bursts,
            "write_bursts": write_bursts,
        }

    def _detect_axi_max_burst_lengths(self) -> tuple[int, int]:
        candidates = [
            self.example_dir / "pysilicon_hist_proj" / "solution1" / "impl" / "verilog" / "hist.v",
            self.example_dir / "pysilicon_hist_proj" / "solution1" / "impl" / "vhdl" / "hist.vhd",
        ]
        patterns = {
            "read": [
                re.compile(r"MAX_READ_BURST_LENGTH\s*=>\s*(\d+)"),
                re.compile(r"\.MAX_READ_BURST_LENGTH\s*\(\s*(\d+)\s*\)"),
            ],
            "write": [
                re.compile(r"MAX_WRITE_BURST_LENGTH\s*=>\s*(\d+)"),
                re.compile(r"\.MAX_WRITE_BURST_LENGTH\s*\(\s*(\d+)\s*\)"),
            ],
        }

        read_length = DEFAULT_MAX_READ_BURST_LENGTH
        write_length = DEFAULT_MAX_WRITE_BURST_LENGTH

        for candidate in candidates:
            if not candidate.exists():
                continue
            text = candidate.read_text(encoding="utf-8", errors="ignore")
            for pattern in patterns["read"]:
                match = pattern.search(text)
                if match is not None:
                    read_length = int(match.group(1))
                    break
            for pattern in patterns["write"]:
                match = pattern.search(text)
                if match is not None:
                    write_length = int(match.group(1))
                    break
            if read_length and write_length:
                break

        return read_length, write_length

    @staticmethod
    def _split_region_into_bursts(region: dict[str, object], max_burst_length: int, bytes_per_word: int) -> list[dict[str, int | str]]:
        addr = int(region["addr"])
        nwords = int(region["nwords"])
        name = str(region["name"])
        bursts: list[dict[str, int | str]] = []

        remaining = nwords
        offset_words = 0
        while remaining > 0:
            burst_words = min(remaining, max_burst_length)
            bursts.append(
                {
                    "name": name,
                    "addr": addr + offset_words * bytes_per_word,
                    "nwords": burst_words,
                }
            )
            offset_words += burst_words
            remaining -= burst_words

        return bursts

    @staticmethod
    def _burst_layout(bursts: list[dict[str, object]], len_key: str) -> list[dict[str, int]]:
        layout = []
        for burst in bursts:
            addr = burst.get("addr")
            if addr is None:
                continue
            raw_len = burst.get(len_key)
            nwords = len(burst.get("data", []))
            if raw_len is not None:
                nwords = int(_json_scalar(raw_len)) + 1
            layout.append({
                "addr": int(_json_scalar(addr)),
                "nwords": int(nwords),
            })
        return layout

    @staticmethod
    def _burst_to_jsonable(burst: dict[str, object], data_bitwidth: int) -> dict[str, object]:
        beat_types = [int(_json_scalar(value)) for value in burst.get("beat_type", [])]
        data_values = [_json_scalar(value) for value in np.asarray(burst.get("data", [])).tolist()]
        return {
            "addr": None if burst.get("addr") is None else int(_json_scalar(burst["addr"])),
            "start_idx": int(_json_scalar(burst["start_idx"])),
            "tstart": float(_json_scalar(burst["tstart"])),
            "data_start_idx": None if burst.get("data_start_idx") is None else int(_json_scalar(burst["data_start_idx"])),
            "data_end_idx": None if burst.get("data_end_idx") is None else int(_json_scalar(burst["data_end_idx"])),
            "data_tstart": None if burst.get("data_tstart") is None else float(_json_scalar(burst["data_tstart"])),
            "data_tend": None if burst.get("data_tend") is None else float(_json_scalar(burst["data_tend"])),
            "queue_wait_cycles": None if burst.get("queue_wait_cycles") is None else int(_json_scalar(burst["queue_wait_cycles"])),
            "beat_type": beat_types,
            "beat_type_names": [AximmBeatType(value).name.lower() for value in beat_types],
            "data": data_values,
            "data_hex": [_json_hex_word(value, data_bitwidth) for value in data_values],
            "awlen": None if burst.get("awlen") is None else int(_json_scalar(burst["awlen"])),
            "arlen": None if burst.get("arlen") is None else int(_json_scalar(burst["arlen"])),
        }

    def extract_bursts(
        self,
        vcd_path: str | Path | None = None,
        output_json: str | Path | None = None,
    ) -> dict[str, object]:
        """Extract AXI-MM bursts from the histogram VCD and write a JSON report."""
        if self.cmd is None:
            self.simulate()

        from vcdvcd import VCDVCD
        from pysilicon.utils.vcd import VcdParser

        if vcd_path is None:
            vcd_path = self.example_dir / "vcd" / "dump.vcd"
        vcd_path = Path(vcd_path)
        if not vcd_path.exists():
            raise FileNotFoundError(f"VCD file not found: {vcd_path}")

        if output_json is None:
            output_json = vcd_path.with_name("burst_info.json")
        output_json = Path(output_json)

        vcd = VCDVCD(str(vcd_path), signals=None, store_tvs=True)
        vp = VcdParser(vcd)
        clk_sig = vp.add_clock_signal()
        aximm_sigs, aximm_bw = vp.add_aximm_signals(
            prefix="m_axi_gmem_",
            dir="both",
            lite_only=False,
            short_name_prefix="gmem_",
        )
        write_bursts, read_bursts, clk_period = vp.extract_aximm_bursts(
            clk_name=clk_sig,
            aximm_sigs=aximm_sigs,
        )

        expected = self._expected_burst_summary()
        report = {
            "vcd_path": str(vcd_path),
            "output_json": str(output_json),
            "clock_signal": clk_sig,
            "clk_period_ns": float(clk_period),
            "beat_type_enum": {member.name.lower(): int(member) for member in AximmBeatType},
            "aximm_bitwidths": {key: int(value) for key, value in aximm_bw.items()},
            "expected": expected,
            "actual": {
                "write_burst_count": len(write_bursts),
                "read_burst_count": len(read_bursts),
                "write_bursts": [self._burst_to_jsonable(burst, data_bitwidth=int(aximm_bw["WDATA"])) for burst in write_bursts],
                "read_bursts": [self._burst_to_jsonable(burst, data_bitwidth=int(aximm_bw["RDATA"])) for burst in read_bursts],
                "write_burst_layout": self._burst_layout(write_bursts, "awlen"),
                "read_burst_layout": self._burst_layout(read_bursts, "arlen"),
            },
        }
        report["checks"] = {
            "write_burst_count_matches": report["actual"]["write_burst_count"] == expected["write_burst_count"],
            "read_burst_count_matches": report["actual"]["read_burst_count"] == expected["read_burst_count"],
            "write_burst_layout_matches": report["actual"]["write_burst_layout"] == [
                {"addr": burst["addr"], "nwords": burst["nwords"]} for burst in expected["write_bursts"]
            ],
            "read_burst_layout_matches": report["actual"]["read_burst_layout"] == [
                {"addr": burst["addr"], "nwords": burst["nwords"]} for burst in expected["read_bursts"]
            ],
        }
        report["validated"] = all(report["checks"].values())

        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

        if not report["validated"]:
            raise RuntimeError(
                "Unexpected AXI-MM burst counts in extracted VCD data. "
                f"Expected read/write burst counts {expected['read_burst_count']}/{expected['write_burst_count']}, "
                f"got {report['actual']['read_burst_count']}/{report['actual']['write_burst_count']}. "
                f"See {output_json}."
            )

        return report

    def test_vitis(
        self,
        start_at: str = "csim",
        through: str = "csim",
        trace_level: str = "none",
        live_output: bool = False,
        reset_includes: bool = False,
    ) -> HistSimResult | dict[str, object] | None:
        """Run a contiguous stage range of the histogram Vitis flow.

        This method orchestrates an inclusive stage range of the file-based
        histogram flow. The supported stages are ``csim``, ``csynth``,
        ``cosim``, ``generate_vcd``, and ``extract_bursts``.

        1. Run the Python model (``simulate()``) if not already done.
        2. If ``start_at == 'csim'``, regenerate Vitis headers and input files.
        3. Run Vitis HLS for the requested inclusive range of stages.
        4. When C-simulation or RTL co-simulation is executed, compare the
           produced outputs against the Python reference.
        5. Optionally generate a VCD after the Vitis stages complete.
        6. Optionally extract AXI-MM bursts from the generated VCD.

        Parameters
        ----------
        start_at : str
            First stage to execute. Defaults to ``'csim'``. Starting at any
            later stage reuses the existing Vitis project without resetting it.
        through : str
            Final stage to execute. Defaults to ``'csim'``.
        trace_level : str
            RTL co-simulation trace level passed through to ``cosim_design``.
            Supported values are ``none``, ``port``, and ``all``. When the
            stage range includes ``generate_vcd`` or ``extract_bursts``,
            ``port`` and ``all`` are passed through directly; ``none`` falls
            back to ``'*'`` for the VCD step.
        live_output : bool
            If ``True``, stream Vitis stdout/stderr directly to the terminal
            while the subprocess runs instead of buffering it for later.
        reset_includes : bool
            If ``True`` and ``start_at == 'csim'``, also delete the generated
            ``include/`` directory before regenerating the Vitis helper headers.

        Returns
        -------
        HistSimResult | dict[str, object] | None
            Result bundle with counts and response read from the Vitis output
            files when the executed range includes ``csim`` or ``cosim``.
            Returns the burst report for ranges ending in ``extract_bursts``.
            Returns ``None`` for ranges that only execute synthesis and/or VCD
            generation.

        Raises
        ------
        RuntimeError
            If the requested Vitis project state does not exist or if the Vitis
            simulation outputs do not match the Python model.
        """
        if trace_level not in TRACE_LEVELS:
            raise ValueError(
                f"Unsupported trace level '{trace_level}'. Expected one of {TRACE_LEVELS}."
            )

        start_at, through = STAGE_FLOW.validate_range(start_at, through)
        start_idx = _stage_index(start_at)
        through_idx = _stage_index(through)

        if self.cmd is None:
            self.simulate()

        data_dir = self.example_dir / "data"
        logs_dir = self.example_dir / "logs"
        project_dir = self.example_dir / "pysilicon_hist_proj"

        if start_at == "csim":
            _remove_path_if_exists(project_dir)
            _remove_path_if_exists(logs_dir)
            if reset_includes:
                _remove_path_if_exists(self.example_dir / self.include_dir)
            self.gen_vitis_code()
            data_dir = self.write_input_files()
        else:
            if start_at in {"csynth", "cosim", "generate_vcd"} and not project_dir.exists():
                raise RuntimeError(
                    f"Cannot start at '{start_at}' without an existing project at {project_dir}."
                )
            if start_at in {"csynth", "cosim", "generate_vcd"}:
                solution_dir = project_dir / "solution1"
                if not solution_dir.exists():
                    raise RuntimeError(
                        f"Cannot start at '{start_at}' without an existing solution at {solution_dir}."
                    )
            if start_at in {"csynth", "cosim"} and not data_dir.exists():
                raise RuntimeError(
                    f"Cannot start at '{start_at}' without existing Vitis input data at {data_dir}."
                )

        vitis_result: HistSimResult | None = None
        vitis_stdout = ""
        vitis_stderr = ""

        if start_idx <= _stage_index("cosim"):
            vitis_through = through if through in VITIS_STAGES else "cosim"
            executed_vitis_stages = VITIS_STAGE_FLOW.range_names(start_at, vitis_through)

            print(
                f"Performing Vitis stages {start_at} through {vitis_through}. "
                "This may take minutes."
            )
            if "cosim" in executed_vitis_stages:
                print(f"Trace level: {trace_level}")

            try:
                result = toolchain.run_vitis_hls(
                    self.example_dir / "run.tcl",
                    work_dir=self.example_dir,
                    capture_output=not live_output,
                    env={
                        "PYSILICON_HIST_START_AT": start_at,
                        "PYSILICON_HIST_THROUGH": vitis_through,
                        "PYSILICON_HIST_TRACE_LEVEL": trace_level,
                    },
                )
            except subprocess.CalledProcessError as exc:
                if exc.stdout:
                    print(exc.stdout)
                if exc.stderr:
                    print(exc.stderr)
                raise RuntimeError(
                    "Vitis execution failed. "
                    f"See logs under {logs_dir} and generated files under {project_dir}."
                ) from exc

            vitis_stdout = result.stdout or ""
            vitis_stderr = result.stderr or ""

            if any(stage in executed_vitis_stages for stage in ("csim", "cosim")):
                vitis_result = self.read_vitis_outputs(data_dir)

        if vitis_stdout:
            print(vitis_stdout)
        if vitis_stderr:
            print(vitis_stderr)

        vcd_path: Path | None = None
        generate_vcd_idx = _stage_index("generate_vcd")
        if start_idx <= generate_vcd_idx <= through_idx:
            vcd_path = self.generate_vcd(trace_level=_vcd_trace_level(trace_level))
            print(f"VCD written to: {vcd_path}")

        if through == "extract_bursts":
            report = self.extract_bursts(vcd_path=vcd_path)
            print(
                "Burst extraction validated. "
                f"read_bursts={report['actual']['read_burst_count']}, "
                f"write_bursts={report['actual']['write_burst_count']}"
            )
            print(f"Burst report written to: {report['output_json']}")
            return report

        return vitis_result

    def generate_vcd(
        self,
        output_vcd: str = "dump.vcd",
        soln: str | None = "solution1",
        trace_level: str = "*",
    ) -> Path:
        """
        Generate a VCD file by re-running the Vivado RTL simulation.

        Delegates to :func:`pysilicon.scripts.xsim_vcd.run_xsim_vcd`.
        Requires Vivado/xsim installed on Windows.

        Parameters
        ----------
        output_vcd : str
            Output VCD filename written inside a ``vcd/`` subdirectory.
        soln : str | None
            Solution name inside the component directory.
        trace_level : str
            VCD trace level (``'*'`` for all signals, ``'port'`` for ports only).

        Returns
        -------
        Path
            Absolute path to the written VCD file.
        """
        from pysilicon.scripts.xsim_vcd import run_xsim_vcd
        return run_xsim_vcd(
            top="hist",
            comp="pysilicon_hist_proj",
            out=output_vcd,
            soln=soln,
            trace_level=trace_level,
            workdir=self.example_dir,
        )


def main() -> None:
    """Command-line entry point for the histogram accelerator example.

    Usage examples::

        # Python simulation only
        python hist_demo.py --skip_vitis

        # Python + Vitis C-simulation
        python hist_demo.py

        # Python + Vitis C-simulation + C-synthesis
        python hist_demo.py --through csynth

        # Python + Vitis C-simulation + RTL co-simulation
        python hist_demo.py --through cosim

        # Reuse an existing project to run synthesis and co-simulation only
        python hist_demo.py --start_at csynth --through cosim

        # Run through VCD generation with full waveform tracing
        python hist_demo.py --through generate_vcd --trace_level all

        # Run through burst extraction and write a JSON report
        python hist_demo.py --through extract_bursts --trace_level port

        # Python + Vitis C-simulation with live Vitis output in the terminal
        python hist_demo.py --live_output

        # Generate a VCD from an existing traced co-sim run
        python hist_demo.py --start_at generate_vcd --through generate_vcd

        # Extract bursts from an existing VCD
        python hist_demo.py --start_at extract_bursts --through extract_bursts
    """
    parser = argparse.ArgumentParser(description="Run the histogram accelerator example.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for test data generation.")
    parser.add_argument("--ndata", type=int, default=37, help="Number of input data elements.")
    parser.add_argument("--nbins", type=int, default=6, help="Number of histogram bins.")
    parser.add_argument("--skip_vitis", action="store_true",
                        help="Only run the Python side of the example.")
    parser.add_argument(
        "--start_at",
        choices=STAGES,
        default="csim",
        help="First stage to execute. Starting at csim clears the Vitis project and logs.",
    )
    parser.add_argument(
        "--reset-includes",
        action="store_true",
        help="When starting at csim, also delete the generated include directory before regenerating it.",
    )
    parser.add_argument(
        "--through",
        choices=STAGES,
        default="csim",
        help="Final stage to execute.",
    )
    parser.add_argument(
        "--trace_level",
        choices=TRACE_LEVELS,
        default="none",
        help="RTL co-simulation trace level passed to Vitis.",
    )
    parser.add_argument(
        "--live_output",
        action="store_true",
        help="Stream Vitis stdout/stderr directly to the terminal while it runs.",
    )
    args = parser.parse_args()

    if _stage_index(args.start_at) > _stage_index(args.through):
        parser.error(f"--start_at {args.start_at} must not come after --through {args.through}.")

    test = HistTest(seed=args.seed, ndata=args.ndata, nbins=args.nbins)

    if args.start_at == "csim" or args.skip_vitis:
        result = test.simulate()
        print(
            f"Python simulation: tx_id={result.resp.tx_id}, "
            f"status={result.resp.status.name}, "
            f"passed={result.passed}"
        )

    if args.skip_vitis:
        return

    try:
        vitis_result = test.test_vitis(
            start_at=args.start_at,
            through=args.through,
            trace_level=args.trace_level,
            live_output=args.live_output,
            reset_includes=args.reset_includes,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"Vitis run failed: {exc}")
        return

    if isinstance(vitis_result, HistSimResult):
        print(f"Vitis simulation matched Python model. counts={vitis_result.counts}")
    elif isinstance(vitis_result, dict):
        print(
            "Burst extraction report generated. "
            f"output_json={vitis_result.get('output_json')}"
        )


if __name__ == "__main__":
    main()



