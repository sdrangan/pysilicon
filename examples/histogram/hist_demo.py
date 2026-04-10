from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pysilicon.build.build import CodeGenConfig
from pysilicon.build.streamutils import copy_streamutils
from pysilicon.hw.arrayutils import (
    gen_array_utils,
    get_nwords,
    read_array,
    read_uint32_file,
    write_array,
    write_uint32_file,
)
from pysilicon.hw.dataschema import DataArray, DataList, EnumField, FloatField, IntField, MemAddr
from pysilicon.toolchain import toolchain
from pysilicon.hw.memory import AddrUnit, Memory


EXAMPLE_DIR = Path(__file__).resolve().parent
INCLUDE_DIR = "include"
WORD_BW_SUPPORTED = [32, 64]
TxIdField = IntField.specialize(bitwidth=16, signed=False)
NdataField = IntField.specialize(bitwidth=32, signed=False)
NbinField = IntField.specialize(bitwidth=32, signed=False)
Uint32Field = IntField.specialize(bitwidth=32, signed=False, include_dir=INCLUDE_DIR)

"""
Parameters
"""
MAX_NDATA = 1024        # number of float32 data elements
MAX_NBINS = 32          # number of histogram bins
STREAM_DWIDTH = 32      # stream data width
MEM_DWIDTH = 32         # memory data width
MEM_AWIDTH = 64         # memory address width
MEM_AUNIT = AddrUnit.byte  # memory address unit

AddrField = MemAddr.specialize(bitwidth=MEM_AWIDTH, include_dir=INCLUDE_DIR)
Float32 = FloatField.specialize(bitwidth=32, include_dir=INCLUDE_DIR)

"""
Define the data schemas for the histogram accelerator
"""


class HistError(IntEnum):
    NO_ERROR = 0
    INVALID_NDATA = 1
    INVALID_NBINS = 2
    ADDRESS_ERROR = 3


HistErrorField = EnumField.specialize(enum_type=HistError, include_dir=INCLUDE_DIR)


class HistCmd(DataList):
    """Command descriptor for the histogram accelerator."""

    elements = {
        "tx_id": {
            "schema": TxIdField,
            "description": "Transaction ID for correlating command and response",
        },
        "data_addr": {
            "schema": AddrField,
            "description": "Base address of the input data buffer",
        },
        "bin_edges_addr": {
            "schema": AddrField,
            "description":("Base address of the output histogram bin edges.  " 
                  "There should be nbins-1 edge values.  "
                  "bin 0 will have values x < bin_edges[0], bin i will have values "
                  "bin_edges[i-1] <= x < bin_edges[i], and the last bin will have values "
                  "x >= bin_edges[nbins-2]"),
        },
        "ndata": {
            "schema": NdataField,
            "description": "Number of input data elements to histogram",
        },
        "nbins": {
            "schema": NbinField,
            "description": "Number of histogram bins to produce",
        },
        "cnt_addr": {
            "schema": AddrField,
            "description": "Base address of the output histogram counts buffer",
        },
    }
    include_dir = INCLUDE_DIR


class HistResp(DataList):
    """Response descriptor returned by the histogram accelerator."""

    elements = {
        "tx_id": {
            "schema": TxIdField,
            "description": "Echo of the transaction ID from the command",
        },
        "status": {
            "schema": HistErrorField,
            "description": "Histogram execution status code",
        },
    }
    include_dir = INCLUDE_DIR


SCHEMA_CLASSES = [
    HistErrorField,
    HistCmd,
    HistResp,
]


@dataclass(slots=True)
class HistSimResult:
    """Result bundle returned by HistTest.simulate."""

    cmd: HistCmd
    resp: HistResp
    counts: np.ndarray
    expected: np.ndarray

    @property
    def passed(self) -> bool:
        return np.array_equal(self.counts, self.expected)



class HistogramAccel(object):
    """
    Python model for the histogram accelerator.  This is where the core logic of the histogram computation lives, and it operates on the Memory model to read inputs and write outputs.  In a real implementation, this logic would be synthesized into hardware, but here we can use the full expressiveness of Python and numpy to implement it in a straightforward way.
    """

    def __init__(
            self,
            mem : Memory,
            max_ndata: int = MAX_NDATA,
            max_nbins: int = MAX_NBINS
            ):
        """
        Constructor with the dimensioning parameters and reference to the shared memory.

        Parameters:
        -----------
        mem: Memory
            Reference to the shared memory instance for reading inputs and writing outputs.
        max_ndata: int
            Maximum number of data elements that the accelerator can process in one command.
        max_nbins: int
            Maximum number of histogram bins that the accelerator can produce in one command.
        """
        self.max_ndata = max_ndata
        self.max_nbins = max_nbins
        self.mem = mem

    def compute_hist(
        self,
        cmd : HistCmd,
    ) -> HistResp:
        """
        Compute the histogram based on the input command descriptor.  This method reads the input data and bin edges from memory, computes the histogram counts, and writes the counts back to memory.  It also constructs a response descriptor with the appropriate status code.

        Parameters:
        -----------
        cmd: HistCmd
            The command descriptor containing the transaction ID, input data address, bin edges address, number of data elements, number of bins, and output counts address.

        Returns:
        --------
        HistResp
            The response descriptor containing the transaction ID and status code.
        """
        resp = HistResp()
        resp.tx_id = cmd.tx_id

        ndata = int(cmd.ndata)
        nbins = int(cmd.nbins)

        if ndata <= 0 or ndata > self.max_ndata:
            resp.status = HistError.INVALID_NDATA
            return resp

        if nbins <= 0 or nbins > self.max_nbins:
            resp.status = HistError.INVALID_NBINS
            return resp

        try:
            data_nwords = get_nwords(Float32, word_bw=self.mem.word_size, shape=ndata)
            data_words = self.mem.read(int(cmd.data_addr), nwords=data_nwords)
            data = np.asarray(read_array(data_words, elem_type=Float32, word_bw=self.mem.word_size, shape=ndata), dtype=np.float32)

            if nbins > 1:
                edge_shape = nbins - 1
                edge_nwords = get_nwords(Float32, word_bw=self.mem.word_size, shape=edge_shape)
                edge_words = self.mem.read(int(cmd.bin_edges_addr), nwords=edge_nwords)
                bin_edges = np.asarray(
                    read_array(edge_words, elem_type=Float32, word_bw=self.mem.word_size, shape=edge_shape),
                    dtype=np.float32,
                )
            else:
                bin_edges = np.array([], dtype=np.float32)

            bin_index = np.searchsorted(bin_edges, data, side="right")
            counts = np.bincount(bin_index, minlength=nbins).astype(np.uint32, copy=False)
            packed_counts = write_array(counts, elem_type=Uint32Field, 
                                        word_bw=self.mem.word_size)
            self.mem.write(int(cmd.cnt_addr), packed_counts)
        except ValueError:
            resp.status = HistError.ADDRESS_ERROR
            return resp

        resp.status = HistError.NO_ERROR
        return resp


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
        cfg = CodeGenConfig(root_dir=self.example_dir, util_dir=self.include_dir)
        generated_paths: list[Path] = []
        for schema_class in SCHEMA_CLASSES:
            generated_paths.append(schema_class.gen_include(cfg=cfg, word_bw_supported=WORD_BW_SUPPORTED))
        generated_paths.append(gen_array_utils(Float32, WORD_BW_SUPPORTED, cfg=cfg))
        generated_paths.append(gen_array_utils(Uint32Field, WORD_BW_SUPPORTED, cfg=cfg))
        copy_streamutils(cfg)
        return generated_paths

    def write_input_files(self, data_dir: Path | None = None) -> Path:
        """Write cmd, data, and bin-edge input files for the Vitis testbench.

        The Python model (``simulate()``) must have been run first so that the
        command descriptor and memory addresses are known.

        Parameters
        ----------
        data_dir : Path | None
            Directory in which to write the binary input files.  Defaults to
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

        self.cmd.write_uint32_file(data_dir / "cmd_data.bin")
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

    def test_vitis(self, cosim: bool = False) -> HistSimResult:
        """Run the Vitis kernel and testbench, then compare against the Python model.

        This method orchestrates the full file-based Vitis simulation workflow:

        1. Run the Python model (``simulate()``) if not already done.
        2. Generate Vitis HLS headers via ``gen_vitis_code()``.
        3. Write binary input files for the C++ testbench via ``write_input_files()``.
        4. Invoke Vitis HLS C-simulation (and, optionally, RTL co-simulation).
        5. Read the testbench output files and compare against the Python reference.

        Parameters
        ----------
        cosim : bool
            If ``True``, also run C-synthesis and RTL co-simulation after the
            C-simulation step.  Default is ``False``.

        Returns
        -------
        HistSimResult
            Result bundle with counts and response read from the Vitis output files.

        Raises
        ------
        RuntimeError
            If the Vitis simulation outputs do not match the Python model.
        """
        if self.cmd is None:
            self.simulate()

        self.gen_vitis_code()
        data_dir = self.write_input_files()

        tcl_args = ["--cosim"] if cosim else None
        toolchain.run_vitis_hls(
            self.example_dir / "run.tcl",
            work_dir=self.example_dir,
            args=tcl_args,
        )

        return self.read_vitis_outputs(data_dir)


def main() -> None:
    """Command-line entry point for the histogram accelerator example.

    Usage examples::

        # Python simulation only
        python hist_demo.py --skip-vitis

        # Python + Vitis C-simulation
        python hist_demo.py

        # Python + Vitis C-simulation + RTL co-simulation
        python hist_demo.py --cosim
    """
    parser = argparse.ArgumentParser(description="Run the histogram accelerator example.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for test data generation.")
    parser.add_argument("--ndata", type=int, default=37, help="Number of input data elements.")
    parser.add_argument("--nbins", type=int, default=6, help="Number of histogram bins.")
    parser.add_argument("--skip-vitis", action="store_true",
                        help="Only run the Python side of the example.")
    parser.add_argument("--cosim", action="store_true",
                        help="Also run RTL co-simulation after C-synthesis.")
    args = parser.parse_args()

    test = HistTest(seed=args.seed, ndata=args.ndata, nbins=args.nbins)
    test.gen_vitis_code()

    result = test.simulate()
    print(
        f"Python simulation: tx_id={result.resp.tx_id}, "
        f"status={result.resp.status.name}, "
        f"passed={result.passed}"
    )

    if args.skip_vitis:
        return

    try:
        vitis_result = test.test_vitis(cosim=args.cosim)
    except RuntimeError as exc:
        print(f"Vitis run failed: {exc}")
        return

    print(f"Vitis simulation matched Python model. counts={vitis_result.counts}")


if __name__ == "__main__":
    main()



