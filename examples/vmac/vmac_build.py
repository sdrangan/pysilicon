"""VMAC conformance + throughput BuildDag — Python golden vs Vitis, bit-exact.

The single ``vmac_compute`` datapath (the hand-written ``vmac_compute_impl.tpp`` hook, lowered
onto the **generated ComplexField serialization** ``read_array_elem`` / ``write_array_elem``)
is checked against :meth:`examples.vmac.vmac.VmacAccel.execute` — the Python golden — for every
op-flag / rounding / saturation config, bit-for-bit.  The Python model is the spec: if Vitis
ever disagrees, fix the kernel, never loosen the compare.

Structure (mirrors :mod:`examples.shared_mem.hist_build` / :mod:`examples.schemas.complex.complex_build`):

- **Structural config** = the synthesis-time widths the kernel is templated on
  (``data_bw`` / ``int_bits`` / ``acc_bw`` / ``out_bw`` / ``q_rnd`` / ``o_sat`` / ``mem_dwidth``).
  ``q_rnd`` / ``o_sat`` / ``out_bw`` are compile-time (``ap_fixed`` Q/O, output width), so each
  distinct combination is its **own** synthesized kernel → its own generated sources + csim run.
- **Runtime case** = one ``VmacCmd`` (op flags + geometry + operands) run within a structural
  config.  All of a config's cases share one compiled testbench and one Vitis csim invocation
  (a manifest of ``cmd``/``mem`` vector files), so the whole sweep is a handful of csim runs.

The conformance runs at ``mem_dwidth = 2·data_bw`` (PF = 1: one complex column per word) so the
layout is trivially word-aligned; the PF > 1 lane packing is validated by the throughput cosim
(``CosimStep`` checks its output against the golden too) as ``mem_dwidth`` is swept up.

CLI::

    python vmac_build.py --through gen          # write headers + per-config TB + vectors
    python vmac_build.py --through py_sim        # + golden-vs-oracle parity (no Vitis)
    python vmac_build.py --through csim          # + the bit-exact Vitis conformance (Vitis)
    python vmac_build.py --through extract_cosim_timing   # + throughput sweep (Vitis)
    python vmac_build.py --list-steps
"""
from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from waveflow.build.build import BuildConfig, BuildDag, BuildStep, SourceStep
from waveflow.build.cli import run_dag_cli
from waveflow.build.streamutils import StreamUtilsStep
from waveflow.hw.arrayutils import (
    ArrayUtilsStep, _array_utils_filename, _array_utils_namespace, write_array,
)
from waveflow.hw.complexfield import ComplexField
from waveflow.hw.dataschema import DataArray, DataSchemaStep
from waveflow.hw.fixpoint import FixedField
from waveflow.toolchain import toolchain
from waveflow.utils import complexutils as cx
from waveflow.utils.fixputils import Format

try:
    from examples.vmac.vmac import VmacAccel
    from examples.vmac.vmac_cmd import Region, Scalar, VmacCmd
except ModuleNotFoundError:  # direct execution from the example dir
    from vmac import VmacAccel  # type: ignore[no-redef]
    from vmac_cmd import Region, Scalar, VmacCmd  # type: ignore[no-redef]

_SOURCE_DIR = Path(__file__).resolve().parent
_BUILD_DIR = Path(__file__).resolve().parents[2] / "waveflow" / "build"  # complex_utils.hpp / wf_cint.h
INCLUDE_DIR = "include"
WORD_BW_SUPPORTED = [32, 64]                 # cmd schema word widths
MEM_WORD_BWS = [16, 32, 64, 128]             # m_axi widths the array-utils support (PF sweep)
CMD_NWORDS = 16                              # VmacCmd.serialize(word_bw=32) word count
MAX_COLS_CAP = 16                            # acc[] capacity in the kernel (>= any case n_cols)
MEM_AWIDTH = 32                              # cmd address width (the kernel's MEM_AWIDTH param)
# the hand-written hook + the complex-arithmetic header it pulls in (no integer-code bridging)
HOOK_FILES = ("vmac_compute_impl.tpp",)
BUILD_HDRS = ("complex_utils.hpp", "wf_cint.h")


# --- structural configs (compile-time kernel widths) --------------------------
@dataclass(frozen=True)
class StructCfg:
    out_bw: int
    q_rnd: int
    o_sat: int
    mem_dwidth: int = 16                      # PF = mem_dwidth / (2*data_bw); 16 -> PF=1
    data_bw: int = 8
    int_bits: int = 4
    acc_bw: int = 48

    @property
    def name(self) -> str:
        return f"ob{self.out_bw}_q{self.q_rnd}_o{self.o_sat}_m{self.mem_dwidth}"

    @property
    def f_in(self) -> int:
        return self.data_bw - self.int_bits

    @property
    def out_int(self) -> int:
        return self.out_bw - self.f_in       # F_out = F_in (structural normalized MAC)

    def accel(self) -> VmacAccel:
        return VmacAccel(
            mem_dwidth=self.mem_dwidth, mem_awidth=32, data_bw=self.data_bw,
            int_bits=self.int_bits, acc_bw=self.acc_bw, out_bw=self.out_bw,
            q_rnd=self.q_rnd, o_sat=self.o_sat,
        )

    def in_elem(self) -> type:
        return ComplexField.specialize(FixedField.specialize(self.data_bw, self.int_bits, True))

    def out_elem(self) -> type:
        return ComplexField.specialize(FixedField.specialize(self.out_bw, self.out_int, True))


# The conformance sweeps the 4 rounding/saturation modes (trn/wrap x rnd/sat) at
# out_bw == data_bw, so the output element equals the input element and the in/out lane
# packing is uniform over the flat ap_uint<MEM_BW> image.  (A wider output element, e.g.
# out_bw=12 -> 24-bit element, would not co-pack with the 16-bit input words in one flat
# image; that wide-output numeric edge is covered by tests/examples/test_vmac_golden's
# test_cg_rnorm at the Python-golden level.)
STRUCT_CFGS = [
    StructCfg(out_bw=8, q_rnd=0, o_sat=0),
    StructCfg(out_bw=8, q_rnd=1, o_sat=0),
    StructCfg(out_bw=8, q_rnd=0, o_sat=1),
    StructCfg(out_bw=8, q_rnd=1, o_sat=1),
]


# --- operand helpers (stored-int (re, im) pairs) ------------------------------
def _pair(re, im=None):
    re = np.asarray(re, dtype=np.int64)
    im = np.zeros_like(re) if im is None else np.asarray(im, dtype=np.int64)
    return (re, im)


def _flat(pair):
    re = np.asarray(pair[0]).ravel()
    im = np.asarray(pair[1]).ravel()
    return cx.make_complex(re, im, Format(8, 4, True))   # dtype only; format irrelevant


def _scalar_field(s, addr):
    if np.ndim(s[0]) == 0:
        return {"direct": 1, "re": int(s[0]), "im": int(s[1]), "addr": 0, "stride": 0}
    return {"direct": 0, "re": 0, "im": 0, "addr": int(addr), "stride": 1}


def build(accel: VmacAccel, flags: dict, a, b, c, alpha, beta):
    """Lay a/b/c (+ per-column alpha/beta) into mem (row-major) and build the matching Cmd.

    Returns ``(cmd, mem)`` with ``mem`` the structured complex image (pre-execute)."""
    n, m = a[0].shape
    nm = n * m
    blocks, addr, cur = [], {}, 0
    for name, op in (("a", a), ("b", b), ("c", c)):
        addr[name] = cur
        blocks.append(_flat(op))
        cur += nm
    alpha_pc, beta_pc = np.ndim(alpha[0]) > 0, np.ndim(beta[0]) > 0
    if alpha_pc:
        addr["alpha"] = cur
        blocks.append(_flat(alpha))
        cur += m
    if beta_pc:
        addr["beta"] = cur
        blocks.append(_flat(beta))
        cur += m
    addr["d"] = cur
    cur += nm                                            # dst region (>= reduced m)
    mem = cx.make_complex(np.zeros(cur), np.zeros(cur), Format(8, 4, True))
    order = ["a", "b", "c"] + (["alpha"] if alpha_pc else []) + (["beta"] if beta_pc else [])
    for name, blk in zip(order, blocks):
        mem[addr[name]:addr[name] + len(blk)] = blk

    cmd = accel.Cmd()
    cmd.n_rows, cmd.n_cols = n, m
    for name in ("a", "b", "c", "d"):
        setattr(cmd, name, {"addr": addr[name], "row_stride": m})
    cmd.alpha = _scalar_field(alpha, addr.get("alpha"))
    cmd.beta = _scalar_field(beta, addr.get("beta"))
    for f in ("b_one", "c_zero", "b_conj", "reduce_rows"):
        setattr(cmd, f, int(flags.get(f, 0)))
    return cmd, mem


# --- the independent oracle (exact rational; the non-Vitis correctness anchor) ----
def _q_real(value: Fraction, out_frac: int, W: int, q_rnd: bool, o_sat: bool) -> int:
    scaled = value * (Fraction(2) ** out_frac)
    f = math.floor(scaled + Fraction(1, 2)) if q_rnd else math.floor(scaled)
    if o_sat:
        return max(-(1 << (W - 1)), min((1 << (W - 1)) - 1, f))
    y = f & ((1 << W) - 1)
    return y - (1 << W) if (y >> (W - 1)) & 1 else y


def oracle(cfg: StructCfg, flags: dict, a, b, c, alpha, beta):
    """Exact (re, im) Fraction arithmetic — independent of the production golden."""
    F = cfg.f_in
    scale = Fraction(2) ** F
    n, m = a[0].shape

    def fr(x):
        return Fraction(int(x), 1) / scale

    def at(op, i, j):
        return (fr(op[0][i, j]), fr(op[1][i, j]))

    def scal(s, j):
        if np.ndim(s[0]) == 0:
            return (fr(s[0]), fr(s[1]))
        return (fr(s[0][j]), fr(s[1][j]))

    def cmul(x, y):
        return (x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0])

    cols_re, cols_im = [], []
    for j in range(m):
        terms = []
        for i in range(n):
            av = at(a, i, j)
            if flags.get("b_one"):
                ab = av
            else:
                bv = at(b, i, j)
                if flags.get("b_conj"):
                    bv = (bv[0], -bv[1])
                ab = cmul(av, bv)
            t = cmul(scal(alpha, j), ab)
            if not flags.get("c_zero"):
                bc = cmul(scal(beta, j), at(c, i, j))
                t = (t[0] + bc[0], t[1] + bc[1])
            terms.append(t)
        if flags.get("reduce_rows"):
            acc = terms[0]
            for t in terms[1:]:
                acc = (acc[0] + t[0], acc[1] + t[1])
            rows = [acc]
        else:
            rows = terms
        cols_re.append([_q_real(r[0], F, cfg.out_bw, cfg.q_rnd, cfg.o_sat) for r in rows])
        cols_im.append([_q_real(r[1], F, cfg.out_bw, cfg.q_rnd, cfg.o_sat) for r in rows])
    re = np.array(cols_re, dtype=np.int64).T
    im = np.array(cols_im, dtype=np.int64).T
    if flags.get("reduce_rows"):
        re, im = re[0], im[0]
    return re, im


# --- runtime cases ------------------------------------------------------------
def _runtime_cases():
    """The op-flag / operand cases for one structural config (the CG + general matrix ops).

    Operands are sized to exercise rounding and saturation so the structural q_rnd / o_sat
    modes genuinely matter.  The wide-output config (out_bw=12) adds the CG |R|^2 case (whose
    non-negative magnitude needs the extra integer headroom to clamp rather than wrap)."""
    rng = np.random.default_rng(2)
    n, m = 4, 3

    def cop():
        return _pair(rng.integers(-60, 61, (n, m)), rng.integers(-60, 61, (n, m)))

    a, b, c = cop(), cop(), cop()
    alpha_s, beta_s = _pair(20, -8), _pair(-12, 5)
    alpha_pc = _pair(rng.integers(-30, 31, m), rng.integers(-30, 31, m))
    beta_pc = _pair(rng.integers(-30, 31, m), rng.integers(-30, 31, m))

    flagsets = [
        ("scaled_copy", dict(b_one=1, c_zero=1)),
        ("hadamard", dict(c_zero=1)),
        ("conj_hadamard", dict(b_conj=1, c_zero=1)),
        ("full_mac", dict()),
        ("axpbc", dict(b_one=1)),
        ("col_sum", dict(b_one=1, c_zero=1, reduce_rows=1)),
        ("conj_inner", dict(b_conj=1, c_zero=1, reduce_rows=1)),
        ("reduced_mac", dict(reduce_rows=1)),
    ]
    cases = []
    for label, flags in flagsets:
        for scl, (al, be) in [("s", (alpha_s, beta_s)), ("pc", (alpha_pc, beta_pc))]:
            cases.append((f"{label}_{scl}", flags, a, b, c, al, be))

    # rounding/saturation-triggering singletons
    sat_op = _pair([[112]]), _pair([[112]])              # 7.0 * 7.0 = 49 -> over +-8 range
    cases.append(("saturate", dict(c_zero=1), sat_op[0], sat_op[1],
                  _pair([[0]]), _pair(16), _pair(0)))
    rr = np.random.default_rng(9)
    ra, rb = _pair(rr.integers(-120, 121, (3, 2))), _pair(rr.integers(-120, 121, (3, 2)))
    cases.append(("round", dict(c_zero=1), ra, rb, _pair(np.zeros((3, 2))), _pair(16), _pair(0)))

    # CG ops
    cg = np.random.default_rng(11)
    P = _pair(cg.integers(-100, 101, (5, 3)), cg.integers(-100, 101, (5, 3)))
    X = _pair(cg.integers(-100, 101, (5, 3)), cg.integers(-100, 101, (5, 3)))
    neg_alpha = _pair(-cg.integers(-30, 31, 3), -cg.integers(-30, 31, 3))
    cases.append(("cg_axpy", dict(b_one=1), P, _pair(np.zeros((5, 3)), np.zeros((5, 3))),
                  X, neg_alpha, _pair(16, 0)))
    cg2 = np.random.default_rng(13)
    S = _pair(cg2.integers(-50, 51, (6, 2)), cg2.integers(-50, 51, (6, 2)))
    Pp = _pair(cg2.integers(-50, 51, (6, 2)), cg2.integers(-50, 51, (6, 2)))
    cases.append(("cg_conj_inner", dict(b_conj=1, c_zero=1, reduce_rows=1),
                  S, Pp, _pair(np.zeros((6, 2)), np.zeros((6, 2))), _pair(16, 0), _pair(0, 0)))
    return cases


def _mem_words(mem, elem_cls, word_bw: int) -> list[int]:
    da = DataArray.specialize(elem_cls, max_shape=(len(mem),))(mem)
    return [int(w) for w in np.asarray(write_array(da, word_bw=word_bw)).ravel()]


def golden_case(cfg: StructCfg, case) -> dict:
    """Run the Python golden + oracle for one case; return the cmd / mem / expected vectors."""
    label, flags, a, b, c, alpha, beta = case
    accel = cfg.accel()
    cmd, mem = build(accel, flags, a, b, c, alpha, beta)
    in_elem = cfg.in_elem()
    cmd_words = [int(w) for w in np.asarray(cmd.serialize(word_bw=32)).ravel()]
    mem_in_words = _mem_words(mem, in_elem, cfg.mem_dwidth)

    post = mem.copy()
    dst = accel.execute(cmd, post)                       # mutates `post` (writes dst region)
    # independent-oracle cross-check on the dst bits (golden == spec)
    exp_re, exp_im = oracle(cfg, flags, a, b, c, alpha, beta)
    got_re, got_im = np.asarray(dst.val["re"]), np.asarray(dst.val["im"])
    oracle_ok = np.array_equal(got_re, exp_re) and np.array_equal(got_im, exp_im)

    mem_exp_words = _mem_words(post, in_elem, cfg.mem_dwidth)
    return {
        "label": label, "n_cols": int(cmd.n_cols), "oracle_ok": bool(oracle_ok),
        "cmd_words": cmd_words, "mem_in_words": mem_in_words, "mem_exp_words": mem_exp_words,
    }


# --- C++ testbench renderer (one batched TB per structural config) -------------
def render_tb(cfg: StructCfg) -> str:
    in_ns, out_ns = _array_utils_namespace(cfg.in_elem()), _array_utils_namespace(cfg.out_elem())
    in_hdr, out_hdr = _array_utils_filename(cfg.in_elem()), _array_utils_filename(cfg.out_elem())
    incs = [f'#include "{in_hdr}"']
    if out_hdr != in_hdr:
        incs.append(f'#include "{out_hdr}"')
    nl = "\n"
    lines = [
        "// Generated VMAC conformance testbench (Vitis C-sim).  One compiled TB per structural",
        "// config; it loops a manifest of (cmd, mem_in) vector files and writes mem_out for each,",
        "// exercising the vmac_compute_impl.tpp hook against the Python golden bit-for-bit.",
        "#include <ap_int.h>",
        "#include <ap_fixed.h>",
        "#include <complex>",
        "#include <fstream>",
        "#include <iostream>",
        "#include <string>",
        "#include <vector>",
        f'#include "{INCLUDE_DIR}/vmac_cmd_data_bw{cfg.data_bw}_mem_awidth32.h"',
        *incs,
        "",
        "namespace vmac_impl {",
        f"typedef VmacCmd_data_bw{cfg.data_bw}_mem_awidth32 VmacCmd;",
        f"namespace vmac_in_au  = ::{in_ns};",
        f"namespace vmac_out_au = ::{out_ns};",
        "}",
        '#include "vmac_compute_impl.tpp"',
        "",
        "static std::vector<unsigned long long> rw(const std::string& p) {",
        "    std::ifstream f(p.c_str()); std::vector<unsigned long long> v; unsigned long long x;",
        "    while (f >> x) v.push_back(x); return v;",
        "}",
        "",
        "static const int MEMCAP = 8192;",
        f"static ap_uint<{cfg.mem_dwidth}> mem[MEMCAP];",
        "",
        "int main(int argc, char** argv) {",
        "    std::ifstream man(argv[1]);",
        "    int ncmd, nmem; std::string cf, mf, of;",
        "    while (man >> ncmd >> nmem >> cf >> mf >> of) {",
        f"        ap_uint<32> cmdw[{CMD_NWORDS}];",
        "        std::vector<unsigned long long> cw = rw(cf);",
        "        std::vector<unsigned long long> mw = rw(mf);",
        "        if ((int)cw.size() < ncmd || (int)mw.size() < nmem) {",
        '            std::cerr << "VMAC_TB_ERROR: missing/short vector file " << cf'
        ' << " or " << mf << char(10); return 2; }',
        "        for (int i = 0; i < ncmd; ++i) cmdw[i] = (ap_uint<32>)cw[i];",
        "        vmac_impl::VmacCmd cmd; cmd.read_array<32>(cmdw);",
        f"        for (int i = 0; i < nmem; ++i) mem[i] = (ap_uint<{cfg.mem_dwidth}>)mw[i];",
        f"        vmac_impl::vmac_compute<{cfg.mem_dwidth}, {MEM_AWIDTH}, {cfg.data_bw}, "
        f"{cfg.int_bits}, {cfg.acc_bw}, {cfg.out_bw}, {cfg.q_rnd}, {cfg.o_sat}, "
        f"{MAX_COLS_CAP}>(cmd, mem);",
        "        std::ofstream o(of.c_str());",
        "        for (int i = 0; i < nmem; ++i) o << (unsigned long long)mem[i] << char(10);",
        "    }",
        "    return 0;",
        "}",
    ]
    return nl.join(lines) + nl


_RUN_TCL = """# Vitis HLS C-sim driver for one VMAC structural-config conformance testbench.
# argv = the manifest of per-case (cmd, mem_in, mem_out) vector files.
open_project -reset vmac_conf_proj
set_top main
add_files -tb kernel.cpp -cflags "-I. -Iinclude"

open_solution -reset "solution1"
set_part {xc7z020clg484-1}
create_clock -period 10

set d [file dirname [file normalize [info script]]]
if {[catch {csim_design -argv [file join $d manifest.txt]} res]} {
    puts "WAVEFLOW_ERROR: HLS C-Simulation failed."
    puts $res
    exit 1
}
puts "WAVEFLOW_SUCCESS: vmac conformance csim passed."
exit 0
"""


def gen_config_sources(cfg: StructCfg, cfg_dir: Path) -> list[dict]:
    """Generate one structural config's dir: shared headers, the batched TB, run.tcl, and the
    per-case cmd/mem/expected vectors + manifest.  Returns the per-case golden metadata."""
    cfg_dir = Path(cfg_dir).resolve()         # absolute, so the manifest paths resolve from
    cfg_dir.mkdir(parents=True, exist_ok=True)  # the deep Vitis csim build dir.
    bc = BuildConfig(root_dir=cfg_dir)
    dag = BuildDag()
    dag.add(StreamUtilsStep(output_dir=INCLUDE_DIR))
    # the command + its nested Region / Scalar schemas (the cmd header #includes them)
    for sch in (Region.specialize(mem_awidth=32),
                Scalar.specialize(mem_awidth=32, data_bw=cfg.data_bw),
                VmacCmd.specialize(mem_awidth=32, data_bw=cfg.data_bw)):
        dag.add(DataSchemaStep(sch, word_bw_supported=WORD_BW_SUPPORTED, include_dir=INCLUDE_DIR))
    dag.add(ArrayUtilsStep(cfg.in_elem(), MEM_WORD_BWS))
    if cfg.out_elem() is not cfg.in_elem():
        dag.add(ArrayUtilsStep(cfg.out_elem(), MEM_WORD_BWS))
    dag.run(bc)

    # the hand-written hook + the complex-arithmetic header travel with the generated headers
    for fname in HOOK_FILES:
        shutil.copy(_SOURCE_DIR / fname, cfg_dir / fname)
    for fname in BUILD_HDRS:
        shutil.copy(_BUILD_DIR / fname, cfg_dir / fname)
    (cfg_dir / "kernel.cpp").write_text(render_tb(cfg), encoding="utf-8")
    (cfg_dir / "run.tcl").write_text(_RUN_TCL, encoding="utf-8")

    cases_dir = cfg_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    manifest, meta = [], []
    for case in _runtime_cases():
        g = golden_case(cfg, case)
        label = g["label"]
        cf = cases_dir / f"{label}_cmd.txt"
        mf = cases_dir / f"{label}_mem.txt"
        of = cases_dir / f"{label}_out.txt"
        cf.write_text("\n".join(str(w) for w in g["cmd_words"]) + "\n", encoding="utf-8")
        mf.write_text("\n".join(str(w) for w in g["mem_in_words"]) + "\n", encoding="utf-8")
        manifest.append(f"{len(g['cmd_words'])} {len(g['mem_in_words'])} "
                        f"{cf.as_posix()} {mf.as_posix()} {of.as_posix()}")
        meta.append({"label": label, "oracle_ok": g["oracle_ok"],
                     "expected": g["mem_exp_words"], "out_file": of.as_posix(),
                     "nmem": len(g["mem_in_words"])})
    (cfg_dir / "manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    (cfg_dir / "expected.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


# --- BuildDag steps -----------------------------------------------------------
@dataclass(kw_only=True)
class GenStep(BuildStep):
    description = "Generate per-structural-config headers + batched TB + cmd/mem/expected vectors."
    consumes = ["vmac_source", "tpp_source"]
    produces = {"gen_dir": Path("gen")}

    def run(self, config: BuildConfig, **_) -> dict[str, Any]:
        gen = config.root_dir / "gen"
        for cfg in STRUCT_CFGS:
            gen_config_sources(cfg, gen / cfg.name)
        return {"gen_dir": gen}


@dataclass(kw_only=True)
class PySimStep(BuildStep):
    description = "Golden-vs-independent-oracle parity over every case (the non-Vitis anchor)."
    consumes = ["gen_dir"]
    produces = {"py_summary": Path("results/py_summary.json")}

    def run(self, config: BuildConfig, **_) -> dict[str, Any]:
        gen = config.root_dir / "gen"
        n_cases, n_ok = 0, 0
        for cfg in STRUCT_CFGS:
            meta = json.loads((gen / cfg.name / "expected.json").read_text(encoding="utf-8"))
            for m in meta:
                n_cases += 1
                n_ok += bool(m["oracle_ok"])
        out = config.root_dir / "results" / "py_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"n_cases": n_cases, "n_oracle_ok": n_ok}, indent=2),
                       encoding="utf-8")
        if n_ok != n_cases:
            raise RuntimeError(f"golden disagreed with the oracle on {n_cases - n_ok}/{n_cases} cases.")
        return {"py_summary": out}


def _csim_config(cfg_dir: Path, *, live_output: bool) -> list[dict]:
    """Run csim for one structural config and compare each case's mem_out bits to the golden."""
    meta = json.loads((cfg_dir / "expected.json").read_text(encoding="utf-8"))
    toolchain.run_vitis_hls(cfg_dir / "run.tcl", work_dir=cfg_dir, capture_output=not live_output)
    results = []
    for m in meta:
        got = [int(t) for t in Path(m["out_file"]).read_text(encoding="utf-8").split()]
        exp = m["expected"]
        mism = [{"i": i, "expected": e, "got": g}
                for i, (e, g) in enumerate(zip(exp, got)) if e != g]
        results.append({"label": m["label"], "n": len(exp),
                        "count_ok": len(got) == len(exp), "mismatches": mism[:5],
                        "exact": len(got) == len(exp) and not mism})
    return results


@dataclass(kw_only=True)
class CsimStep(BuildStep):
    description = "Vitis C-sim per structural config; assert mem_out bits == the golden, exactly."
    consumes = ["gen_dir"]
    produces = {"csim_report": Path("results/csim_report.json")}
    params: dict = field(default_factory=lambda: {"live_output": False})

    def run(self, config: BuildConfig, live_output, **_) -> dict[str, Any]:
        gen = config.root_dir / "gen"
        report = {}
        n_cases, n_exact = 0, 0
        for cfg in STRUCT_CFGS:
            res = _csim_config(gen / cfg.name, live_output=live_output)
            report[cfg.name] = res
            n_cases += len(res)
            n_exact += sum(r["exact"] for r in res)
        out = config.root_dir / "results" / "csim_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"n_cases": n_cases, "n_exact": n_exact, "all_exact": n_exact == n_cases,
             "by_config": report}, indent=2), encoding="utf-8")
        failed = [(cn, r) for cn, rs in report.items() for r in rs if not r["exact"]]
        if failed:
            cn, r = failed[0]
            raise RuntimeError(
                f"STOP — Vitis disagreed with the Python golden on {len(failed)}/{n_cases} cases. "
                f"The golden is the spec; fix the kernel, do not loosen the compare. "
                f"First failure: config={cn} case={r['label']} mismatches={r['mismatches']}")
        return {"csim_report": out}


# --- throughput sweep (mem_dwidth scaling) ------------------------------------
# One synthesized accelerator per mem_dwidth; PF = mem_dwidth / (2*data_bw) complex columns
# packed per memory word.  With the inner column loop unrolled to PF lanes, the per-row column
# iterations fall ~1/PF, so cosim transaction cycles should roughly halve as mem_dwidth doubles
# — the bus-width parallelism the wide packing is meant to deliver.  The cosim also re-checks
# the result against the golden, so it validates the PF > 1 lane packing (which the PF = 1
# csim conformance does not exercise).
TPUT_MEM_BWS = [16, 32, 64]                  # PF = 1, 2, 4  (element = 2*data_bw = 16 bits)
TPUT_N_ROWS = 8
TPUT_N_COLS = 64                             # multiple of every PF; row_stride = N_COLS
TPUT_MAX_COLS = 64                           # kernel acc[] capacity for the sweep


def _tput_cfg(mem_bw: int) -> StructCfg:
    return StructCfg(out_bw=8, q_rnd=0, o_sat=0, mem_dwidth=mem_bw)


def _tput_vectors(cfg: StructCfg):
    """The fixed throughput problem: a Hadamard product D = A·B (c_zero, alpha = 1.0) over an
    N_ROWS x N_COLS image — a per-element op whose inner column loop is the unroll target.

    Returns ``(scalar_args, mem_in_words, mem_exp_words)`` — the cmd is reduced to the scalar
    fields the top rebuilds it from (so it crosses into RTL as plain s_axilite registers, not a
    nested struct that cosim mis-marshals)."""
    rng = np.random.default_rng(7)
    n, m = TPUT_N_ROWS, TPUT_N_COLS
    a = _pair(rng.integers(-30, 31, (n, m)), rng.integers(-30, 31, (n, m)))
    b = _pair(rng.integers(-30, 31, (n, m)), rng.integers(-30, 31, (n, m)))
    c = _pair(np.zeros((n, m)), np.zeros((n, m)))
    cmd, mem = build(cfg.accel(), dict(c_zero=1), a, b, c, _pair(16, 0), _pair(0, 0))
    post = mem.copy()
    cfg.accel().execute(cmd, post)
    in_elem = cfg.in_elem()
    flags = (int(cmd.b_one) | (int(cmd.c_zero) << 1)
             | (int(cmd.b_conj) << 2) | (int(cmd.reduce_rows) << 3))
    scalars = [int(cmd.n_rows), int(cmd.n_cols),
               int(cmd.a.addr), int(cmd.a.row_stride), int(cmd.b.addr), int(cmd.b.row_stride),
               int(cmd.c.addr), int(cmd.c.row_stride), int(cmd.d.addr), int(cmd.d.row_stride),
               int(cmd.alpha.re), int(cmd.alpha.im), int(cmd.beta.re), int(cmd.beta.im), flags]
    return (scalars,
            _mem_words(mem, in_elem, cfg.mem_dwidth),
            _mem_words(post, in_elem, cfg.mem_dwidth))


_TOP_ARGS = ("int n_rows, int n_cols, int a_addr, int a_rs, int b_addr, int b_rs, "
             "int c_addr, int c_rs, int d_addr, int d_rs, int al_re, int al_im, "
             "int be_re, int be_im, unsigned flags")


def _vmac_hpp(cfg: StructCfg) -> str:
    in_ns = _array_utils_namespace(cfg.in_elem())
    in_hdr = _array_utils_filename(cfg.in_elem())
    return "\n".join([
        "#ifndef VMAC_HPP", "#define VMAC_HPP",
        "#include <ap_int.h>", "#include <ap_fixed.h>", "#include <complex>",
        f'#include "{INCLUDE_DIR}/vmac_cmd_data_bw{cfg.data_bw}_mem_awidth32.h"',
        f'#include "{in_hdr}"',
        "namespace vmac_impl {",
        f"typedef VmacCmd_data_bw{cfg.data_bw}_mem_awidth32 VmacCmd;",
        f"namespace vmac_in_au  = ::{in_ns};",
        f"namespace vmac_out_au = ::{in_ns};",
        "}",
        '#include "vmac_compute_impl.tpp"',
        "#endif", "",
    ])


def render_top(cfg: StructCfg, depth: int) -> str:
    mbw = cfg.mem_dwidth
    scalar_ports = ("n_rows n_cols a_addr a_rs b_addr b_rs c_addr c_rs d_addr d_rs "
                    "al_re al_im be_re be_im flags").split()
    lines = [
        "// Synthesizable VMAC top — m_axi shared memory + scalar s_axilite command fields —",
        "// wrapping the vmac_compute hook at one structural mem_dwidth (PF = mem_dwidth/(2*data_bw)).",
        "// The command is rebuilt from plain scalar registers (a nested struct mis-marshals over",
        "// cosim's s_axilite adapter), then handed to the same .tpp datapath the csim conformance runs.",
        '#include "vmac.hpp"',
        "",
        f"void vmac(ap_uint<{mbw}>* gmem, {_TOP_ARGS}) {{",
        f"#pragma HLS INTERFACE m_axi port=gmem offset=slave bundle=gmem depth={depth}",
    ]
    lines += [f"#pragma HLS INTERFACE s_axilite port={p} bundle=control" for p in scalar_ports]
    lines += [
        "#pragma HLS INTERFACE s_axilite port=return bundle=control",
        "    // Call the scalar-arg core directly (no VmacCmd struct — it mis-decomposes at csynth).",
        f"    vmac_impl::vmac_compute_core<{mbw}, {MEM_AWIDTH}, {cfg.data_bw}, {cfg.int_bits}, "
        f"{cfg.acc_bw}, {cfg.out_bw}, {cfg.q_rnd}, {cfg.o_sat}, {TPUT_MAX_COLS}>(",
        "        gmem, n_rows, n_cols, flags & 1, (flags >> 1) & 1, (flags >> 2) & 1,",
        "        (flags >> 3) & 1, a_addr, a_rs, b_addr, b_rs, c_addr, c_rs, d_addr, d_rs,",
        "        true, al_re, al_im, 0, 0, true, be_re, be_im, 0, 0);",
        "}",
        "",
    ]
    return "\n".join(lines)


def render_cosim_tb(cfg: StructCfg, scalars: list[int], nmem: int, depth: int) -> str:
    mbw = cfg.mem_dwidth
    call_args = ", ".join(str(v) for v in scalars)
    return "\n".join([
        "// Cosim testbench: load the mem image, drive the vmac top with the (baked) command",
        "// fields, and re-check the result against the golden — so cosim validates the PF>1 lane",
        "// packing as well as the timing.",
        '#include "vmac.hpp"',
        "#include <fstream>", "#include <iostream>", "#include <string>", "#include <vector>",
        "",
        f"void vmac(ap_uint<{mbw}>* gmem, {_TOP_ARGS});",
        "",
        "static std::vector<unsigned long long> rw(const std::string& p) {",
        "    std::ifstream f(p.c_str()); std::vector<unsigned long long> v; unsigned long long x;",
        "    while (f >> x) v.push_back(x); return v;",
        "}",
        "",
        f"static ap_uint<{mbw}> mem[{depth}];",
        "",
        "int main(int argc, char** argv) {",
        "    std::string d = argv[1];",
        '    std::vector<unsigned long long> mw = rw(d + "/mem_in.txt");',
        '    std::vector<unsigned long long> ew = rw(d + "/mem_exp.txt");',
        f"    if ((int)mw.size() < {nmem} || (int)ew.size() < {nmem}) {{",
        '        std::cerr << "VMAC_TB_ERROR: short vector files" << char(10); return 2; }',
        f"    for (int i = 0; i < {nmem}; ++i) mem[i] = (ap_uint<{mbw}>)mw[i];",
        f"    vmac(mem, {call_args});",
        "    int bad = 0;",
        f"    for (int i = 0; i < {nmem}; ++i) if ((unsigned long long)mem[i] != ew[i]) ++bad;",
        '    if (bad) { std::cerr << "VMAC_COSIM_MISMATCH " << bad << char(10); return 1; }',
        '    std::cout << "VMAC_COSIM_OK" << char(10);',
        "    return 0;",
        "}",
        "",
    ])


_TPUT_TCL = """# Vitis HLS csim -> csynth -> cosim for one VMAC mem_dwidth (throughput point).
set d [file dirname [file normalize [info script]]]
set data_dir [file join $d data]
open_project -reset vmac_tput_proj
set_top vmac
add_files vmac_top.cpp -cflags "-I. -Iinclude"
add_files -tb vmac_tb.cpp -cflags "-I. -Iinclude"
set su [file join $d include streamutils.cpp]
if {[file exists $su]} { add_files -tb $su -cflags "-I. -Iinclude" }
open_solution -reset "solution1"
set_part {xc7z020clg484-1}
create_clock -period 10
foreach {stage cmd} {csim csim_design csynth csynth_design cosim cosim_design} {
    if {$stage eq "csim"}  { set rc [catch {csim_design -argv $data_dir} res] }
    if {$stage eq "csynth"} { set rc [catch {csynth_design} res] }
    if {$stage eq "cosim"} { set rc [catch {cosim_design -argv $data_dir -trace_level none} res] }
    if {$rc} { puts "WAVEFLOW_ERROR: vmac $stage failed."; puts $res; exit 1 }
}
puts "WAVEFLOW_SUCCESS: vmac tput csim/csynth/cosim passed."
exit 0
"""


def gen_tput_config(cfg: StructCfg, tdir: Path) -> dict:
    tdir = Path(tdir).resolve()
    tdir.mkdir(parents=True, exist_ok=True)
    bc = BuildConfig(root_dir=tdir)
    dag = BuildDag()
    dag.add(StreamUtilsStep(output_dir=INCLUDE_DIR))
    for sch in (Region.specialize(mem_awidth=32),
                Scalar.specialize(mem_awidth=32, data_bw=cfg.data_bw),
                VmacCmd.specialize(mem_awidth=32, data_bw=cfg.data_bw)):
        dag.add(DataSchemaStep(sch, word_bw_supported=WORD_BW_SUPPORTED, include_dir=INCLUDE_DIR))
    dag.add(ArrayUtilsStep(cfg.in_elem(), MEM_WORD_BWS))
    dag.run(bc)
    for fname in HOOK_FILES:
        shutil.copy(_SOURCE_DIR / fname, tdir / fname)
    for fname in BUILD_HDRS:
        shutil.copy(_BUILD_DIR / fname, tdir / fname)

    scalars, mem_in_w, mem_exp_w = _tput_vectors(cfg)
    nmem = len(mem_in_w)
    depth = nmem
    (tdir / "vmac.hpp").write_text(_vmac_hpp(cfg), encoding="utf-8")
    (tdir / "vmac_top.cpp").write_text(render_top(cfg, depth), encoding="utf-8")
    (tdir / "vmac_tb.cpp").write_text(render_cosim_tb(cfg, scalars, nmem, depth), encoding="utf-8")
    (tdir / "run.tcl").write_text(_TPUT_TCL, encoding="utf-8")
    data = tdir / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "mem_in.txt").write_text("\n".join(str(w) for w in mem_in_w) + "\n", encoding="utf-8")
    (data / "mem_exp.txt").write_text("\n".join(str(w) for w in mem_exp_w) + "\n", encoding="utf-8")
    return {"mem_dwidth": cfg.mem_dwidth, "pf": cfg.mem_dwidth // (2 * cfg.data_bw), "nmem": nmem}


@dataclass(kw_only=True)
class GenTputStep(BuildStep):
    description = "Generate the per-mem_dwidth synthesizable top + cosim TB + vectors (PF sweep)."
    consumes = ["vmac_source", "tpp_source"]
    produces = {"tput_dir": Path("tput")}

    def run(self, config: BuildConfig, **_) -> dict[str, Any]:
        tput = config.root_dir / "tput"
        for mbw in TPUT_MEM_BWS:
            gen_tput_config(_tput_cfg(mbw), tput / f"m{mbw}")
        return {"tput_dir": tput}


@dataclass(kw_only=True)
class CosimStep(BuildStep):
    description = "Per mem_dwidth: Vitis csim/csynth/cosim (cosim re-checks the golden)."
    consumes = ["tput_dir"]
    produces = {"cosim_done": Path("results/cosim_done.json")}
    params: dict = field(default_factory=lambda: {"live_output": False})

    def run(self, config: BuildConfig, live_output, **_) -> dict[str, Any]:
        tput = config.root_dir / "tput"
        done = []
        for mbw in TPUT_MEM_BWS:
            d = (tput / f"m{mbw}").resolve()
            toolchain.run_vitis_hls(d / "run.tcl", work_dir=d, capture_output=not live_output)
            done.append({"mem_dwidth": mbw})
        out = config.root_dir / "results" / "cosim_done.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(done, indent=2), encoding="utf-8")
        return {"cosim_done": out}


@dataclass(kw_only=True)
class ExtractCosimTimingStep(BuildStep):
    description = "Parse cosim transaction cycles per mem_dwidth; report throughput vs PF."
    consumes = ["cosim_done"]
    produces = {"cosim_timing": Path("results/cosim_timing.json")}

    def run(self, config: BuildConfig, **_) -> dict[str, Any]:
        from waveflow.utils.cosimparse import CosimReportParser
        tput = config.root_dir / "tput"
        rows = []
        for mbw in TPUT_MEM_BWS:
            sol = tput / f"m{mbw}" / "vmac_tput_proj" / "solution1"
            cycles = CosimReportParser(sol_path=sol, top="vmac").get_transaction_cycles()
            rows.append({"mem_dwidth": mbw, "pf": mbw // (2 * StructCfg(out_bw=8, q_rnd=0,
                         o_sat=0).data_bw), "transaction_cycles": cycles})
        base = next((r["transaction_cycles"] for r in rows if r["pf"] == 1), None)
        for r in rows:
            r["speedup_vs_pf1"] = (base / r["transaction_cycles"]
                                   if base and r["transaction_cycles"] else None)
        out = config.root_dir / "results" / "cosim_timing.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"n_rows": TPUT_N_ROWS, "n_cols": TPUT_N_COLS, "points": rows},
                                  indent=2), encoding="utf-8")
        return {"cosim_timing": out}


def build_vmac_dag() -> BuildDag:
    dag = BuildDag()
    dag.add(SourceStep(artifact="vmac_source", path=_SOURCE_DIR / "vmac.py"))
    dag.add(SourceStep(artifact="tpp_source", path=_SOURCE_DIR / "vmac_compute_impl.tpp"))
    dag.add(GenStep(name="gen"))
    dag.add(PySimStep(name="py_sim"))
    dag.add(CsimStep(name="csim"))
    dag.add(GenTputStep(name="gen_tput"))
    dag.add(CosimStep(name="cosim"))
    dag.add(ExtractCosimTimingStep(name="extract_cosim_timing"))
    return dag


def main() -> None:
    run_dag_cli(
        build_vmac_dag,
        description="VMAC Python-vs-Vitis bit-exact conformance (generated ComplexField serialization).",
        default_through="py_sim",
        root_dir=_SOURCE_DIR,
        extra_args=[(("--live-output",), {"action": "store_true"})],
        params_from_args=lambda a: {"live_output": a.live_output},
    )


if __name__ == "__main__":
    main()
