"""VMAC conformance harness — the Python golden (``VmacAccel.execute``) vs the Vitis HLS
m_axi kernel, bit-exact.

For each curated case (a CG / general ``VmacCmd`` config × a numeric param set × real |
complex), this lays the operands into a shared-memory image, runs ``VmacAccel.execute`` to
get the golden dst bits, renders the parameterized m_axi kernel (``ap_fixed`` accumulator =
``VmacAccel.accumulator_format``, output = ``VmacAccel.output_format``), runs it in Vitis
C-sim over the same memory image, and asserts the emitted dst bits equal the golden bits —
**bit-for-bit, zero mismatch**.  If they ever differ the Python golden is the spec; fix the
kernel, never loosen the comparison.  Built on the shared ``BuildDag`` + ``run_dag_cli`` rig
(same shape as ``examples/schemas/{fixedpoint,complex}``).

CLI::

    python vmac_build.py --through gen_conformance    # render kernels + mem images (no Vitis)
    python vmac_build.py --through run_conformance     # the full csim conformance (Vitis)
    python vmac_build.py --list-steps
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from examples.vmac.golden import VmacAccel
from examples.vmac.kernels import KernelSpec, render
from examples.vmac.vmac_cmd import VmacMode
from waveflow.build.build import BuildConfig, BuildDag, BuildStep, SourceStep
from waveflow.build.cli import run_dag_cli
from waveflow.toolchain import toolchain
from waveflow.utils import complexutils as cx
from waveflow.utils.fixputils import Format, to_bits

_SOURCE_DIR = Path(__file__).resolve().parent


# --- numeric param sets + configs ---------------------------------------------
@dataclass(frozen=True)
class Params:
    name: str
    data_bw: int
    int_bits: int
    acc_bw: int
    out_bw: int
    shift: int
    q_rnd: int
    o_sat: int


# trn/wrap at one width; rnd/sat at a wider one — exercise both rounding + saturation edges
P_TRN = Params("trn", data_bw=8, int_bits=4, acc_bw=64, out_bw=8, shift=8, q_rnd=0, o_sat=0)
P_RND = Params("rnd", data_bw=12, int_bits=6, acc_bw=64, out_bw=10, shift=12, q_rnd=1, o_sat=1)


@dataclass(frozen=True)
class Config:
    name: str
    b_one: int = 0
    c_zero: int = 0
    b_conj: int = 0
    reduce_rows: int = 0
    alpha_percol: bool = False
    beta_one: bool = False          # beta = 1.0 immediate (CG axpy uses beta = 1)
    complex_only: bool = False


CONFIGS = [
    Config("scaled_copy", b_one=1, c_zero=1),                       # dst = alpha*A
    Config("hadamard", c_zero=1),                                   # dst = alpha*A*B
    Config("full_mac"),                                             # dst = alpha*A*B + beta*C
    Config("axpy_percol", b_one=1, alpha_percol=True, beta_one=True),  # CG: X - P*alpha[col]
    Config("colsum", b_one=1, c_zero=1, reduce_rows=1),             # dst = sum_rows(alpha*A)
    Config("reduced_mac", reduce_rows=1),                           # sum_rows(alpha*A*B + beta*C)
    Config("conj_inner", b_conj=1, c_zero=1, reduce_rows=1, complex_only=True),  # CG: sum conj(P)*S
]


# --- operand generation (rounding / saturation-triggering, fixed seed) --------
def _rng(name: str) -> np.random.Generator:
    return np.random.default_rng(abs(hash(name)) % (2 ** 32))


def _vals(rng, data_bw, shape):
    hi = (1 << (data_bw - 1)) - 1
    v = rng.integers(-hi, hi + 1, shape, dtype=np.int64)
    # plant a few near-extreme magnitudes to push the requantize into saturation
    if v.size >= 2:
        flat = v.ravel()
        flat[0] = hi
        flat[1] = -hi - 1
    return v


# --- one case: lay mem, run golden, render kernel, collect bits ---------------
def _to_bits_list(arr, w):
    return [int(x) for x in np.atleast_1d(to_bits(np.asarray(arr).ravel(), w))]


def _interleave(re, im):
    return [int(x) for pair in zip(np.atleast_1d(re), np.atleast_1d(im)) for x in pair]


def make_case(cfg: Config, params: Params, mode: str, n: int = 4, m: int = 3) -> dict:
    complex_mode = mode == "complex"
    p = params
    accel = VmacAccel(data_bw=p.data_bw, mem_awidth=32, acc_bw=p.acc_bw, out_bw=p.out_bw)
    rng = _rng(f"{cfg.name}_{params.name}_{mode}")
    fmt = Format(p.data_bw, p.int_bits, True)
    nm = n * m

    def operand():
        if complex_mode:
            return (_vals(rng, p.data_bw, (n, m)), _vals(rng, p.data_bw, (n, m)))
        return _vals(rng, p.data_bw, (n, m))

    a, b, c = operand(), operand(), operand()

    # memory layout: a | b | c | [alpha_pc] | [beta_pc] | dst
    blocks_re, blocks_im, addr, cur = [], [], {}, 0
    for name, op in (("a", a), ("b", b), ("c", c)):
        addr[name] = cur
        blocks_re.append((op[0] if complex_mode else op).ravel())
        blocks_im.append((op[1] if complex_mode else np.zeros(nm)).ravel())
        cur += nm

    hi = (1 << (p.data_bw - 1)) - 1
    if cfg.alpha_percol:
        addr["alpha"] = cur
        a_re = rng.integers(-hi, hi + 1, m, dtype=np.int64)
        a_im = rng.integers(-hi, hi + 1, m, dtype=np.int64) if complex_mode else np.zeros(m, np.int64)
        blocks_re.append(a_re)
        blocks_im.append(a_im)
        cur += m
    addr["d"] = cur
    cur += nm

    re = np.zeros(cur, dtype=np.int64)
    im = np.zeros(cur, dtype=np.int64)
    order = ["a", "b", "c"] + (["alpha"] if cfg.alpha_percol else [])
    for name, bre, bim in zip(order, blocks_re, blocks_im):
        re[addr[name]:addr[name] + len(bre)] = bre
        im[addr[name]:addr[name] + len(bim)] = bim
    mem = cx.make_complex(re, im, fmt) if complex_mode else re

    # alpha / beta immediates (data_bw-range)
    alpha_re_imm = int(rng.integers(-hi, hi + 1))
    alpha_im_imm = int(rng.integers(-hi, hi + 1)) if complex_mode else 0
    beta_re_imm = (1 << p.int_bits) if cfg.beta_one else int(rng.integers(-hi, hi + 1))
    beta_im_imm = 0 if cfg.beta_one else (int(rng.integers(-hi, hi + 1)) if complex_mode else 0)

    cmd = accel.Cmd()
    cmd.n_rows, cmd.n_cols = n, m
    for name in ("a", "b", "c", "d"):
        setattr(cmd, name, {"addr": addr[name], "row_stride": m, "col_stride": 1})
    if cfg.alpha_percol:
        cmd.alpha = {"direct": 0, "re": 0, "im": 0, "addr": addr["alpha"], "stride": 1}
    else:
        cmd.alpha = {"direct": 1, "re": alpha_re_imm, "im": alpha_im_imm, "addr": 0, "stride": 0}
    cmd.beta = {"direct": 1, "re": beta_re_imm, "im": beta_im_imm, "addr": 0, "stride": 0}
    cmd.b_one, cmd.c_zero, cmd.b_conj, cmd.reduce_rows = cfg.b_one, cfg.c_zero, cfg.b_conj, cfg.reduce_rows
    cmd.mode = VmacMode.COMPLEX if complex_mode else VmacMode.REAL
    cmd.int_bits, cmd.shift, cmd.q_rnd, cmd.o_sat = p.int_bits, p.shift, p.q_rnd, p.o_sat

    # golden + expected bits (dst, row-major; interleaved re/im for complex)
    dst = accel.execute(cmd, mem.copy())
    if complex_mode:
        expected = _interleave(to_bits(np.asarray(dst.val["re"]).ravel(), p.out_bw),
                               to_bits(np.asarray(dst.val["im"]).ravel(), p.out_bw))
    else:
        expected = _to_bits_list(dst.val, p.out_bw)

    # kernel spec (ap_fixed types straight from the golden's format methods)
    acc = accel.accumulator_format(cmd)
    out_cls = accel.output_format(cmd)

    def _reg(r):
        return (int(r.addr), int(r.row_stride), int(r.col_stride))

    def _imm(v):
        return int(to_bits(np.int64(v), p.data_bw))

    alpha_spec = (("indirect", addr["alpha"], 1) if cfg.alpha_percol
                  else ("direct", _imm(alpha_re_imm), _imm(alpha_im_imm)))
    spec = KernelSpec(
        mode=mode, data_bw=p.data_bw, int_bits=p.int_bits,
        a_t=f"ap_fixed<{p.data_bw}, {p.int_bits}, AP_TRN, AP_WRAP>",
        acc_t=f"ap_fixed<{acc.W}, {acc.int_bits}, AP_TRN, AP_WRAP>",
        out_t=out_cls.cpp_type, out_bw=out_cls.get_bitwidth(),
        n_rows=n, n_cols=m,
        a=_reg(cmd.a), b=_reg(cmd.b), c=_reg(cmd.c), d=_reg(cmd.d),
        b_one=bool(cfg.b_one), c_zero=bool(cfg.c_zero), b_conj=bool(cfg.b_conj),
        reduce_rows=bool(cfg.reduce_rows),
        alpha=alpha_spec, beta=("direct", _imm(beta_re_imm), _imm(beta_im_imm)),
    )
    kernel = render(spec)

    # mem image (in_a.txt): real stored ints; complex interleaved re/im
    if complex_mode:
        mem_bits = _interleave(to_bits(np.asarray(mem["re"]), p.data_bw),
                               to_bits(np.asarray(mem["im"]), p.data_bw))
    else:
        mem_bits = _to_bits_list(mem, p.data_bw)

    return {
        "name": f"{cfg.name}_{params.name}_{mode}",
        "kernel": kernel,
        "in_a": "\n".join(map(str, mem_bits)) + "\n",
        "in_b": "0\n",
        "expected": expected,
    }


def build_cases() -> list[dict]:
    cases: list[dict] = []
    for cfg in CONFIGS:
        modes = ["complex"] if cfg.complex_only else ["real", "complex"]
        for mode in modes:
            for params in (P_TRN, P_RND):
                cases.append(make_case(cfg, params, mode))
    return cases


# --- gen / csim / compare (shared rig shape) ----------------------------------
def gen_case_sources(case: dict, work_dir: Path) -> Path:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "kernel.cpp").write_text(case["kernel"], encoding="utf-8")
    (work_dir / "in_a.txt").write_text(case["in_a"], encoding="utf-8")
    (work_dir / "in_b.txt").write_text(case["in_b"], encoding="utf-8")
    (work_dir / "expected.json").write_text(
        json.dumps({"name": case["name"], "expected": case["expected"]}, indent=2),
        encoding="utf-8")
    shutil.copy(_SOURCE_DIR / "run.tcl", work_dir / "run.tcl")
    return work_dir


def csim_and_compare(work_dir: Path, *, live_output: bool = False) -> dict:
    work_dir = Path(work_dir)
    expected = json.loads((work_dir / "expected.json").read_text(encoding="utf-8"))
    toolchain.run_vitis_hls(work_dir / "run.tcl", work_dir=work_dir, capture_output=not live_output)
    vitis = [int(tok) for tok in (work_dir / "out_bits.txt").read_text(encoding="utf-8").split()]
    exp = expected["expected"]
    mism = [{"i": i, "expected": e, "vitis": g}
            for i, (e, g) in enumerate(zip(exp, vitis)) if e != g]
    return {"name": expected["name"], "n": len(exp),
            "count_ok": len(vitis) == len(exp), "mismatches": mism,
            "exact": len(vitis) == len(exp) and not mism}


def conformance_for_case(case: dict, work_dir: Path, *, live_output: bool = False) -> dict:
    gen_case_sources(case, work_dir)
    return csim_and_compare(work_dir, live_output=live_output)


# --- BuildDag steps -----------------------------------------------------------
@dataclass(kw_only=True)
class GenConformanceStep(BuildStep):
    description = "Render the per-case VMAC m_axi kernels + shared-memory images + golden bits."
    consumes = ["vmac_source", "run_tcl", "kernels_source", "golden_source"]
    produces = {"conformance_gen": Path("gen")}
    params: dict = field(default_factory=dict)

    def run(self, config: BuildConfig, **_) -> dict:
        gen = config.root_dir / "gen"
        gen.mkdir(parents=True, exist_ok=True)
        for case in build_cases():
            gen_case_sources(case, gen / case["name"])
        return {"conformance_gen": gen}


@dataclass(kw_only=True)
class RunConformanceStep(BuildStep):
    description = "Per case: Vitis csim, assert kernel bits == VmacAccel.execute bits exactly."
    consumes = ["conformance_gen"]
    produces = {"conformance_report": Path("results/conformance_report.json")}
    params: dict = field(default_factory=lambda: {"live_output": False})

    def run(self, config: BuildConfig, live_output, **_) -> dict:
        gen = config.root_dir / "gen"
        results = [csim_and_compare(gen / case["name"], live_output=live_output)
                   for case in build_cases()]
        report_path = config.root_dir / "results" / "conformance_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        n_exact = sum(r["exact"] for r in results)
        report_path.write_text(json.dumps(
            {"n_cases": len(results), "n_exact": n_exact,
             "all_exact": n_exact == len(results), "results": results}, indent=2),
            encoding="utf-8")
        failed = [r for r in results if not r["exact"]]
        if failed:
            raise RuntimeError(
                f"STOP — Vitis disagreed with the Python golden on {len(failed)}/{len(results)} "
                "cases. The golden is the spec; fix the kernel, do not loosen the comparison. "
                f"First failure: {failed[0]}")
        return {"conformance_report": report_path}


def build_vmac_dag() -> BuildDag:
    dag = BuildDag()
    dag.add(SourceStep(artifact="vmac_source", path=_SOURCE_DIR / "vmac_build.py"))
    dag.add(SourceStep(artifact="kernels_source", path=_SOURCE_DIR / "kernels.py"))
    dag.add(SourceStep(artifact="golden_source", path=_SOURCE_DIR / "golden.py"))
    dag.add(SourceStep(artifact="run_tcl", path=_SOURCE_DIR / "run.tcl"))
    dag.add(GenConformanceStep(name="gen_conformance"))
    dag.add(RunConformanceStep(name="run_conformance"))
    return dag


def main() -> None:
    run_dag_cli(
        build_vmac_dag,
        description="VMAC Python-golden-vs-Vitis bit-exact conformance (real + complex).",
        default_through="gen_conformance",
        root_dir=_SOURCE_DIR,
        extra_args=[(("--live-output",), {"action": "store_true"})],
        params_from_args=lambda a: {"live_output": a.live_output},
    )


if __name__ == "__main__":
    main()
