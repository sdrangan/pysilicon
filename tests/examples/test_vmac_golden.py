"""Phase-1 VMAC golden tests — bit-level results for every config vs an independent oracle.

The production golden (:func:`examples.vmac.golden.execute`) composes the integer-backed
numpy ``FixedField`` / ``ComplexField`` operators.  The **oracle** here is independent: it
works in exact rational (``Fraction``) space over ``(re, im)`` pairs, computes the
accumulator value exactly (full precision), then quantizes once with the right-shift
``SHIFT`` → round → saturate spec.  Two independent implementations of the same datapath
must agree bit-for-bit.  A few literal hand-checked outputs anchor the oracle.

Operands are supplied as **stored integers** at fractional scale ``F = in_bw - int_bits``;
real value = ``stored · 2**-F`` (exact).  Real mode carries ``im = 0`` throughout, so one
oracle covers both modes.
"""
import math
from fractions import Fraction

import numpy as np
import pytest

from examples.vmac.golden import VmacAccel
from examples.vmac.vmac_cmd import VmacMode
from waveflow.utils import complexutils as cx
from waveflow.utils.fixputils import Format


# --- exact (re, im) Fraction complex arithmetic -------------------------------
def _cmul(a, b):
    (ar, ai), (br, bi) = a, b
    return (ar * br - ai * bi, ar * bi + ai * br)


def _cadd(a, b):
    return (a[0] + b[0], a[1] + b[1])


def _conj(a):
    return (a[0], -a[1])


def _q_real(value: Fraction, out_frac: int, W: int, q_rnd: bool, o_sat: bool) -> int:
    """Quantize an exact real ``value`` to ``W`` bits at ``out_frac`` frac bits, signed."""
    scaled = value * (Fraction(2) ** out_frac)
    f = math.floor(scaled + Fraction(1, 2)) if q_rnd else math.floor(scaled)
    if o_sat:
        return max(-(1 << (W - 1)), min((1 << (W - 1)) - 1, f))
    y = f & ((1 << W) - 1)
    return y - (1 << W) if (y >> (W - 1)) & 1 else y


# --- the oracle (independent of the production golden) ------------------------
def oracle(cfg, a, b, c, alpha, beta):
    """a/b/c: (n, m) stored-int pairs (re, im arrays); alpha/beta: (re, im) scalars or
    per-column (m,) arrays.  Returns the expected dst stored ints — (n,m) or (m,) reduced —
    as a pair (re, im) of int arrays (im all-zero for real mode)."""
    F = cfg["in_bw"] - cfg["int_bits"]
    scale = Fraction(2) ** F
    n, m = a[0].shape

    def fr(stored):
        return Fraction(int(stored), 1) / scale

    def at(operand, i, j):
        return (fr(operand[0][i, j]), fr(operand[1][i, j]))

    def scal(s, j):
        if np.ndim(s[0]) == 0:
            return (fr(s[0]), fr(s[1]))
        return (fr(s[0][j]), fr(s[1][j]))

    F_acc = (2 if cfg["b_one"] else 3) * F
    out_frac = F_acc - cfg["shift"]

    cols_re, cols_im = [], []
    for j in range(m):
        terms = []
        for i in range(n):
            av = at(a, i, j)
            if cfg["b_one"]:
                ab = av
            else:
                bv = at(b, i, j)
                if cfg["b_conj"]:
                    bv = _conj(bv)
                ab = _cmul(av, bv)
            t = _cmul(scal(alpha, j), ab)
            if not cfg["c_zero"]:
                t = _cadd(t, _cmul(scal(beta, j), at(c, i, j)))
            terms.append(t)
        acc = terms[0]
        if cfg["reduce_rows"]:
            for t in terms[1:]:
                acc = _cadd(acc, t)
            rows = [acc]
        else:
            rows = terms
        col_re = [_q_real(r[0], out_frac, cfg["out_bw"], cfg["q_rnd"], cfg["o_sat"]) for r in rows]
        col_im = [_q_real(r[1], out_frac, cfg["out_bw"], cfg["q_rnd"], cfg["o_sat"]) for r in rows]
        cols_re.append(col_re)
        cols_im.append(col_im)
    re = np.array(cols_re, dtype=np.int64).T          # (rows, m)
    im = np.array(cols_im, dtype=np.int64).T
    if cfg["reduce_rows"]:
        re, im = re[0], im[0]                          # (m,)
    return re, im


# --- harness: lay operands into mem + build the matching VmacCmd --------------
def _flat(pair_or_arr, complex_mode):
    """Flatten an operand to mem slots (row-major): structured for complex, int for real."""
    re = np.asarray(pair_or_arr[0]).ravel()
    if complex_mode:
        im = np.asarray(pair_or_arr[1]).ravel()
        return cx.make_complex(re, im, Format(8, 4, True))   # dtype only; format irrelevant here
    return re.astype(np.int64)


def _accel(cfg):
    """The accelerator carrying the structural widths for this config."""
    return VmacAccel(
        mem_dwidth=512, mem_awidth=32,
        data_bw=cfg["in_bw"], acc_bw=cfg.get("acc_bw", 48), out_bw=cfg["out_bw"],
    )


def build(accel, cfg, a, b, c, alpha, beta):
    """Lay a/b/c (+ per-column alpha/beta) into mem and build the Cmd. Returns (cmd, mem)."""
    complex_mode = cfg["mode"] == VmacMode.COMPLEX
    n, m = a[0].shape
    nm = n * m
    blocks, addr = [], {}
    cur = 0
    for name, op in (("a", a), ("b", b), ("c", c)):
        addr[name] = cur
        blocks.append(_flat(op, complex_mode))
        cur += nm
    alpha_pc = np.ndim(alpha[0]) > 0
    beta_pc = np.ndim(beta[0]) > 0
    if alpha_pc:
        addr["alpha"] = cur
        blocks.append(_flat((alpha[0], alpha[1] if complex_mode else np.zeros(m)), complex_mode))
        cur += m
    if beta_pc:
        addr["beta"] = cur
        blocks.append(_flat((beta[0], beta[1] if complex_mode else np.zeros(m)), complex_mode))
        cur += m
    addr["d"] = cur
    cur += nm
    if complex_mode:
        mem = cx.make_complex(np.zeros(cur), np.zeros(cur), Format(8, 4, True))
    else:
        mem = np.zeros(cur, dtype=np.int64)
    # write each block at its address (row-major)
    order = ["a", "b", "c"] + (["alpha"] if alpha_pc else []) + (["beta"] if beta_pc else [])
    for name, blk in zip(order, blocks):
        mem[addr[name]:addr[name] + len(blk)] = blk

    cmd = accel.Cmd()
    cmd.n_rows, cmd.n_cols = n, m
    cmd.a = {"addr": addr["a"], "row_stride": m, "col_stride": 1}
    cmd.b = {"addr": addr["b"], "row_stride": m, "col_stride": 1}
    cmd.c = {"addr": addr["c"], "row_stride": m, "col_stride": 1}
    cmd.d = {"addr": addr["d"], "row_stride": m, "col_stride": 1}
    cmd.alpha = _scalar_field(alpha, addr.get("alpha"), complex_mode)
    cmd.beta = _scalar_field(beta, addr.get("beta"), complex_mode)
    # in_bw / out_bw / acc_bw are now structural (on the accelerator), not cmd fields
    for f in ("b_one", "c_zero", "b_conj", "reduce_rows", "mode", "int_bits", "shift",
              "q_rnd", "o_sat"):
        setattr(cmd, f, cfg[f])
    return cmd, mem


def _scalar_field(s, addr, complex_mode):
    if np.ndim(s[0]) == 0:
        return {"direct": 1, "re": int(s[0]), "im": int(s[1]) if complex_mode else 0,
                "addr": 0, "stride": 0}
    return {"direct": 0, "re": 0, "im": 0, "addr": int(addr), "stride": 1}


def run(cfg, a, b, c, alpha, beta):
    accel = _accel(cfg)
    cmd, mem = build(accel, cfg, a, b, c, alpha, beta)
    dst = accel.execute(cmd, mem)
    exp_re, exp_im = oracle(cfg, a, b, c, alpha, beta)
    if cfg["mode"] == VmacMode.COMPLEX:
        got_re, got_im = np.asarray(dst.val["re"]), np.asarray(dst.val["im"])
    else:
        got_re = np.asarray(dst.val)
        got_im = np.zeros_like(got_re)
    return (got_re, got_im), (exp_re, exp_im)


# --- operand helpers ----------------------------------------------------------
def _pair(re, im=None):
    re = np.asarray(re, dtype=np.int64)
    im = np.zeros_like(re) if im is None else np.asarray(im, dtype=np.int64)
    return (re, im)


REAL = VmacMode.REAL
CPLX = VmacMode.COMPLEX


def _cfg(**kw):
    base = dict(mode=REAL, in_bw=8, int_bits=4, out_bw=8, shift=4, acc_bw=48,
                b_one=0, c_zero=0, b_conj=0, reduce_rows=0, q_rnd=0, o_sat=0)
    base.update(kw)
    return base


# --- literal hand-checked anchors --------------------------------------------
def test_scaled_copy_literal():
    # dst = 1.0 * a, F=4: alpha=16 (=1.0); a=[1.5,2.0]=[24,32] -> dst=[24,32]
    cfg = _cfg(b_one=1, c_zero=1, shift=4)
    a = _pair([[24, 32]])
    (gr, _), (er, _) = run(cfg, a, _pair([[0, 0]]), _pair([[0, 0]]), _pair(16), _pair(0))
    np.testing.assert_array_equal(gr, [[24, 32]])
    np.testing.assert_array_equal(gr, er)


def test_hadamard_literal():
    # dst = a*b, alpha=1.0, c_zero. a=[2.0]=32, b=[1.5]=24 -> 3.0; F_acc=3*4=12, shift=8 -> F_out=4
    cfg = _cfg(c_zero=1, shift=8)
    a, b = _pair([[32]]), _pair([[24]])
    (gr, _), (er, _) = run(cfg, a, b, _pair([[0]]), _pair(16), _pair(0))
    np.testing.assert_array_equal(gr, [[48]])      # 3.0 at F=4 = 48
    np.testing.assert_array_equal(gr, er)


def test_column_sum_literal():
    # b_one, c_zero, alpha=1, reduce=rows: dst = sum_rows(a). a col=[1.0,2.0,0.5]=[16,32,8] -> 3.5=56
    cfg = _cfg(b_one=1, c_zero=1, reduce_rows=1, shift=4)
    a = _pair([[16], [32], [8]])
    (gr, _), (er, _) = run(cfg, a, _pair(np.zeros((3, 1))), _pair(np.zeros((3, 1))),
                           _pair(16), _pair(0))
    np.testing.assert_array_equal(gr, [56])
    np.testing.assert_array_equal(gr, er)


# --- config sweep vs the oracle (real) ----------------------------------------
@pytest.mark.parametrize("q_rnd,o_sat", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_real_configs_vs_oracle(q_rnd, o_sat):
    rng = np.random.default_rng(1)
    n, m = 4, 3
    a = _pair(rng.integers(-120, 121, (n, m)))
    b = _pair(rng.integers(-120, 121, (n, m)))
    c = _pair(rng.integers(-120, 121, (n, m)))
    alpha_s, beta_s = _pair(20), _pair(-12)
    alpha_pc = _pair(rng.integers(-40, 41, m))
    beta_pc = _pair(rng.integers(-40, 41, m))
    configs = [
        _cfg(b_one=1, c_zero=1, shift=4, q_rnd=q_rnd, o_sat=o_sat),                  # scaled copy
        _cfg(c_zero=1, shift=8, q_rnd=q_rnd, o_sat=o_sat),                           # hadamard
        _cfg(shift=8, q_rnd=q_rnd, o_sat=o_sat),                                     # full MAC
        _cfg(b_one=1, shift=4, q_rnd=q_rnd, o_sat=o_sat),                            # alpha*a + beta*c
        _cfg(b_one=1, c_zero=1, reduce_rows=1, shift=4, q_rnd=q_rnd, o_sat=o_sat),   # column sum
        _cfg(shift=8, reduce_rows=1, q_rnd=q_rnd, o_sat=o_sat),                      # reduced MAC
    ]
    for cfg in configs:
        for al, be in [(alpha_s, beta_s), (alpha_pc, beta_pc)]:           # scalar + per-column
            got, exp = run(cfg, a, b, c, al, be)
            np.testing.assert_array_equal(got[0], exp[0], err_msg=f"{cfg} re")


# --- config sweep vs the oracle (complex) -------------------------------------
@pytest.mark.parametrize("q_rnd,o_sat", [(0, 0), (1, 1)])
def test_complex_configs_vs_oracle(q_rnd, o_sat):
    rng = np.random.default_rng(2)
    n, m = 4, 3
    def cop():
        return _pair(rng.integers(-60, 61, (n, m)), rng.integers(-60, 61, (n, m)))
    a, b, c = cop(), cop(), cop()
    alpha_s = _pair(20, -8)
    beta_s = _pair(-12, 5)
    alpha_pc = _pair(rng.integers(-30, 31, m), rng.integers(-30, 31, m))
    beta_pc = _pair(rng.integers(-30, 31, m), rng.integers(-30, 31, m))
    configs = [
        _cfg(mode=CPLX, b_one=1, c_zero=1, shift=4, q_rnd=q_rnd, o_sat=o_sat),
        _cfg(mode=CPLX, c_zero=1, shift=8, q_rnd=q_rnd, o_sat=o_sat),
        _cfg(mode=CPLX, b_conj=1, c_zero=1, shift=8, q_rnd=q_rnd, o_sat=o_sat),       # conj
        _cfg(mode=CPLX, shift=8, q_rnd=q_rnd, o_sat=o_sat),                           # full MAC
        _cfg(mode=CPLX, b_conj=1, c_zero=1, reduce_rows=1, shift=8, q_rnd=q_rnd, o_sat=o_sat),
        _cfg(mode=CPLX, shift=8, reduce_rows=1, q_rnd=q_rnd, o_sat=o_sat),
    ]
    for cfg in configs:
        for al, be in [(alpha_s, beta_s), (alpha_pc, beta_pc)]:
            got, exp = run(cfg, a, b, c, al, be)
            np.testing.assert_array_equal(got[0], exp[0], err_msg=f"{cfg} re")
            np.testing.assert_array_equal(got[1], exp[1], err_msg=f"{cfg} im")


# --- saturation + rounding are actually exercised -----------------------------
def test_saturation_triggered():
    # large product saturates at OUT_BW=8 signed: max=127. alpha=8.0, a=8.0 -> 64.0 >> ... saturates
    cfg = _cfg(c_zero=1, shift=4, o_sat=1)               # F_acc=3*4=12, shift=4 -> F_out=8 (range +-8)
    a, b = _pair([[112]]), _pair([[112]])               # 7.0 * 7.0 = 49 >> way over +-8 range
    (gr, _), (er, _) = run(cfg, a, b, _pair([[0]]), _pair(16), _pair(0))
    assert gr[0, 0] == 127                                # saturated to OUT_BW max
    np.testing.assert_array_equal(gr, er)


def test_rounding_triggered():
    # a value landing between LSBs differs under TRN vs RND
    rng = np.random.default_rng(9)
    a = _pair(rng.integers(-120, 121, (3, 2)))
    b = _pair(rng.integers(-120, 121, (3, 2)))
    trn = run(_cfg(c_zero=1, shift=7, q_rnd=0), a, b, _pair(np.zeros((3, 2))), _pair(16), _pair(0))
    rnd = run(_cfg(c_zero=1, shift=7, q_rnd=1), a, b, _pair(np.zeros((3, 2))), _pair(16), _pair(0))
    np.testing.assert_array_equal(trn[0][0], trn[1][0])
    np.testing.assert_array_equal(rnd[0][0], rnd[1][0])
    assert not np.array_equal(trn[0][0], rnd[0][0])      # the two modes genuinely differ


# --- CG op coverage (the configs the CG matrix inverse needs) ------------------
def test_cg_real_axpy_steps():
    # X = X - P·α[col]  and  P = R - P·β[col]:  a=P, b_one, α=-vec (per-col), c=X/R, β=1
    rng = np.random.default_rng(11)
    n, m = 5, 3
    P = _pair(rng.integers(-100, 101, (n, m)))
    X = _pair(rng.integers(-100, 101, (n, m)))
    neg_alpha = _pair(-rng.integers(-30, 31, m))         # -α per column
    beta_one = _pair(16)                                 # β = 1.0 in F4
    cfg = _cfg(b_one=1, shift=4)                         # alpha·P + 1·X, no reduce
    got, exp = run(cfg, P, _pair(np.zeros((n, m))), X, neg_alpha, beta_one)
    np.testing.assert_array_equal(got[0], exp[0])


def test_cg_real_r_eq_i_minus_qx():
    # R = I - QX:  a=QX, b_one, α=-1, c=I, β=1
    rng = np.random.default_rng(12)
    n, m = 4, 4
    QX = _pair(rng.integers(-100, 101, (n, m)))
    Imat = _pair(rng.integers(-100, 101, (n, m)))
    cfg = _cfg(b_one=1, shift=4)
    got, exp = run(cfg, QX, _pair(np.zeros((n, m))), Imat, _pair(-16), _pair(16))  # α=-1, β=1
    np.testing.assert_array_equal(got[0], exp[0])


def test_cg_complex_inner_product_conj():
    # ps = Σ conj(P)·S:  a=S, b=P, b_conj, c_zero, reduce=rows (complex)
    rng = np.random.default_rng(13)
    n, m = 6, 2
    S = _pair(rng.integers(-50, 51, (n, m)), rng.integers(-50, 51, (n, m)))
    P = _pair(rng.integers(-50, 51, (n, m)), rng.integers(-50, 51, (n, m)))
    cfg = _cfg(mode=CPLX, b_conj=1, c_zero=1, reduce_rows=1, shift=8)
    got, exp = run(cfg, S, P, _pair(np.zeros((n, m)), np.zeros((n, m))), _pair(16, 0), _pair(0, 0))
    np.testing.assert_array_equal(got[0], exp[0])
    np.testing.assert_array_equal(got[1], exp[1])


def test_cg_rnorm_sum_abs_sq_is_real():
    # rnorm = Σ |R|²:  a=R, b=R, b_conj, c_zero, reduce=rows -> result is REAL (im == 0)
    rng = np.random.default_rng(14)
    n, m = 6, 2
    R = _pair(rng.integers(-50, 51, (n, m)), rng.integers(-50, 51, (n, m)))
    # saturate so the (non-negative) magnitude clamps to OUT_BW max rather than wrapping
    cfg = _cfg(mode=CPLX, b_conj=1, c_zero=1, reduce_rows=1, shift=8, out_bw=12, o_sat=1)
    got, exp = run(cfg, R, R, _pair(np.zeros((n, m)), np.zeros((n, m))), _pair(16, 0), _pair(0, 0))
    np.testing.assert_array_equal(got[0], exp[0])
    np.testing.assert_array_equal(got[1], np.zeros_like(got[1]))   # |R|² is real
    assert np.all(got[0] >= 0)                                     # and non-negative (no wrap)
