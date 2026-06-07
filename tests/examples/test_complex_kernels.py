"""Phase-3 codegen tests for the ComplexField C++ kernel renderers.

Structure tests (always run) lock the per-inner C++ contract: ``std::complex`` for
float/fixed, the Waveflow ``wf_cint`` struct for int, the explicit component formula for
fixed/int arithmetic, ``std::complex`` ``operator*`` for float (the edge), interleaved
I/Q, and IEEE ``memcpy`` (de)serialization for float.  A ``@pytest.mark.vitis`` smoke
test then confirms each inner's ``cmult`` kernel **csim-compiles and runs** on real Vitis.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from examples.schemas.complex import kernels as K  # noqa: E402
from waveflow.hw.complexfield import ComplexField, cmult  # noqa: E402
from waveflow.hw.dataschema import DataArray, FloatField, IntField  # noqa: E402
from waveflow.hw.fixpoint import FixedField  # noqa: E402
from waveflow.toolchain import toolchain  # noqa: E402
from waveflow.utils import complexutils as cx  # noqa: E402
from waveflow.utils.fixputils import Format, to_bits  # noqa: E402

_RUN_TCL = _ROOT / "examples" / "schemas" / "complex" / "run.tcl"


# --- structure tests (no Vitis) -----------------------------------------------
def test_cmult_fixed_uses_explicit_formula_at_grown_format():
    k = K.render_cmult("fixed", "ap_fixed<8, 4, AP_TRN, AP_WRAP>", 8,
                       "ap_fixed<17, 9, AP_TRN, AP_WRAP>", 17)
    assert "ar * br - ai * bi" in k and "ar * bi + ai * br" in k
    assert "ap_fixed<17, 9, AP_TRN, AP_WRAP> yr" in k
    assert "std::complex" not in k.split("int main")[1]   # no operator* in the loop
    assert ".range(8 - 1, 0) = (ap_uint<8>)A[i]" in k     # reconstruct from interleaved bits
    assert 'out << (unsigned long long)yr.range(17 - 1, 0)' in k


def test_cmult_int_emits_wf_cint_and_explicit_formula():
    k = K.render_cmult("int", "ap_int<8>", 8, "ap_int<17>", 17)
    assert "struct wf_cint" in k
    assert "ar * br - ai * bi" in k and "ap_int<17> yr" in k
    assert "(unsigned long long)(ap_uint<17>)yr.range(17 - 1, 0)" in k


def test_cmult_float_uses_std_complex_operator_and_memcpy():
    k = K.render_cmult("float", "float", 32, "float", 32)
    assert "std::complex<float> a(ar, ai), b(br, bi);" in k
    assert "std::complex<float> y = a * b;" in k          # the float edge
    assert "std::memcpy" in k


def test_load_real_quantizes_fixed_and_reads_stored_int():
    kf = K.render_load_real("fixed", "ap_fixed<8, 4, AP_TRN, AP_WRAP>", 8)
    assert "while (fin >> re >> im)" in kf
    assert "ap_fixed<8, 4, AP_TRN, AP_WRAP> yr = re;" in kf
    ki = K.render_load_real("int", "ap_int<8>", 8)
    assert "read_bits(argv[1])" in ki and ">> re >> im" not in ki


def test_caddsub_and_conj_structure():
    ka = K.render_caddsub("+", "fixed", "ap_fixed<8, 4, AP_TRN, AP_WRAP>", 8,
                          "ap_fixed<9, 5, AP_TRN, AP_WRAP>", 9)
    assert "ar + br" in ka and "ai + bi" in ka
    kc = K.render_conj("fixed", "ap_fixed<8, 4, AP_TRN, AP_WRAP>", 8,
                       "ap_fixed<9, 5, AP_TRN, AP_WRAP>", 9)
    assert "yi = -ai" in kc and "read_bits(argv[2])" not in kc   # conj ignores in_b


# --- Vitis smoke: cmult csim-compiles & runs per inner ------------------------
def _da(cf, val):
    arr = np.asarray(val)
    return DataArray.specialize(cf, max_shape=(arr.shape[0],))(arr)


def _interleave(struct, W):
    re = np.atleast_1d(to_bits(np.asarray(struct["re"]), W))
    im = np.atleast_1d(to_bits(np.asarray(struct["im"]), W))
    return [int(x) for pair in zip(re, im) for x in pair]


def _float_bits(arr):
    return [int(x) for z in np.atleast_1d(arr) for x in
            (np.asarray(np.float32(z.real)).view(np.uint32),
             np.asarray(np.float32(z.imag)).view(np.uint32))]


def _csim(tmp_path, kernel, in_a, in_b):
    (tmp_path / "kernel.cpp").write_text(kernel, encoding="utf-8")
    (tmp_path / "in_a.txt").write_text("\n".join(map(str, in_a)) + "\n", encoding="utf-8")
    (tmp_path / "in_b.txt").write_text("\n".join(map(str, in_b)) + "\n", encoding="utf-8")
    shutil.copy(_RUN_TCL, tmp_path / "run.tcl")
    toolchain.run_vitis_hls(tmp_path / "run.tcl", work_dir=tmp_path, capture_output=True)
    return [int(t) for t in (tmp_path / "out_bits.txt").read_text().split()]


@pytest.mark.vitis
def test_cmult_csim_compiles_and_runs_fixed(tmp_path):
    if not toolchain.find_vitis_path():
        pytest.skip("Vitis installation not found.")
    inner = FixedField.specialize(8, 4, True)
    a = _da(ComplexField.specialize(inner), cx.make_complex([1, -2], [3, 4], Format(8, 4, True)))
    b = _da(ComplexField.specialize(inner), cx.make_complex([2, 1], [-1, 2], Format(8, 4, True)))
    out = cmult(a, b)
    ri = out.element_type.inner_type
    k = K.render_cmult("fixed", inner.cpp_type, 8, ri.cpp_type, ri.get_bitwidth())
    got = _csim(tmp_path, k, _interleave(a.val, 8), _interleave(b.val, 8))
    assert len(got) == 4   # 2 elements x (re, im) — compiles and runs


@pytest.mark.vitis
def test_cmult_csim_compiles_and_runs_int(tmp_path):
    if not toolchain.find_vitis_path():
        pytest.skip("Vitis installation not found.")
    inner = IntField.specialize(8, True)
    fmt = cx.int_format(8, True)
    a = _da(ComplexField.specialize(inner), cx.make_complex([1, -2], [3, 4], fmt))
    b = _da(ComplexField.specialize(inner), cx.make_complex([2, 1], [-1, 2], fmt))
    out = cmult(a, b)
    ri = out.element_type.inner_type
    k = K.render_cmult("int", "ap_int<8>", 8, f"ap_int<{ri.get_bitwidth()}>", ri.get_bitwidth())
    got = _csim(tmp_path, k, _interleave(a.val, 8), _interleave(b.val, 8))
    assert len(got) == 4


@pytest.mark.vitis
def test_cmult_csim_compiles_and_runs_float(tmp_path):
    if not toolchain.find_vitis_path():
        pytest.skip("Vitis installation not found.")
    cf = ComplexField.specialize(FloatField.specialize(32))
    a = _da(cf, np.array([1 + 2j, -3 + 0.5j], dtype=np.complex64))
    b = _da(cf, np.array([0.5 - 1j, 4 + 2j], dtype=np.complex64))
    k = K.render_cmult("float", "float", 32, "float", 32)
    got = _csim(tmp_path, k, _float_bits(a.val), _float_bits(b.val))
    assert len(got) == 4
