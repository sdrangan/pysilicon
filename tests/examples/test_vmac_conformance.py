"""VMAC Phase-3 milestone: the Python golden (``VmacAccel.execute``) == the Vitis HLS m_axi
kernel, bit-exact (real + complex).

The ``-m vitis`` test renders each curated case's m_axi kernel (``ap_fixed`` accumulator =
``VmacAccel.accumulator_format``; output = ``VmacAccel.output_format``), runs it in Vitis
C-sim over the same shared-memory image, and asserts the emitted dst bits equal the golden
bits with zero LSB disagreement.  A failed csim is a real failure — only skip when Vitis is
absent.  The non-vitis tests below are sanity checks on the generated cases.
"""
import pytest

from examples.vmac.vmac_build import build_cases, conformance_for_case
from waveflow.toolchain import toolchain

CASES = build_cases()


def test_cases_cover_cg_configs_and_both_modes():
    names = {c["name"] for c in CASES}
    # the CG / general configs, real and complex
    assert "full_mac_trn_real" in names and "full_mac_trn_complex" in names
    assert "axpy_percol_trn_real" in names           # CG: X - P*alpha[col] (per-column alpha)
    assert "colsum_trn_real" in names                # b_one, c_zero, reduce=rows
    assert "reduced_mac_rnd_complex" in names        # reduce + RND/SAT
    assert "conj_inner_trn_complex" in names         # CG: sum conj(P)*S (complex b_conj)
    assert not any(n.startswith("conj_inner") and n.endswith("real") for n in names)  # signed-only
    assert sum(n.endswith("real") for n in names) and sum(n.endswith("complex") for n in names)


def test_every_case_has_golden_bits():
    for c in CASES:
        assert c["expected"] and all(isinstance(b, int) and b >= 0 for b in c["expected"])
        assert "vmac_kernel" in c["kernel"]          # the rendered m_axi kernel function


@pytest.mark.vitis
@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_kernel_matches_golden_bit_exact(tmp_path, case):
    if not toolchain.find_vitis_path():
        pytest.skip("Vitis installation not found; cannot run bit-exact conformance.")
    result = conformance_for_case(case, tmp_path)
    assert result["count_ok"], f"{case['name']}: Vitis emitted a different number of outputs."
    assert result["exact"], (
        f"{case['name']}: {len(result['mismatches'])} LSB disagreement(s) between the Vitis "
        f"kernel and VmacAccel.execute — the golden is the spec, fix the kernel (do NOT loosen). "
        f"First few: {result['mismatches'][:5]}")
