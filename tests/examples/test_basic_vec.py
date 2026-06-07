"""Phase 2 milestone: basic_vec — vectorized Python golden == vectorized Vitis, bit-exact.

The -m vitis test runs the int/float/fixed MAC (y = a*b + c) in Vitis C-sim and asserts
the bits equal the Python operator golden, zero LSB disagreement. A failed csim is a
real failure; only skip when Vitis is absent.
"""
import pytest

from examples.basic_vec.basic_vec_build import build_cases, conformance_for_case
from pysilicon.toolchain import toolchain

CASES = build_cases()


def test_three_kinds_with_golden():
    assert {c["name"] for c in CASES} == {"int_mac", "float_mac", "fixed_mac"}
    for c in CASES:
        assert c["expected"] and len(c["a"]) == len(c["b"]) == len(c["c"]) == len(c["expected"])


@pytest.mark.vitis
@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_vectorized_python_matches_vitis_bit_exact(tmp_path, case):
    if not toolchain.find_vitis_path():
        pytest.skip("Vitis installation not found; cannot run basic_vec conformance.")
    r = conformance_for_case(case, tmp_path)
    assert r["count_ok"], f"{case['name']}: wrong output count"
    assert r["exact"], f"{case['name']}: {len(r['mismatches'])} LSB disagreement(s): {r['mismatches'][:5]}"
