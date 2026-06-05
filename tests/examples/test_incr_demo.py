"""Phase 1 tests for the increment-buffer toy (SimPy model + golden)."""
from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from examples.increment.incr import (
    IncrCmd,
    IncrError,
    IncrResp,
    build_inputs,
    golden,
    run_sim,
)


def test_increment_matches_golden():
    rng = np.random.default_rng(7)
    input_buf = rng.integers(0, 1000, size=37, dtype=np.uint32)
    res = run_sim(input_buf)
    assert res.passed
    assert res.status is IncrError.NO_ERROR
    npt.assert_array_equal(res.result, input_buf + 1)
    npt.assert_array_equal(res.result, golden(input_buf))


@pytest.mark.parametrize("n", [1, 2, 8, 100])
def test_increment_various_sizes(n):
    input_buf = np.arange(n, dtype=np.uint32) * 3 + 5
    res = run_sim(input_buf)
    assert res.passed
    npt.assert_array_equal(res.result, input_buf + 1)


def test_increment_preserves_input_buffer_independence():
    """Result is a fresh array; the kernel wrote +1 back to memory in place."""
    input_buf = np.array([10, 20, 30, 40], dtype=np.uint32)
    res = run_sim(input_buf)
    npt.assert_array_equal(res.input_buf, [10, 20, 30, 40])
    npt.assert_array_equal(res.result, [11, 21, 31, 41])


def test_build_inputs_writes_files(tmp_path):
    input_buf = np.arange(16, dtype=np.uint32)
    out = build_inputs(tmp_path, input_buf)
    assert (out / "in.bin").exists()
    assert (out / "cmd.bin").exists()
    assert (out / "params.json").exists()
    # cmd.bin round-trips n.
    cmd = IncrCmd()
    cmd.read_uint32_file(out / "cmd.bin")
    assert int(cmd.n) == 16


def test_resp_schema_roundtrip():
    resp = IncrResp()
    resp.status = IncrError.NO_ERROR
    words = resp.serialize(word_bw=32)
    recovered = IncrResp().deserialize(words, word_bw=32)
    assert recovered.status == IncrError.NO_ERROR
