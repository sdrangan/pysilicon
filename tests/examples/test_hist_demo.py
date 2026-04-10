import numpy as np
import pytest

from examples.histogram.hist_demo import HistError, HistTest


@pytest.mark.parametrize("mem_dwidth", [32, 64, 128])
def test_hist_test_simulate_matches_expected_counts(mem_dwidth: int) -> None:
    hist_test = HistTest(seed=11, ndata=41, nbins=7, mem_dwidth=mem_dwidth)

    result = hist_test.simulate()

    assert hist_test.mem is not None
    assert hist_test.hist_accel is not None
    assert hist_test.cmd is not None
    assert hist_test.resp is not None
    assert hist_test.counts is not None
    assert hist_test.expected is not None

    assert hist_test.mem.word_size == mem_dwidth
    assert result.cmd is hist_test.cmd
    assert result.resp is hist_test.resp
    assert result.counts is hist_test.counts
    assert result.expected is hist_test.expected
    assert result.passed is True

    assert hist_test.resp.tx_id == hist_test.cmd.tx_id
    assert hist_test.resp.status is HistError.NO_ERROR
    assert hist_test.counts.dtype == np.uint32
    assert hist_test.expected.dtype == np.uint32
    assert np.array_equal(hist_test.counts, hist_test.expected)


def test_hist_test_gen_test_data_initializes_state_before_simulate() -> None:
    hist_test = HistTest(seed=5, ndata=13, nbins=4, mem_dwidth=64)

    hist_test.gen_test_data()

    assert hist_test.cmd is None
    assert hist_test.data is not None
    assert hist_test.bin_edges is not None
    assert hist_test.data.shape == (13,)
    assert hist_test.bin_edges.shape == (3,)

    result = hist_test.simulate()

    assert hist_test.cmd is not None
    assert result.passed is True