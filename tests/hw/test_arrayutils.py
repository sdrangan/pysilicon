from pathlib import Path

import numpy as np

from pysilicon.build.build import CodeGenConfig
from pysilicon.hw.arrayutils import gen_array_utils, read_uint32_file, write_array, write_uint32_file
from pysilicon.hw.dataschema import FloatField, IntField


F32 = FloatField.specialize(bitwidth=32)
S16 = IntField.specialize(bitwidth=16, signed=True)


def test_write_uint32_file_roundtrip_int16(tmp_path: Path):
    data = np.array([-32768, -12345, -17, -1, 0, 1, 23, 255, 1024, 32767], dtype=np.int16)
    out_path = write_uint32_file(data, elem_type=S16, file_path=tmp_path / "arr.bin")

    words = np.fromfile(out_path, dtype="<u4")
    expected = np.asarray(write_array(data, elem_type=S16, word_bw=32), dtype="<u4")
    got = np.asarray(read_uint32_file(out_path, elem_type=S16, shape=data.size), dtype=np.int16)

    assert np.array_equal(words, expected)
    assert np.array_equal(got, data)


def test_write_uint32_file_nwrite_selects_prefix(tmp_path: Path):
    data = np.arange(7, dtype=np.int16)
    out_path = write_uint32_file(data, elem_type=S16, file_path=tmp_path / "prefix.bin", nwrite=5)

    words = np.fromfile(out_path, dtype="<u4")
    expected = np.asarray(write_array(data[:5], elem_type=S16, word_bw=32), dtype="<u4")
    got = np.asarray(read_uint32_file(out_path, elem_type=S16, shape=5), dtype=np.int16)

    assert np.array_equal(words, expected)
    assert np.array_equal(got, data[:5])


def test_write_uint32_file_write_slice_selects_subarray(tmp_path: Path):
    data = np.arange(12, dtype=np.float32).reshape(4, 3)
    out_path = write_uint32_file(
        data,
        elem_type=F32,
        file_path=tmp_path / "slice.bin",
        write_slice=np.s_[1:3, :],
    )

    words = np.fromfile(out_path, dtype="<u4")
    expected_data = data[1:3, :]
    expected = np.asarray(write_array(expected_data, elem_type=F32, word_bw=32), dtype="<u4")
    got = np.asarray(read_uint32_file(out_path, elem_type=F32, shape=expected_data.shape), dtype=np.float32)

    assert np.array_equal(words, expected)
    assert np.array_equal(got, expected_data)


def test_write_uint32_file_rejects_conflicting_selection_args(tmp_path: Path):
    data = np.arange(4, dtype=np.int16)

    try:
        write_uint32_file(
            data,
            elem_type=S16,
            file_path=tmp_path / "invalid.bin",
            write_slice=np.s_[1:3],
            nwrite=2,
        )
    except ValueError as exc:
        assert "Specify only one of write_slice or nwrite." in str(exc)
    else:
        raise AssertionError("Expected ValueError when both write_slice and nwrite are provided.")


def test_gen_array_utils_writes_companion_tb_header(tmp_path: Path):
    Int16Inc = IntField.specialize(bitwidth=16, signed=True, include_dir="include")

    out_path = gen_array_utils(Int16Inc, [32], cfg=CodeGenConfig(root_dir=tmp_path, util_dir="common"))
    tb_path = tmp_path / "include" / "int16_array_utils_tb.h"

    content = out_path.read_text(encoding="utf-8")
    tb_content = tb_path.read_text(encoding="utf-8")

    assert out_path == tmp_path / "include" / "int16_array_utils.h"
    assert tb_path.exists()
    assert "#ifndef INCLUDE_INT16_ARRAY_UTILS_TB_H" in tb_content
    assert '#include "../common/streamutils_tb.h"' in tb_content
    assert '#include "int16_array_utils.h"' in tb_content
    assert '#include "../common/streamutils_hls.h"' in content
    assert '#include <hls_stream.h>' in content
    assert '#if __has_include(<hls_axi_stream.h>)' in content
    assert f"namespace {out_path.stem} {{" in content
    assert f"namespace {out_path.stem} {{" in tb_content
    assert "static constexpr int pf() {" in content
    assert "return word_bw / 16;" in content
    assert "inline void read_array_elem(const ap_uint<word_bw>* src, value_type out[pf<word_bw>()], int n = pf<word_bw>()) {" in content
    assert "inline void write_array_elem(const value_type in[pf<word_bw>()], ap_uint<word_bw>* dst, int n = pf<word_bw>()) {" in content
    assert "inline void read_stream_elem(hls::stream<ap_uint<word_bw>>& s, value_type out[pf<word_bw>()], int n = pf<word_bw>()) {" in content
    assert "inline void read_axi4_stream_elem(hls::stream<hls::axis<ap_uint<word_bw>, 0, 0, 0>>& s, value_type out[pf<word_bw>()], int n = pf<word_bw>()) {" in content
    assert "inline void write_stream_elem(hls::stream<ap_uint<word_bw>>& s, const value_type in[pf<word_bw>()], int n = pf<word_bw>()) {" in content
    assert "inline void write_axi4_stream_elem(hls::stream<hls::axis<ap_uint<word_bw>, 0, 0, 0>>& s, const value_type in[pf<word_bw>()], bool tlast = false, int n = pf<word_bw>()) {" in content
    assert "inline void read_stream(hls::stream<ap_uint<word_bw>>& s, value_type* dst, int len) {" in content
    assert "inline void read_axi4_stream(hls::stream<hls::axis<ap_uint<word_bw>, 0, 0, 0>>& s, value_type* dst, int len) {" in content
    assert "inline void write_stream(hls::stream<ap_uint<word_bw>>& s, const value_type* src, int len) {" in content
    assert "inline void write_axi4_stream(hls::stream<hls::axis<ap_uint<word_bw>, 0, 0, 0>>& s, const value_type* src, bool tlast = true, int len = pf<word_bw>()) {" in content
    assert "read_stream_elem<word_bw>(s, dst + i, len - i);" in content
    assert "read_axi4_stream_elem<word_bw>(s, dst + i, len - i);" in content
    assert "write_stream_elem<word_bw>(s, src + i, len - i);" in content
    assert "const bool lane_tlast = (i + pf<word_bw>() >= len) ? tlast : false;" in content
    assert "write_axi4_stream_elem<word_bw>(s, src + i, lane_tlast, len - i);" in content
    assert "ap_uint<32> w = src[0];" in content
    assert "dst[0] = w;" in content
    assert "ap_uint<32> w = s.read().data;" in content
    assert "streamutils::write_axi4_word<32>(s, w, tlast);" in content
    assert "inline void read_uint32_file_array(value_type* dst, const char* file_path, int n0) {" in tb_content
    assert "inline void write_uint32_file_array(const value_type* src, const char* file_path, int n0) {" in tb_content
    assert "const int nwords = (n0 * 16 + 31) / 32;" in tb_content
    assert "words.push_back(streamutils::read_le_uint32(ifs));" in tb_content
    assert "streamutils::write_le_uint32(ofs, static_cast<uint32_t>(word));" in tb_content


def test_gen_array_utils_tb_header_uses_local_streamutils_path(tmp_path: Path):
    out_path = gen_array_utils(F32, [32], cfg=CodeGenConfig(root_dir=tmp_path))
    tb_path = tmp_path / "float32_array_utils_tb.h"
    content = out_path.read_text(encoding="utf-8")
    tb_content = tb_path.read_text(encoding="utf-8")

    assert out_path == tmp_path / "float32_array_utils.h"
    assert '#include "streamutils_hls.h"' in content
    assert '#include "streamutils_tb.h"' in tb_content
    assert '#include "float32_array_utils.h"' in tb_content