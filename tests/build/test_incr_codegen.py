"""Phase 3 tests: m_axi kernel codegen for the increment toy.

Diff target: examples/histogram/hist.cpp (signature, m_axi pragma,
byte_addr_to_word_index, array_utils read/write).
"""
from __future__ import annotations

from examples.increment.incr import IncrAccel
from pysilicon.build.hwgen import header_to_cpp, kernel_signature, kernel_to_cpp
from pysilicon.simulation.simulation import Simulation


def test_kernel_signature_has_m_axi_and_ap_ctrl_hs():
    comp = IncrAccel(name="incr", sim=Simulation())
    sig = kernel_signature(comp)
    expected = [
        "void incr(",
        "hls::stream<streamutils::axi4s_word<32>>& s_in",
        "hls::stream<streamutils::axi4s_word<32>>& m_out",
        "ap_uint<32>* m_mem",
        "#pragma HLS INTERFACE axis port=s_in",
        "#pragma HLS INTERFACE axis port=m_out",
        "#pragma HLS INTERFACE m_axi port=m_mem offset=slave bundle=gmem depth=1024",
        "#pragma HLS INTERFACE ap_ctrl_hs port=return",
    ]
    for sub in expected:
        assert sub in sig, f"missing {sub!r} in:\n{sig}"
    # Stream-controlled kernel: no s_axilite control anywhere.
    assert "s_axilite" not in sig
    assert "template" not in sig


def test_kernel_body_lowers_mm_read_write():
    cpp = kernel_to_cpp(IncrAccel)
    # byte→word address conversion + array-utils bursts (mirrors hist.cpp).
    assert (
        "static ap_uint<32> buf[1024];" in cpp
    ), cpp
    assert (
        "uint32_array_utils::read_array<32>("
        "m_mem + memmgr::byte_addr_to_word_index<32>(cmd.addr), buf, cmd.n);"
        in cpp
    ), cpp
    assert (
        "uint32_array_utils::write_array<32>("
        "buf, m_mem + memmgr::byte_addr_to_word_index<32>(cmd.addr), cmd.n);"
        in cpp
    ), cpp
    # The kernel reads the command then calls the transform + respond hooks.
    assert "cmd.read_axi4_stream<32>(s_in);" in cpp
    assert "incr_impl::transform(buf, cmd.n);" in cpp
    assert "incr_impl::respond(m_out);" in cpp
    # memmgr namespace alias present (byte_addr_to_word_index lives there).
    assert "namespace memmgr = pysilicon::memmgr;" in cpp


def test_header_includes_memmgr_and_array_utils_once():
    hpp = header_to_cpp(IncrAccel)
    assert hpp.count('#include "include/memmgr.hpp"') == 1
    assert hpp.count('#include "include/uint32_array_utils.h"') == 1
    assert '#include "include/incr_cmd.h"' in hpp
    # Concrete kernel forward decl + stream-controlled (no s_axilite).
    assert "void incr(" in hpp
    assert "ap_uint<32>* m_mem" in hpp


def test_no_regmap_component_unaffected_uses_ap_ctrl_hs():
    """Sanity: the toy has no regmap, so control is ap_ctrl_hs, not s_axilite."""
    comp = IncrAccel(name="incr", sim=Simulation())
    sig = kernel_signature(comp)
    assert sig.count("port=return") == 1
    assert "ap_ctrl_hs port=return" in sig
