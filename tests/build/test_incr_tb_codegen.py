"""Phase 5 tests: m_axi testbench codegen for the increment toy.

Lowers ``IncrTBHls.main()`` to ``incr_tb.cpp`` via the four new TB statement
types (decision 9): MemComponent → static array + MemMgr; alloc_array →
mgr.alloc + byte-addr + write_array; read_array → read_array; KernelCallStmt
includes the mem pointer in canonical signature order.
"""
from __future__ import annotations

from examples.increment.incr import IncrTBHls
from pysilicon.build.hwgen import tb_files_to_str


def _gen() -> str:
    return tb_files_to_str(IncrTBHls, output_dir="gen")["incr_tb.cpp"]


def test_membind_lowers_to_array_and_memmgr():
    cpp = _gen()
    assert "static ap_uint<32> mem[1024] = {};" in cpp
    assert "pysilicon::memmgr::MemMgr<32> mem_mgr(mem, 1024);" in cpp


def test_alloc_array_lowers_to_alloc_byteaddr_writearray():
    cpp = _gen()
    assert "mem_mgr.alloc(uint32_array_utils::get_nwords<32>(cmd.n))" in cpp
    assert "cmd.addr = _cmd_addr_widx * (32 / 8);" in cpp
    assert "uint32_array_utils::write_array<32>(buf, mem + _cmd_addr_widx, cmd.n);" in cpp


def test_read_array_lowers_to_buffer_and_readarray():
    cpp = _gen()
    assert "static ap_uint<32> out[1024] = {};" in cpp
    assert (
        "uint32_array_utils::read_array<32>("
        "mem + pysilicon::memmgr::byte_addr_to_word_index<32>(cmd.addr), out, cmd.n);"
        in cpp
    )


def test_kernel_call_includes_mem_in_canonical_order():
    cpp = _gen()
    # streams first, then the m_axi pointer (no regmap on this toy).
    assert "incr(s_in, m_out, mem);" in cpp


def test_includes_memmgr_and_command_read():
    cpp = _gen()
    assert '#include "include/memmgr_tb.hpp"' in cpp
    assert '#include "include/memmgr.hpp"' in cpp
    assert '#include "include/uint32_array_utils_tb.h"' in cpp
    # command read from cmd.bin (schema struct), not a JSON parse.
    assert "streamutils::read_uint32_file(cmd," in cpp
    assert "json_parse" not in cpp


def test_writes_verify_outputs():
    cpp = _gen()
    assert "write_uint32_file(resp," in cpp
    assert 'resp_data.bin' in cpp
    assert "write_uint32_file_array(out," in cpp
    assert 'out_data.bin' in cpp
