# incr codegen — diff notes vs the histogram

The increment toy's generated `m_axi` kernel is a smaller `examples/histogram/hist.cpp`.
Each codegen phase is diffed against the corresponding hand-written histogram file.

## Phase 3 — kernel (`gen/incr.cpp` / `gen/incr.hpp`) vs `hist.cpp` / `hist.hpp`

**Same shape (the load-bearing m_axi parts):**

- `.hpp` includes: `streamutils_hls.h`, the schema headers, `memmgr.hpp`, and
  one `<elem>_array_utils.h` per element type read/written over `m_axi`.
- `.cpp` aliases `namespace memmgr = pysilicon::memmgr;`.
- Signature: `void <name>(hls::stream<axi4s_word<W>>& in, hls::stream<axi4s_word<W>>& out, ap_uint<MW>* mem)`.
- Pragmas: `axis` per stream, `m_axi port=mem offset=slave bundle=gmem depth=<MAX>`,
  and `ap_ctrl_hs port=return` (stream control — no `s_axilite`, decision 2/3).
- Body: read command from the input stream → `static <elem> buf[MAX];` →
  `<elem>_array_utils::read_array<MW>(mem + memmgr::byte_addr_to_word_index<MW>(addr), buf, n);`
  → transform → `write_array<MW>(buf, mem + byte_addr_to_word_index<MW>(addr), n);`.

**Intentionally smaller than `hist.cpp`:**

- One buffer (`buf`) vs three (`data_buf` / `edge_buf` / `count_buf`).
- No validation branches / early-return error paths (hist checks ndata/nbins/alignment).
- The transform is delegated to the `incr_impl::transform(buf, n)` hook (in-place +1)
  rather than inlined; the response is delegated to `incr_impl::respond(out)`.
  (hist inlines its histogram loops and writes the response inline.)
- `depth=<MAX>` comes from the `max_n` HwParam (decision 5); hist uses `max_mem_words`.

The buffer max is sourced from the component's `max_n` `HwParam`; codegen raises
`SynthesisError` if a buffered read has no such bound (decision 5, fail-loud).

## Phase 4 — hand-written `incr_tb.cpp` vs `hist_tb.cpp`

Same flow (decisions 7–10): read scalar params → `read_uint32_file_array` the
input → `MemMgr<bw> mgr(mem, MEM_SIZE)` + `mgr.alloc(nwords)` (alloc order
preserved, address not baked) → `write_array` populate → build the command with
the allocated byte address → push command + `incr(in, out, mem)` → read the
response → `read_array` the kernel-produced buffer back → write `out_data.bin` +
`resp_data.bin` for `FunctionalVerifyStep`.

Smaller than `hist_tb.cpp`: one region (`in_buf`/`out_buf`) vs three; one scalar
(`n`) vs three (`tx_id`/`ndata`/`nbins`); no edge buffer.  The hand-written TB
uses literal `ap_uint<32>` / `axi4s_word<32>` types (the generated `incr.hpp`
emits literal widths, not the `mem_word_t`/`stream_dwidth` typedefs hist.hpp
defines).

## Milestone status (decision 11)

The build DAG (`incr_build.py`) and all non-Vitis branches are green:
`build_inputs`, `py_sim` (golden = in+1), `gen_include` (schemas + memmgr +
array-utils), `gen_kernel` (the m_axi kernel), and `gen_tb` (the testbench).
The generated kernel passes Vitis C-sim against the Python model (verified on a
Vitis machine; the `-Igen` include-path fix in commit 272ad0b made it build).

## Phase 5 — generated `gen/incr_tb.cpp` vs the hand-written `incr_tb.cpp`

`IncrTBHls.main()` now lowers to `gen/incr_tb.cpp` via the four new TB statement
types (decision 9), and the hand-written `incr_tb.cpp` is deleted (replaced):

- `mem = MemComponent(..., nwords_tot=MAX_N)` → `MemBindStmt` →
  `static ap_uint<32> mem[1024] = {}; pysilicon::memmgr::MemMgr<32> mem_mgr(mem, 1024);`
- `cmd.addr = mem.alloc_array(buf, Uint32Field, count=cmd.n)` → `MemAllocArrayStmt` →
  `mem_mgr.alloc(uint32_array_utils::get_nwords<32>(cmd.n))` (word index) →
  `cmd.addr = widx * (32/8)` → `uint32_array_utils::write_array<32>(buf, mem+widx, cmd.n)`.
- `out = mem.read_array(cmd.addr, Uint32Field, count=cmd.n)` → `MemReadArrayStmt` →
  `static ap_uint<32> out[1024] = {};` +
  `uint32_array_utils::read_array<32>(mem + pysilicon::memmgr::byte_addr_to_word_index<32>(cmd.addr), out, cmd.n)`.
- `dut.run(mem=mem)` → `KernelCallStmt(mem_local="mem")` → `incr(s_in, m_out, mem)`
  (mem appended in canonical signature order: streams, regmap, m_axi).

**Functionally equivalent to the hand-written TB; cosmetic/structural diffs:**

- Reads the command from `cmd.bin` (schema struct read) instead of parsing
  `params.json` by hand — no JSON-parse block in the generated file.
- Readback converts the byte address with `byte_addr_to_word_index<32>` instead
  of reusing the alloc word-index local.
- Drops the `sync_status.json` write (it was never consumed by
  `FunctionalVerifyStep`).
- Local names follow the framework convention (`s_in`/`m_out`, `mem_mgr`,
  `buf`/`out`) rather than the hand-written `in_stream`/`out_stream`/`mgr`.

`run.tcl` adds the generated TB with `-cflags "-I. -Igen"` so the gen/ header is
on the include path. **C-sim with the generated TB is verified separately on a
Vitis machine** (not run in the codegen environment).

