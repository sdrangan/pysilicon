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
array-utils), and `gen_kernel` (the m_axi kernel).  The `csim` / `validate_csim`
steps are **Vitis-gated** (`pytest -m vitis`) — C-sim itself requires a Vitis
install to execute and was not run in the codegen environment.

