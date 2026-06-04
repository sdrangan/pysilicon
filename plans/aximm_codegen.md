# AXI-MM HLS Codegen (increment-buffer toy)

## Goal

Generate a Vitis HLS kernel **and** testbench, from Python, for an `HwComponent`
that reads and writes external memory over an AXI-MM (`m_axi`) interface — and
prove it with C-sim against the Python model. This is the **first `m_axi`
codegen in the repo**; today only AXI-Stream and AXI-Lite (regmap) generate.

The vehicle is the smallest example that fully stresses the path: an
**increment-a-buffer** accelerator. It exercises m_axi read + write, byte→word
address conversion, a local buffer, a stream command, and a stream response —
everything the queue and a real histogram will later need.

## Why this de-risks the whole AXI-MM program

The hard runtime **already exists and is cosim-proven by the hand-written
histogram**:

- [examples/histogram/hist.cpp](../examples/histogram/hist.cpp) — a working
  `m_axi` kernel; this plan's generated `incr.cpp` is a smaller version of it.
- [examples/histogram/hist_tb.cpp](../examples/histogram/hist_tb.cpp) — a working
  `m_axi` testbench; this plan's generated `incr_tb.cpp` targets its exact shape.
- `include/memmgr.hpp` / `memmgr_tb.hpp` — fixed support headers
  (`byte_addr_to_word_index<bw>`, `MemMgr` first-fit allocator). Already present.
- `<elem>_array_utils::read_array<bw>(ptr, buf, n)` / `write_array<bw>(...)` —
  the m_axi copy-loop primitives, **already generated** by `ArrayUtilsStep`.
- `MemComponent` ([pysilicon/hw/memory.py](../pysilicon/hw/memory.py)) — the
  Python memory SimObj, completed by [plans/memory_simobj.md](memory_simobj.md).
- The statement IR even has a placeholder comment for `MMArrayReadStmt`.

So the work is **teaching the generator to emit what the histogram proves by
hand** — not building a runtime from scratch. Each codegen phase diffs its
output against the corresponding hand-written histogram file.

## Scope

**In scope:** the toy `HwComponent` + Python sim; m_axi *kernel* codegen; m_axi
*testbench* codegen; a full build DAG to **C-sim** verified against the Python
model.

**Stretch (last phase):** C-synth + cosim + VCD burst-extraction (reuse the
histogram's existing machinery) to validate real AXI bursts.

**Out of scope:** the AXI-MM queue's own codegen (this unblocks it — see
"Future"); `s_axilite`/offset-register control (we use stream control, per
decision 2); richer memory features (multiple bundles, scatter/gather).

## The central mapping: master / interconnect / slave collapse in codegen

In the **Python sim** four distinct objects model a memory transaction:

```
kernel.m_mem (MMIFMaster) ──DirectMMIF/crossbar──▶ MemComponent.s_mm (MMIFSlave) ──▶ Memory (+latency)
```

In **Vitis codegen** three of the four vanish:

```
kernel:  void incr(hls::stream<…>& in, hls::stream<…>& out, ap_uint<W>* mem)  // master port → m_axi pointer
TB:      static ap_uint<W> mem[MEM_SIZE];   // MemComponent + Memory  → a flat C array
         incr(in_stream, out_stream, mem);  // the interconnect       → just passing the pointer
```

The interconnect, the slave endpoint, and the **latency model do not codegen**.
The `clk`/`latency_*` knobs on `MemComponent` are a *Python timing prediction*
whose job is to be validated against cosim (the cycle-model loop) — they are
never emitted. C-sim is purely functional: the kernel reads/writes a plain
array.

## Design decisions (settled — do NOT re-litigate)

### Example & control

1. **Vehicle = increment-a-buffer.** Command schema `IncrCmd{addr: MemAddr,
   n: uint32}`; the kernel reads `n` words from `mem[addr]`, adds 1, writes them
   back in place; response `IncrResp{status}`. One address, one buffer, one
   transform — minimal but complete.

2. **Control via AXI-Stream + `ap_ctrl_hs`** (copy the histogram), not
   `s_axilite`. The command/response ride streams; this reuses existing stream
   codegen and adds **no** new control-path generation. (An `s_axilite`/offset
   variant is "Future".)

### Kernel-side codegen

3. **`m_axi` port mapping.** A `MMIFMaster` port on the component → a
   `mem_word_t* <port>` kernel parameter + pragma
   `#pragma HLS INTERFACE m_axi port=<port> offset=slave bundle=gmem depth=<MAX>`
   (mirrors [hist.cpp:9](../examples/histogram/hist.cpp#L9)). Plus
   `#pragma HLS INTERFACE ap_ctrl_hs port=return`.

4. **Lower MM access to the existing `array_utils` primitives.**
   `read_array(ElemT, n, addr)` → `<elem>_array_utils::read_array<bw>(<port> +
   memmgr::byte_addr_to_word_index<bw>(addr), <buf>, n)`. `write_array` →
   the dual. No new C++ runtime — these functions already exist and burst
   correctly under HLS.

5. **Local buffer sizing — the one genuinely new concept.** Stream codegen never
   buffers; m_axi does: `forward()` reads `n` (runtime) elements into a buffer
   that HLS needs sized at compile time. Codegen declares
   `static <ctype> <buf>[<MAX>]` where `<MAX>` comes from a `HwParam`/schema
   bound on the component (e.g. `max_n`). The read fills `[0,n)`. Document the
   bound; fail loudly if a buffered read has no resolvable max.

6. **Address provenance.** The address is a `MemAddr` command field; codegen
   threads `cmd.addr` through `byte_addr_to_word_index<bw>` (decision 4). The
   `MemAddr` field type already exists.

### Testbench-side codegen

7. **`MemComponent` → flat array + `MemMgr`.** A `MemComponent` in the TB
   `main()` lowers to `static mem_word_t mem[MEM_SIZE] = {};` +
   `pysilicon::memmgr::MemMgr<bw> mgr(mem, MEM_SIZE);`. `MEM_SIZE` is a
   compile-time max from `MemComponent.nwords_tot` (static arrays need a fixed
   size).

8. **Preserve allocation order; do not bake addresses.** Byte addresses come
   from `Memory.alloc` first-fit *in order*. The generated TB emits
   `mgr.alloc(nwords)` calls **in the same order** the Python TB allocated, so it
   stays parametric in `n` (this is what [hist_tb.cpp:65-67](../examples/histogram/hist_tb.cpp#L65-L67)
   does). Do **not** resolve addresses in Python and emit them as constants.

9. **The four new TB constructs** (everything else already generates):
   - `MemComponent(...)` → array + `MemMgr` decl (decision 7).
   - `mem.alloc_array(buf, count=n)` → `int widx = mgr.alloc(nwords);
     ap_uint<AW> addr = widx*bpw;` + `array_utils::write_array<bw>(buf, mem+widx, n)`.
   - `mem.read_array(addr, ElemT, count=n)` → `array_utils::read_array<bw>(mem+widx, buf, n)`.
   - `dut.run(mem=mem)` → `KernelCallStmt` that now includes the `mem` pointer in
     the **canonical signature order** (kept in sync with `kernel_signature()`).

10. **Read-back after the call is kernel output.** A `mem.read_array(...)` that
    follows `dut.run()` reads kernel-produced data and is written to a file for
    `FunctionalVerifyStep` to compare against the Python model — exactly the
    stream `pop` → `write_uint32_file` pattern, but sourced from memory.

### Phasing

11. **Prove the kernel before automating the TB.** Generate the kernel first and
    get C-sim green with a **hand-written** `incr_tb.cpp` (Phase 4). Only then
    generate the TB (Phase 5) and diff it against the hand-written one. This
    isolates "does m_axi kernel codegen work" from "does TB codegen work."

## Reference reading (read once before starting)

- [examples/histogram/hist.cpp](../examples/histogram/hist.cpp) /
  [hist.hpp](../examples/histogram/hist.hpp) — kernel diff target (signature,
  pragmas, `read_array`/`write_array`, `byte_addr_to_word_index`).
- [examples/histogram/hist_tb.cpp](../examples/histogram/hist_tb.cpp) — TB diff
  target (the `MemMgr` alloc → populate → call → read-back flow).
- [examples/poly/poly.py](../examples/poly/poly.py) `PolyAccelComponent` +
  `PolyTBHls` — the codegen-source patterns to mirror (synthesizable `on_start`,
  the sequential TB `main()`).
- [pysilicon/build/hwgen.py](../pysilicon/build/hwgen.py) — `kernel_signature`,
  `_discover_stream_endpoints`, `_emit_stream_get/write`, and the testbench
  emitters (`tb_files_to_str`, `_emit_tb_stream_io`). The functions to extend.
- [pysilicon/build/hwcodegen.py](../pysilicon/build/hwcodegen.py) +
  `hwstmt.py` — the statement IR + extractor (note the `MMArrayReadStmt`
  placeholder comment).
- [pysilicon/hw/memory.py](../pysilicon/hw/memory.py) — `MemComponent`,
  `MMIFMaster` schema/array methods to make synthesizable.

## Working convention

- One commit per phase, in order; push after each. Single PR, multiple commits.
- After every phase: `pytest tests/hw/ tests/build/ tests/examples/ -k "not vitis"`
  green (note, don't fix, pre-existing unrelated failures).
- C-sim/cosim phases require Vitis — gate them under `-m vitis` like the
  histogram does; the non-vitis suite must still pass without a toolchain.
- Each codegen phase **diffs its output against the named histogram file** and
  records the diff in a sandbox note.
- If a buffered MM read has no resolvable compile-time max (decision 5), or a
  TB alloc order can't be reproduced (decision 8), STOP and ask.

---

## Phase 1: Python toy + SimPy sim + golden model

**Goal:** the all-Python `IncrAccel` + `IncrTBHls` + a runnable sim, before any
codegen. Establishes the golden reference C-sim will check against.

**Changes:** new `examples/increment/` (or `examples/aximm_incr/`):

- Schemas: `IncrCmd{addr: MemAddr, n: uint32}`, `IncrResp{status}`.
- `IncrAccel(HwComponent)`: a stream slave `s_in` for the command, a stream
  master `m_out` for the response, an `MMIFMaster` `m_mem`, and a `max_n`
  `HwParam`. The synthesizable kernel body lives in **`run_proc`** — the
  extractor (`hwcodegen.extract_kernel`) roots at `on_start` only for a
  regmap-controlled kernel; this toy is stream-controlled (no regmap), so the
  root is `run_proc` (as implemented in Phase 1). Body: `cmd = s_in.get(IncrCmd)`;
  `buf = m_mem.read_array(Uint32Field, cmd.n, cmd.addr)`; `transform(buf, cmd.n)`
  (the in-place +1 hook — C++ can't return an array by value);
  `m_mem.write_array(buf, Uint32Field, cmd.addr, cmd.n)`; `respond(m_out)` (the
  response-write hook).
- `IncrTBHls(HwTestbench)`: the sequential `main()` from the mapping above
  (MemComponent, read input file, `alloc_array`, push cmd, `run(mem=mem)`, pop
  resp, `read_array`, write output file).
- A SimPy harness wiring `IncrAccel` + a controller + `MemComponent` over a
  `DirectMMIF`, plus a numpy golden model (`out = in + 1`). `build_inputs` writes
  `in.bin` + `params.json`.

**Tests:** `tests/examples/test_incr_demo.py` — run the SimPy sim, assert the
accelerator's memory result equals `in + 1`.

**Commit:** `incr: increment-buffer toy — HwComponent + HwTestbench + SimPy model`

---

## Phase 2: MM access becomes synthesizable (IR)

**Goal:** the extractor recognizes `m_mem.read_array(...)` / `write_array(...)`.

**Changes:**

- Add `MMArrayReadStmt` / `MMArrayWriteStmt` to the statement IR
  ([hwstmt.py](../pysilicon/build/hwcodegen.py)) — `SynthCallStmt` subclasses
  carrying `(port, elem_type, count_expr, addr_expr, target_var)`.
- Decorate `MMIFMaster.read_array` / `write_array` with `@synthesizable`
  (`synth_fn` + `stmt_class`), mirroring `StreamIFSlave.get` → `StreamGetStmt`.
  (Add `read_schema`/`write_schema` variants only if the toy needs them — it
  doesn't; defer.)

**Tests:** extract `IncrAccel().on_start` and assert the IR contains an
`MMArrayReadStmt` (port `m_mem`, elem `Uint32Field`, count `cmd.n`, addr
`cmd.addr`) and an `MMArrayWriteStmt`. No C++ yet.

**Commit:** `hwcodegen: MMArrayRead/WriteStmt — synthesizable m_axi array access`

---

## Phase 3: Kernel m_axi codegen

**Goal:** generate `incr.cpp` / `incr.hpp` with a correct `m_axi` interface and
lowered read/write — diffing against `hist.cpp`'s shape.

**Changes:** in [hwgen.py](../pysilicon/build/hwgen.py):

- `_discover_mm_masters(comp)` (alongside `_discover_stream_endpoints`).
- Extend `kernel_signature()`: append `mem_word_t* <port>` args and emit
  `#pragma HLS INTERFACE m_axi port=<port> offset=slave bundle=gmem depth=<MAX>`
  + the `ap_ctrl_hs` return pragma (decision 3).
- `to_cpp()` cases for `MMArrayReadStmt`/`MMArrayWriteStmt` →
  `_emit_mm_array_read/write()` producing the `array_utils::read_array<bw>(port +
  byte_addr_to_word_index<bw>(addr), buf, n)` calls (decision 4).
- Declare the local buffer `static <ctype> <buf>[<MAX>]` from the `max_n` bound
  (decision 5). Add the `#include "include/memmgr.hpp"` + array-utils includes.

**Tests:** generate into `examples/increment/gen/`; assert the emitted signature
+ pragmas match the expected m_axi shape, and that the read/write lower to the
array-utils calls. Capture a diff vs `hist.cpp` in a sandbox note.

**Commit:** `hwgen: m_axi kernel codegen — signature, pragmas, array read/write lowering`

---

## Phase 4: Hand-written TB + C-sim green (the milestone)

**Goal:** prove the **generated kernel** is functionally correct under Vitis
C-sim, using a hand-written testbench.

**Changes:**

- Hand-write `examples/increment/incr_tb.cpp` following the mapping (decisions
  7–10) — essentially a trimmed `hist_tb.cpp`.
- Build DAG (`examples/increment/incr_build.py`): `DataSchemaStep`,
  `ArrayUtilsStep(Uint32Field)`, `StreamUtilsStep`, `HlsCodegenStep` (kernel),
  `BuildInputsStep`, `PySimStep`, `CSimStep`, `FunctionalVerifyStep` comparing
  `out.bin` to the Python model. Model after `poly_build.py` + the histogram.

**Tests (vitis-gated):** `pytest -m vitis` runs C-sim and asserts the generated
kernel's output matches `in + 1`. The milestone: **generated m_axi kernel works**.

**Commit:** `incr: hand-written TB + build DAG — generated m_axi kernel passes C-sim`

---

## Phase 5: Testbench codegen

**Goal:** generate `incr_tb.cpp` from `IncrTBHls`, replacing the hand-written one;
diff against the Phase-4 file.

**Changes:** extend the testbench emitter (`tb_files_to_str` and friends in
[hwgen.py](../pysilicon/build/hwgen.py)) + the TB statement IR with the four new
constructs (decision 9): `MemComponent` decl → array+`MemMgr`; `alloc_array` →
`mgr.alloc` + byte-addr + `write_array`; `read_array` → `read_array`;
`KernelCallStmt` includes the `mem` pointer in canonical order. Recognize these
in the TB extractor.

**Tests:** generate `gen/incr_tb.cpp`; assert it is functionally equivalent to
the Phase-4 hand-written TB (diff/structure check), and re-run C-sim
(vitis-gated) with the *generated* TB — still matches the Python model.

**Commit:** `hwgen: m_axi testbench codegen — MemComponent/alloc/read-back/kernel-call`

---

## Phase 6 (stretch): C-synth + cosim + burst validation

**Goal:** confirm the generated kernel synthesizes and that its RTL m_axi bursts
match expectation — the real stress test for m_axi.

**Changes:** add `CSynthStep`, cosim, and reuse the histogram's
VCD/`extract_aximm_bursts` machinery ([hist_demo.py](../examples/histogram/hist_demo.py))
to assert the increment kernel issues the expected read/write bursts. Optionally
feed cosim cycle counts into `ValidateTimingStep` against the Python prediction
(the cycle-model loop).

**Tests (vitis-gated):** cosim passes; burst-extraction report validates.

**Commit:** `incr: cosim + AXI-MM burst validation for the generated kernel`

---

## Future / out of scope (capture, don't build)

- **Queue codegen unblocked.** Once kernel + TB m_axi codegen exist, the AXI-MM
  queue's ring-buffer `write`/`get` can be generated on top — see
  [plans/aximm_queue.md](aximm_queue.md). That was always the prerequisite.
- **`s_axilite`/offset control variant** — a kernel whose base address + count
  come from control registers instead of a stream command. Needs the regmap
  codegen to emit `offset=` on the m_axi pragma.
- **Richer memory codegen** — multiple `m_axi` bundles, separate read/write
  ports, scatter/gather, burst-length pragmas.
- **Promote the toy's lessons into the histogram** — once codegen is proven on
  the toy, regenerate the (currently hand-written) `hist.cpp`/`hist_tb.cpp` from
  a `HistAccel` `HwComponent` to close the loop on a real example.
