# A Proper SimPy Memory Object (latency-modeling `MemComponent`)

## Goal

Turn the existing memory model into a real SimPy participant: a
latency-modeling `SimObj` that serves AXI-MM traffic through its `MMIFSlave`,
registered in the `Simulation` lifecycle, and proven end-to-end by a small
read-modify-write example. This is the foundation that the AXI-MM queue (and
later a transactional histogram simulation) should be built on — instead of the
throwaway `MMMemory` that was created for the queue.

Concretely:

- Promote `MemComponent` ([pysilicon/hw/memory.py](../pysilicon/hw/memory.py))
  from a plain `@dataclass` to a `SimObj` with a **simple parameterized latency
  model** (fixed init + per-word) on its AXI-MM slave path.
- Add a minimal standalone example exercising it over a real interconnect.
- Add the first tests that drive `Memory` through an actual `Simulation`
  (`run_sim()`), not just direct method calls.

## Scope

**In scope:** SimObj integration, simple parameterized latency, a toy example,
and tests inside a `Simulation`.

**Out of scope (captured in "Future"):** the transactional histogram sim, richer
DRAM-like timing, HLS codegen, and resuming the AXI-MM queue.

## Background — what exists today (verified, do NOT rebuild)

Read before starting:

- `Memory` ([memory.py:15](../pysilicon/hw/memory.py#L15)) — a solid sparse
  reference store: `AddrUnit.byte`/`word`, `alloc`/`free`/`read`/`write`,
  wide-word (>64b) support. Unit-tested in
  [tests/hw/test_memory.py](../tests/hw/test_memory.py) (direct calls, **no
  Simulation**). **Leave this class as-is** — it is the backing store.
- `MemComponent` ([memory.py:395](../pysilicon/hw/memory.py#L395)) — already
  wraps a `Memory` and exposes:
  - `s_mm`: an `MMIFSlave` with `_on_write`/`_on_read` callbacks
    ([memory.py:450-457](../pysilicon/hw/memory.py#L450-L457)) — **currently
    zero-latency stubs** (`yield` nothing).
  - `m_mm`: a `_DirectBackedMMIFMaster` (zero-latency, for the owner's inline
    use) with `as_words`/`as_array`/`as_schema` zero-copy views.
  - `inline=True` (pre-allocate the whole capacity, local-BRAM style) vs
    `inline=False` (external DDR; callers `alloc`/`free`).
  - It is a plain `@dataclass`, **not a `SimObj`**, and is **instantiated
    nowhere** — so we can change it freely.
- `SimObj` ([simulation/simobj.py:36](../pysilicon/simulation/simobj.py#L36)) —
  dataclass with `sim`; `__post_init__` calls `self.sim.add_obj(self)`; exposes
  `timeout(delay)`, `now`, `env`, and `pre_sim`/`run_proc`/`post_sim`.
- `AXIMMCrossBarIF` / `DirectMMIF`
  ([memif.py](../pysilicon/hw/memif.py)) — the interconnects. **They already
  model bus/wire latency** (`latency_init`, `latency_read_return`) and then
  invoke the slave's `rx_*_proc`. See decision 2 — the memory's latency is
  *complementary* to this, not a duplicate.
- Reference pattern: `MemBank` / `CPU` in
  [examples/interface/aximm_demo.py](../examples/interface/aximm_demo.py) — a
  `SimObj` slave with a latency-bearing `rx_read`, and a master driving it.

## Design decisions (settled — do NOT re-litigate)

1. **Promote `MemComponent` to `SimObj`; keep `Memory` untouched.** `Memory`
   stays a pure reference store. `MemComponent(SimObj)` is the SimPy wrapper
   that owns a `Memory`, registers with the `Simulation`, and adds timing. Do
   not fork a new class — `MemComponent` is unused, so evolve it in place.

2. **The memory models *access* latency; the interconnect models *bus*
   latency.** They compose, they do not double-count. The crossbar/`DirectMMIF`
   delay represents wire/arbitration; the memory's `_on_read`/`_on_write` delay
   represents the RAM access (CAS/BRAM read). A read's total time is
   `bus_request + memory_access + bus_return`, exactly as the crossbar's
   docstring already lays out (it budgets "slave `rx_read_proc` duration"
   separately).

3. **Simple parameterized latency, in cycles, mirroring the crossbar's knobs.**
   `MemComponent` takes a `clk: Clock` and two parameters:
   `latency_init` (fixed cycles per access) and `latency_per_word` (cycles per
   word). Each access yields
   `self.timeout((latency_init + nwords * latency_per_word) / clk.freq)`.
   Defaults are `0.0` (so a zero-latency memory is still expressible). No
   separate read/write/return knobs in this version — that's "Future".

4. **The `m_mm` direct master stays zero-latency.** It models the owner reading
   its *own* inline block (maps to a local C array in HLS), so no bus/access
   delay applies. The latency model is **only** on the `s_mm` (AXI-MM slave)
   path. Document this asymmetry.

5. **Latency is applied inside the slave callback, before the data op.** Order:
   `yield self.timeout(delay)` then `self._mem.read/write(...)`. The interconnect
   already wraps the callback in `env.process(...)` and waits for it, so the
   caller observes `bus + access` time. (Matches `MemBank.rx_read`.)

6. **`addr_unit` is exposed but defaults to byte.** `Memory` already supports
   word-addressing; surface it on `MemComponent` (default `AddrUnit.byte`, the
   AXI-MM convention) so a future BRAM/direct use can flip it. No new logic —
   just pass-through.

7. **The toy example uses `inline=False` + `alloc`.** The driver allocates a
   region, writes, reads, modifies, writes back, reads, and verifies — the
   smallest thing that exercises the slave path *and* the latency model. A
   second optional driver over a crossbar demonstrates serialization, but the
   single-driver `DirectMMIF` path is the required deliverable.

8. **This replaces `MMMemory` when the queue resumes.** The aximm-queue work is
   paused, not deleted. When it resumes, its `MMMemory` is dropped in favor of
   `MemComponent`; note this in "Future", don't touch the queue branch here.

## Working convention

- One commit per phase, in order; push after each. Single PR, multiple commits.
- After every phase: `pytest tests/hw/ tests/examples/ -k "not vitis"` — keep
  green (note, don't fix, pre-existing unrelated failures, e.g. the
  `test_dataschema_poly.py` missing-`poly.hpp` failure).
- The example in Phase 2 must run standalone:
  `python -m examples.memory.mem_demo` (or wherever it lands) and self-check.
- If making `MemComponent` a `SimObj` surfaces a field-ordering or
  double-registration issue with its `s_mm`/`m_mm` endpoints (they are also
  sim-registered), STOP and ask rather than hacking around it.

---

## Phase 1: `MemComponent` becomes a latency-modeling `SimObj`

**Goal:** `MemComponent` registers with the `Simulation`, carries a clock +
latency params, and its AXI-MM slave path consumes the modeled access time.

**Changes:** in [pysilicon/hw/memory.py](../pysilicon/hw/memory.py):

- Make `MemComponent` inherit `SimObj` (it already has `name`/`sim`; let the
  base provide them and reorder dataclass fields so base fields come first).
- Add fields:

  ```python
  clk: Clock = field(default_factory=lambda: Clock(freq=1.0))
  latency_init: float = 0.0       # fixed cycles per access
  latency_per_word: float = 0.0   # cycles per word
  addr_unit: AddrUnit = AddrUnit.byte
  ```

- Update the slave callbacks to model access latency (decision 5):

  ```python
  def _access_delay(self, nwords: int) -> float:
      return (self.latency_init + nwords * self.latency_per_word) / self.clk.freq

  def _on_write(self, words, local_addr):
      yield self.timeout(self._access_delay(len(words)))
      self._mem.write(local_addr, words)

  def _on_read(self, nwords, local_addr):
      yield self.timeout(self._access_delay(nwords))
      return self._mem.read(local_addr, nwords)
  ```

- Keep `_DirectBackedMMIFMaster` (`m_mm`) zero-latency (decision 4). Keep
  `inline`/`alloc`/`free` behavior. Pass `addr_unit` through to the `Memory`
  ctor.
- Ensure `__post_init__` still builds `s_mm`/`m_mm` and that calling
  `super().__post_init__()` (which runs `sim.add_obj`) happens correctly.

**Tests:** add to [tests/hw/test_memory.py](../tests/hw/test_memory.py) (or a new
`test_memory_simobj.py`) — the **first tests that use a real `Simulation`**:

- Driver `SimObj` with an `MMIFMaster` → `DirectMMIF` → `MemComponent.s_mm`,
  inside one `Simulation`; `run_sim()`. Write an array, read it back: assert
  data round-trips.
- **Latency**: with `latency_init=L0`, `latency_per_word=Lw`, and a known bus
  latency on `DirectMMIF`, assert elapsed `sim` time for a read/write of N words
  equals the expected `bus + (L0 + N*Lw)/freq`. This is the test that proves the
  access-vs-bus composition (decision 2).
- `latency_init=latency_per_word=0` still works (zero-time, data correct).
- Schema/array path: `write_array`/`read_array` through the slave round-trips a
  typed batch.

**Commit:** `memory: MemComponent is a latency-modeling SimObj on the AXI-MM slave path`

---

## Phase 2: Toy read-modify-write example

**Goal:** a minimal, standalone, self-checking demo that drives the memory
SimObj over AXI-MM — the smallest end-to-end proof.

**Changes:** new `examples/memory/mem_demo.py` (create the dir):

- A `MemDriver(SimObj)` holding an `MMIFMaster`. Its `run_proc`:
  1. allocate a region (`MemComponent.alloc`, `inline=False`),
  2. `write` a known array (use `write_array` for typed access),
  3. `read` it back and assert equality,
  4. modify (e.g. `+1`), write back,
  5. read again and assert, printing transfer times (showing latency).
- Harness wires `MemDriver` + `MemComponent` over a `DirectMMIF` with a `Clock`,
  runs the `Simulation`, and self-checks. Module docstring carries an ASCII
  topology diagram (like `aximm_demo.py`). `run_and_check()` + `__main__`.
- *(Optional within this file)* a two-driver `AXIMMCrossBarIF` variant showing
  serialized access to one memory, with a timing printout.

**Tests:** `tests/examples/test_mem_demo.py` — import the harness, run it, assert
it passes (the pattern other example tests use).

**Commit:** `examples: mem_demo — read-modify-write over AXI-MM to a memory SimObj`

---

## Phase 3 (optional): Docs

**Goal:** document the SimObj + latency model; today
[docs/guide/memory/](../docs/guide/memory/) only describes the pure-Python
reference path.

**Changes:** add a short page/section covering: `MemComponent` as a `SimObj`,
the `clk`/`latency_init`/`latency_per_word` knobs, the access-vs-bus latency
distinction (decision 2), the `m_mm` vs `s_mm` asymmetry, and a pointer to
`mem_demo`. Update the memory section index/nav to match its conventions.

**Commit:** `docs: document the memory SimObj + latency model`

---

## Future / out of scope (capture, don't build)

- **Transactional histogram simulation** — the originally-discussed validator:
  a controller `SimObj` streaming `HistCmd` over AXI-Stream to a histogram-accel
  `SimObj` that reads/writes data via AXI-MM to this memory SimObj and streams
  back `HistResp`. The existing `HistogramAccel` numpy logic + `Memory` +
  `hist_demo` Vitis flow ([examples/histogram/hist_demo.py](../examples/histogram/hist_demo.py))
  are the golden reference. Its own plan once this foundation is solid.
- **Resume the AXI-MM queue** — replace the throwaway `MMMemory` in
  [pysilicon/hw/aximm_queue.py](../pysilicon/hw/aximm_queue.py) with
  `MemComponent`; switch the queue's tests/example to the real memory SimObj.
  See [plans/aximm_queue.md](aximm_queue.md). Nothing on the `aximm-queue`
  branch is deleted by this plan.
- **Richer timing** — separate read/write latencies, per-burst overhead,
  bandwidth/outstanding-transaction caps, bank contention. Deferred (decision 3).
- **HLS codegen** for `s_mm`/`m_mm` — blocked on AXI-MM codegen existing at all.
- **Word-addressed / BRAM** — `Memory` + `addr_unit` already support it
  (decision 6); a direct-BRAM mapping is a later use of the same knob.
