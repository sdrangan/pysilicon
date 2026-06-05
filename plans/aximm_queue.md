# AXI-MM Queue: Memory-Backed Ring Buffer over AXI-MM

## Goal

Add an `AXIMMQueue` — a producer/consumer FIFO whose storage lives in an AXI-MM
memory region — built entirely on the existing `MMIFMaster` / `MMIFSlave`
primitives in [pysilicon/hw/memif.py](../pysilicon/hw/memif.py). This is the
classic CPU↔accelerator software queue / DMA descriptor ring: a circular buffer
in shared memory with head/tail pointers, enqueue/dequeue (named `write` / `get`
to match the stream queue), full/empty detection, and wrap-around.

The deliverable is:

- A new module `pysilicon/hw/aximm_queue.py` containing `AXIMMQueueLayout`
  (memory map) and `AXIMMQueue` (the proxy a master uses to `write`/`get`).
- A small reusable memory slave so tests and examples don't each redefine the
  demo's local `MemBank`.
- A runnable example `examples/interface/aximm_queue_demo.py`.
- Unit tests in `tests/hw/test_aximm_queue.py`.

The queue is a **simulation-level abstraction layered on the existing MM
transaction API** (`MMIFMaster.write` / `.read`). It works over *any* MM
interconnect — `AXIMMCrossBarIF` or `DirectMMIF` — because the ring lives in
ordinary MM memory and the pointers are ordinary MM words.

## Scope

**In scope:** simulation model, single-producer-single-consumer (SPSC)
semantics, raw-word and schema-typed elements, a worked example, unit tests,
and a short doc page.

**Out of scope (explicitly):**

- HLS codegen for the queue. AXI-MM has *no* codegen path today (only
  AXI-Stream and AXI-Lite do), so generating `m_axi` ring-buffer code is a
  separate, larger effort. This plan does not touch `pysilicon/build/`.
- Multi-producer / multi-consumer (MPMC). MPSC/MPMC needs a lock register or
  atomic pointer update, which AXI-MM in this repo does not model. Noted in
  "Future" below.

## RESUME STATUS (read first)

Phases 1–3 are **done and committed** on the `aximm-queue` branch (core
`AXIMMQueueLayout` + `try_write`/`try_get` + blocking `write`/`get` + concurrent
SPSC tests). Work then paused while the memory foundation was built. Two things
changed during the pause; both are folded into **Phase 3.5 below, which runs
before Phase 4**:

1. **The memory foundation now exists.** [plans/memory_simobj.md](memory_simobj.md)
   promoted `MemComponent` ([pysilicon/hw/memory.py](../pysilicon/hw/memory.py))
   into a real latency-modeling `SimObj` exposing an `MMIFSlave` (`s_mm`). The
   throwaway `MMMemory` created in queue Phase 1 is now redundant — **replace it
   with `MemComponent`** (decision 9, updated).
2. **`write` becomes streaming/chunking, not all-or-nothing** (decision 8,
   updated). The committed Phase 3 `write` raised when a batch exceeded usable
   depth; it must instead feed the batch through in pieces, blocking as the
   consumer drains, mirroring `get`.

Also fix the latent `get(nslots=0)` crash (returns `out[0]` on an empty list).

So the remaining sequence is: **Phase 3.5 (migrate) → Phase 4 → Phase 6**
(Phase 5 caching and Phase 7 docs remain optional).

## Background — what already exists (do NOT rebuild)

Read these before starting:

- [pysilicon/hw/memif.py](../pysilicon/hw/memif.py):
  - `MMIFMaster.write(words, global_addr)` and `.read(nwords, global_addr)` —
    the raw word-transfer generators every queue operation builds on.
  - `MMIFMaster.write_schema/read_schema/write_array/read_array` — schema-typed
    convenience methods reused in Phase 4.
  - `MMIFSlave(rx_write_proc=…, rx_read_proc=…)` — the storage side. The
    callbacks are `(words, local_addr)` and `(nwords, local_addr) -> Words`.
  - `AXIMMCrossBarIF` (multi-master, address-routed) and `DirectMMIF`
    (point-to-point). `assign_address_ranges(slaves, [(base, size), …])` sets
    each slave's `addr_range`.
  - `byte_addressable=True` (default): a 32-bit word spans 4 byte-addresses, so
    word *i* is at `base + i*4`. This plan follows that convention.
- [examples/interface/aximm_demo.py](../examples/interface/aximm_demo.py):
  - `MemBank` (FULL, dict-backed RAM), `RegFile` (LITE, per-word registers),
    `CPU` / `DMA` masters. The producer/consumer/memory wiring pattern the
    example in Phase 6 follows. **`MemBank` is local to that file** — Phase 1
    promotes a reusable version.

## Design decisions (settled — do NOT re-litigate)

1. **Pointers live in MM, not in Python.** Head and tail are stored as words in
   the memory region itself. This is what makes it a *memory-mapped* queue: two
   independent masters (producer on `master_0`, consumer on `master_1`) share
   state purely through memory, with no shared Python object. (Python-side
   pointer caching is added as an optimization in Phase 5 — but the source of
   truth is always MM.)

2. **SPSC, lock-free.** Exactly one producer (writes `tail` + data slots, reads
   `head`) and one consumer (writes `head` + reads data slots, reads `tail`).
   No pointer is written by both sides, so no lock is needed. This is the only
   regime where the design is correct without atomics; enforce it by contract
   (documented, not policed at runtime).

3. **Word-index pointers; reserve one slot to disambiguate full vs empty.**
   `head` and `tail` are slot indices in `[0, capacity)`. `empty ⇔ head == tail`;
   `full ⇔ (tail + 1) % capacity == head`. Usable depth is therefore
   `capacity - 1`. This is the standard single-reserved-slot ring; it avoids a
   separate count register that both sides would write.

4. **Layout is contiguous and parameterized by memory width — no hard-coded
   strides.** The region is sized in terms of a memory data width `mem_bw`
   (bits). One control field occupies exactly **one memory word**; control words
   precede the data slots:

   ```
   word 0 : head      (consumer-owned slot index)        ── control words
   word 1 : tail      (producer-owned slot index)          (NUM_CONTROL_WORDS)
   word 2 : capacity  (number of slots; informational)
   word 3 : reserved  (0; room for status/flags later)
   data   : capacity slots × elem_words words            ── ring storage
   ```

   Every byte offset is **computed** from `mem_bw`, never a literal:
   `word_bytes = mem_bw // 8` and `control_bytes = NUM_CONTROL_WORDS * word_bytes`.
   There is no module-level `CONTROL_BYTES = 16` constant — `0x04`/`0x0C`/`16`
   only happen to be the values when `mem_bw == 32`. `elem_words` (words per
   slot) defaults to 1; Phase 4 generalizes it.

   **Addressing is byte-addressed, full stop.** AXI-MM is always byte-addressed
   (DDR/Pynq style), so the queue assumes it — there is no `byte_addressable`
   knob and no word-addressed path. Word-addressed memory (FPGA BRAM on a direct
   interface) is a deliberate non-goal here; see "Future" for the generalization
   path via `memory.py`'s `AddrUnit`.

5. **`AXIMMQueueLayout` owns all address math; nothing is hard-coded.** Given
   `mem_bw`, `capacity`, and `elem_words`, it derives `word_bytes`,
   `control_bytes`, `head_addr`, `tail_addr`, `capacity_addr`, `slot_addr(i)`,
   and `total_bytes` (for `assign_address_ranges`). `NUM_CONTROL_WORDS` is the
   only structural constant (a count of control fields, not a byte size). The
   `AXIMMQueue` proxy never computes raw offsets inline.

6. **Terminology matches the stream queue: `write` (enqueue) / `get` (dequeue).**
   Not `push`/`pop`, and not `read`/`write`. This mirrors `StreamIF.write` /
   `StreamIFSlave.get` and `QueuedTransferIFMaster.write` /
   `QueuedTransferIFSlave.get` ([pysilicon/hw/interface.py](../pysilicon/hw/interface.py)).
   In particular the typed dequeue reuses the **exact** stream signature
   `get(schema_type=None, count=None, *, nwords_max=None)`. `read`/`write` is
   rejected because `read` collides with `MMIFMaster.read`, the *memory*
   transport the queue calls internally — keeping "queue verbs" (`write`/`get`)
   distinct from "transport verbs" (`MMIFMaster.write`/`read`) is deliberate,
   exactly as `StreamIF.write`/`get` sit above `_push_to_endpoint`. Bonus: a
   future MM-queue codegen mirrors the stream side's `StreamGetStmt` naming.

7. **One `AXIMMQueue` class, used by both sides.** It wraps an `MMIFMaster` plus
   an `AXIMMQueueLayout`. The producer calls `write*`; the consumer calls `get*`.
   Two instances (one per master endpoint) over the same layout/base form the
   full SPSC queue. No separate `Producer`/`Consumer` subclasses — they'd only
   differ by which methods you call.

8. **Blocking `write`/`get` are the public API; non-blocking `try_*` are the
   primitives.** To match the stream queue's "yield until it completes" feel,
   `write(words)` blocks until all words are enqueued and `get(...)` blocks until
   the requested data is available, each built on a `try_write` / `try_get`
   primitive with a caller-supplied `poll_interval`. `try_write` returns a bool;
   `try_get` returns a possibly-short array. Blocking calls must be cancellable
   by simulation end like any other SimObj process.

   **`write` streams/chunks (updated):** it must accept an array of *any* size —
   feeding it through in pieces (each ≤ usable depth), blocking as the consumer
   drains, until the whole array is enqueued. It must **not** raise when the
   batch exceeds `capacity - 1`. This mirrors `get`, which already collects
   across multiple `try_get` calls. The atomic single-shot enqueue stays
   available as `try_write`. (Supersedes the committed Phase 3 behavior, which
   raised; Phase 3.5 reworks it.)

9. **Use `MemComponent` as the backing store (updated).** Queue Phase 1 added a
   throwaway `MMMemory` slave; the real latency-modeling `MemComponent`
   ([pysilicon/hw/memory.py](../pysilicon/hw/memory.py)) now exists, so Phase 3.5
   **deletes `MMMemory` and switches tests/example to `MemComponent`** (bind its
   `s_mm`). The queue proxy is storage-agnostic — it only issues `MMIFMaster`
   transactions — so it works over any MM slave; `MemComponent` is now the
   canonical one (and brings a real `alloc`/`free` + access-latency model for
   free).

10. **Wrap-around splits into at most two transfers.** A `write`/`get` of `n`
    slots starting at index `p` where `p + n > capacity` issues two MM transfers:
    `[p, capacity)` then `[0, n - (capacity - p))`. Never a modular per-element
    loop — keep bursts intact for FULL-protocol latency.

11. **`mem_bw` is the layout's memory width and must match the interconnect.**
    The layout's `mem_bw` (default 32) is the single source of word width;
    `word_bytes` derives from it (decision 4). It must equal the bound
    `MMIFMaster`/interconnect `bitwidth` — validate this when the `AXIMMQueue`
    is constructed (or on first transaction) and raise a clear error on
    mismatch, rather than silently producing wrong addresses.

## Working convention

- One commit per phase, in order. Push after each.
- Run `pytest tests/hw/ tests/examples/ -k "not vitis"` after every phase; keep
  green (ignore pre-existing failures unrelated to this work — note them if you
  see them, don't fix them here).
- The example in Phase 6 is the visible deliverable; it must run standalone
  (`python -m examples.interface.aximm_queue_demo`) and self-check.
- If any `MMIFMaster`/`MMIFSlave`/crossbar behavior contradicts an assumption
  here (e.g. address decode, wrap, latency), STOP and ask — that's a real
  interconnect issue, not something to paper over.

---

## Phase 1: Module scaffold, `AXIMMQueueLayout`, reusable `MMMemory`

**Goal:** Create the module and the address-math layer, plus a reusable memory
slave. No queue logic yet.

**Changes:**

- New file `pysilicon/hw/aximm_queue.py`. Add the layout:

  ```python
  from __future__ import annotations
  from dataclasses import dataclass

  @dataclass(frozen=True)
  class AXIMMQueueLayout:
      """Memory map for a ring buffer in an AXI-MM region.

      Every offset is derived from `mem_bw`; one control field occupies one
      memory word.  Pointers (head/tail) are slot indices in [0, capacity).
      Usable depth is capacity - 1 (one slot reserved to distinguish full from
      empty).
      """
      # One control field per memory word: head, tail, capacity, reserved.
      NUM_CONTROL_WORDS: ClassVar[int] = 4

      base_addr: int
      capacity: int            # number of slots
      elem_words: int = 1      # words per slot
      mem_bw: int = 32         # memory data width in bits (AXI-MM: byte-addressed)

      @property
      def word_bytes(self) -> int:
          # AXI-MM is byte-addressed: a word spans mem_bw//8 byte addresses
          return self.mem_bw // 8

      @property
      def control_bytes(self) -> int:
          return self.NUM_CONTROL_WORDS * self.word_bytes

      @property
      def head_addr(self) -> int: return self.base_addr + 0 * self.word_bytes
      @property
      def tail_addr(self) -> int: return self.base_addr + 1 * self.word_bytes
      @property
      def capacity_addr(self) -> int: return self.base_addr + 2 * self.word_bytes
      @property
      def data_base(self) -> int: return self.base_addr + self.control_bytes
      def slot_addr(self, idx: int) -> int:
          return self.data_base + idx * self.elem_words * self.word_bytes
      @property
      def total_bytes(self) -> int:
          return self.control_bytes + self.capacity * self.elem_words * self.word_bytes
  ```

  Validate in `__post_init__` (frozen, so no mutation needed — just raise):
  `capacity >= 2`, `elem_words >= 1`, `mem_bw in (8, 16, 32, 64)`. (Add the
  `from typing import ClassVar` import.)

- Add a reusable memory slave to the same module (or `memif.py` — decide and
  note it; `aximm_queue.py` keeps the dependency local):

  ```python
  @dataclass
  class MMMemory(SimObj):
      """Word-addressed RAM backing an MM slave (generalized aximm_demo MemBank)."""
      bitwidth: int = 32       # must match the queue layout's mem_bw
      access_latency: float = 0.0
      def __post_init__(self):
          super().__post_init__()
          self._mem: dict[int, int] = {}
          self.slave_ep = MMIFSlave(sim=self.sim, bitwidth=self.bitwidth,
                                    rx_write_proc=self.rx_write,
                                    rx_read_proc=self.rx_read)
      # rx_write/rx_read mirror aximm_demo.MemBank, keyed by local_addr; the
      # address stride per word is bitwidth//8, NOT a hard-coded 4.
  ```

**Tests:** `tests/hw/test_aximm_queue.py`

- `AXIMMQueueLayout` math: `word_bytes`/`control_bytes`/`head_addr`/`tail_addr`/
  `capacity_addr`/`data_base`, `slot_addr(0)`, `slot_addr(capacity-1)`,
  `total_bytes` — across `mem_bw ∈ {32, 64}` and `elem_words ∈ {1, 4}`.
  Explicitly assert that `mem_bw=64` doubles the control-region size and word
  stride (the regression the old hard-coded `0x04`/`CONTROL_BYTES=16` would have
  failed).
- Wrap helper if you factor one out (see Phase 2) — otherwise defer.
- `MMMemory` round-trip: write a burst via a `DirectMMIF`, read it back.

**Commit:** `aximm_queue: AXIMMQueueLayout address map + reusable MMMemory slave`

---

## Phase 2: Core proxy — `reset`, `space`/`count`, non-blocking `try_write`/`try_get`

**Goal:** The working SPSC ring for `elem_words == 1`, raw words. The
non-blocking `try_*` primitives that Phase 3's blocking `write`/`get` build on.

**Changes:** in `pysilicon/hw/aximm_queue.py`:

```python
@dataclass
class AXIMMQueue:
    master: MMIFMaster
    layout: AXIMMQueueLayout

    def reset(self) -> ProcessGen[None]:
        """Zero head and tail (call once, by whichever side owns setup)."""
        yield from self.master.write(np.zeros(2, np.uint32), self.layout.head_addr)
        # optionally write capacity word

    def _read_ptrs(self) -> ProcessGen[tuple[int, int]]:
        w = yield from self.master.read(2, self.layout.head_addr)
        return int(w[0]), int(w[1])

    def count(self) -> ProcessGen[int]:   # occupied slots
        head, tail = yield from self._read_ptrs()
        return (tail - head) % self.layout.capacity

    def space(self) -> ProcessGen[int]:   # free slots (usable = capacity-1)
        c = yield from self.count()
        return (self.layout.capacity - 1) - c

    def try_write(self, words) -> ProcessGen[bool]:
        """Try to enqueue len(words) slots. Returns False (no-op) if it won't fit."""
        # 1. read head (to compute free space), read own tail
        # 2. if nslots > free: return False
        # 3. write data at slot_addr(tail), splitting on wrap (decision 10)
        # 4. write new tail = (tail + nslots) % capacity
        ...

    def try_get(self, max_slots) -> ProcessGen[Words]:
        """Dequeue up to max_slots; returns the words actually read (may be short/empty)."""
        # mirror of try_write: read tail, read own head, read data with wrap, advance head
        ...
```

- Note: `try_write`/`try_get` are the non-blocking primitives. The public
  blocking `write`/`get` (decision 8) arrive in Phase 3.
- Factor the wrap split into a private helper used by both, e.g.
  `_split(idx, nslots) -> list[(idx, count)]` returning one or two runs.
- `try_write` writes data **before** advancing `tail` (consumer must never see a
  tail that points past unwritten data). `try_get` reads data **before**
  advancing `head` (producer must never reclaim a slot still being read). This
  ordering is the correctness crux of decision 2 — comment it.

**Tests:**

- Single `AXIMMQueue` over a `DirectMMIF`+`MMMemory`, exercising `try_write` then
  `try_get` sequentially from one process:
  - FIFO order preserved across many enqueue/dequeue cycles.
  - Fill to `capacity-1`, assert next `try_write` returns False (full).
  - Drain to empty, assert `try_get` returns empty, `count()==0`.
  - **Wrap-around**: sizes that force `tail`/`head` past the end (e.g. capacity
    8, repeatedly `try_write` 5 / `try_get` 5) — verify data integrity.
  - `space()`/`count()` track correctly throughout.

**Commit:** `aximm_queue: AXIMMQueue core try_write/try_get with wrap-around (SPSC)`

---

## Phase 3: Public blocking `write`/`get` + concurrent producer/consumer

**Goal:** The public `write` / `get` API (decision 8), validated with a *real*
concurrent producer and consumer (two masters, one shared region) on a crossbar.
These are the stream-consistent names; they block by polling the Phase 2
`try_*` primitives.

**Changes:**

```python
def write(self, words, poll_interval) -> ProcessGen[None]:
    while not (yield from self.try_write(words)):
        yield self.master.timeout(poll_interval)

def get(self, nslots, poll_interval) -> ProcessGen[Words]:
    out = []
    while len(out) < nslots:
        chunk = yield from self.try_get(nslots - len(out))
        if len(chunk) == 0:
            yield self.master.timeout(poll_interval)
        else:
            out.extend(chunk)
    return np.array(out, ...)
```

- Confirm `MMIFMaster` (an `InterfaceEndpoint`) exposes `timeout` / `process` /
  `now` the same way SimObjs do; if the poll/yield must go through the owning
  SimObj instead, route it that way and note it. (Check `InterfaceEndpoint` in
  [pysilicon/hw/interface.py](../pysilicon/hw/interface.py).) **Resolve this
  before writing Phase 3 code** — it dictates how the poll loop is spelled.

**Tests:**

- Crossbar with `nports_master=2`, one `MMMemory` slave (FULL) holding the
  region; `assign_address_ranges` from `layout.total_bytes`.
- `Producer` SimObj: `write` N words (e.g. 100), in chunks, into a queue of
  small capacity (e.g. 8) so it genuinely blocks on full.
- `Consumer` SimObj: `get` all N, collect.
- Assert consumer received exactly the producer's sequence, in order, no loss,
  no duplication. This is the test that proves the SPSC ordering (decision 2).

**Commit:** `aximm_queue: blocking write/get + concurrent SPSC crossbar test`

---

## Phase 3.5: Migrate to `MemComponent` + stream `write` + `get(0)` fix

**Goal:** Fold in the two changes from the pause (decisions 8 & 9) plus the
minor crash, on top of the committed Phases 1–3, before adding the typed layer.

**Changes:** in [pysilicon/hw/aximm_queue.py](../pysilicon/hw/aximm_queue.py):

- **Delete `MMMemory`.** Switch all tests/wiring to `MemComponent` (bind its
  `s_mm`). Where a test used `MMMemory(sim, bitwidth=...)`, use
  `MemComponent(sim=..., clk=..., word_size=..., inline=False)` and `alloc` the
  ring region (or `inline=True` and use its base) — pick whichever keeps the
  tests simplest, but exercise the real slave. Update imports in
  [tests/hw/test_aximm_queue.py](../tests/hw/test_aximm_queue.py).
- **Rework `write` to stream/chunk** (decision 8). Replace the
  raise-if-too-large body with a loop that enqueues the array in pieces of at
  most `capacity - 1` slots, polling `try_write` and `timeout(poll_interval)`
  when full, until the whole array is in. No size guard / no `ValueError`.

  ```python
  def write(self, words, poll_interval=DEFAULT_POLL_INTERVAL):
      ew = self.layout.elem_words
      chunk = (self.layout.capacity - 1) * ew      # max words per attempt
      off = 0
      while off < len(words):
          piece = words[off:off + chunk]
          if (yield from self.try_write(piece)):
              off += len(piece)
          else:
              yield self.master.timeout(poll_interval)
  ```

- **Fix `get(nslots=0)`**: return an empty array instead of indexing `out[0]`
  on an empty list.

**Tests:**

- Update existing tests to `MemComponent`; keep all Phase 1–3 assertions green.
- New: `write` of an array **larger than usable depth** (e.g. 1000 words into a
  capacity-8 queue) with a concurrent consumer drains completely, in order — the
  case the old code raised on. Reuse the concurrent SPSC harness.
- `get(0)` returns an empty array (no crash).
- Remove the now-obsolete `test_write_too_large_raises`.

**Commit:** `aximm_queue: back queue with MemComponent; stream write; fix get(0)`

---

## Phase 4: Multi-word slots + typed `write`/`get` (stream signature)

**Goal:** `elem_words > 1`, and fold typed access into `write`/`get` so the
dequeue side matches the stream queue's signature **exactly** rather than adding
separate `*_array` methods (decision 6).

**Changes:**

- Generalize the `try_*` core for `elem_words > 1`: a slot is `elem_words`
  contiguous words; slot counts drive the wrap split; transfer addresses use
  `slot_addr`; word counts drive the actual MM transfers.
- Extend `get` to the stream signature instead of a bespoke one:

  ```python
  def get(self, schema_type=None, count=None, *, nwords_max=None,
          poll_interval=...) -> ProcessGen[Words | Any]:
  ```

  - `schema_type=None` → raw words (the Phase 3 behavior; `count`/`nwords_max`
    select how many words/slots).
  - `schema_type` given → set `elem_words = schema_type.nwords_per_inst(word_bw)`
    (validate it matches the layout), dequeue, and deserialize — `count` returns
    a `DataArray` of `count` elements, mirroring `StreamIFSlave.get`. Reuse
    `MMIFMaster.read_array`/`read_schema` for the deserialization step; the queue
    only owns the slot addressing + wrap.
- Extend `write` symmetrically: `write(data, element_type=None, *, poll_interval=...)`
  — raw words when `element_type=None`, else serialize via
  `MMIFMaster.write_array`/`write_schema` semantics before enqueue.
- Keep the raw-word path byte-for-byte compatible with Phase 3.

**Tests:**

- `elem_words=4` raw-word ring: multi-word slots, wrap-around, FIFO.
- Typed round-trip: define or reuse a small `DataSchema` (e.g. a 2-field
  struct), `write` a batch, `get(SchemaType, count=N)` it back, assert
  field-equality — and assert the call shape matches `StreamIFSlave.get`.
- Validation: typed access against a layout whose `elem_words` disagrees with
  the schema raises a clear error.

**Commit:** `aximm_queue: multi-word slots + typed write/get (stream signature)`

---

## Phase 5 (optional): Local pointer caching optimization

**Goal:** Cut pointer round-trips. Each `write` currently does read(head) +
write(data) + write(tail); the producer can cache `head` and only re-read it
when its cached copy says "full". Symmetric for the consumer's cached `tail`.

**Changes:**

- Add `cache_peer_ptr: bool = False` to `AXIMMQueue`. When set, the producer
  caches the last-seen `head`; `write` only re-reads `head` if the cached value
  indicates no space (the peer may have advanced it). Consumer mirrors with
  `tail`. The owned pointer is always exact (we wrote it).
- Source of truth stays in MM (decision 1) — caching only skips *reads known to
  be conservatively stale-safe*: a cached-stale `head` can only *under*-report
  free space, never over-report, so correctness holds.

**Tests:**

- Re-run Phase 3 concurrent test with `cache_peer_ptr=True`; assert identical
  data result, and assert fewer MM reads occur (instrument the `MemComponent`
  backing store with a read counter, or compare sim end-time).

**Commit:** `aximm_queue: optional peer-pointer caching to reduce round-trips`

---

## Phase 6: Worked example

**Goal:** A standalone, self-checking demo mirroring
[aximm_demo.py](../examples/interface/aximm_demo.py)'s style.

**Changes:** `examples/interface/aximm_queue_demo.py`

- Topology: `Producer (master_0)` and `Consumer (master_1)` over an
  `AXIMMCrossBarIF`, one `MemComponent` (`s_mm`) holding the ring region (FULL).
- Producer `write`s a known sequence (e.g. `np.arange(64)`), blocking on a small
  capacity (e.g. 8 slots) to exercise backpressure; consumer `get`s all and the
  harness asserts exact FIFO equality and prints transfer/timing summary.
- Module docstring with the topology diagram (ASCII, like the existing demo).
- `if __name__ == "__main__": …run_and_check()`.

**Tests:** `tests/examples/test_aximm_queue_demo.py` — import the harness, run
it, assert it passes (the pattern other example tests use).

**Commit:** `examples: aximm_queue_demo — SPSC ring buffer over AXI-MM crossbar`

---

## Phase 7 (optional): Docs

**Goal:** Document the queue alongside the other interface docs.

**Changes:**

- Add a page under `docs/guide/interface/` (check the existing
  `regmap.md`/interface docs for nav_order and front-matter conventions).
  Cover: the memory layout, SPSC contract, `write`/`get` API, blocking vs
  non-blocking, and a pointer to the demo.
- Update the interface section index/nav if there is one.

**Commit:** `docs: document AXIMMQueue memory-backed ring buffer`

---

## Future / out of scope (capture, don't build)

- **Word-addressed / BRAM backing.** This queue is byte-addressed AXI-MM only
  (decision 4). Mapping a queue onto word-addressed FPGA BRAM over a direct
  memory interface needs an `AddrUnit`-style abstraction. `pysilicon/hw/memory.py`
  already has exactly that (`AddrUnit.byte`/`word`, `word_size`, byte↔word
  conversion + alignment helpers, and `read/write/*_schema/*_array`) — but it is
  a **standalone numpy model not integrated into the SimPy/codegen framework**
  (its methods return `Any`, not `ProcessGen`, and it is not an `MMIFSlave`).
  Generalizing the queue would mean either wiring `Memory` into the SimPy
  framework or teaching `AXIMMQueueLayout` an `AddrUnit`. Deliberately deferred.
- **MPSC/MPMC** needs an atomic pointer update or a lock register — AXI-MM here
  has no atomics. Would require a new interconnect primitive (e.g. an
  exclusive-access or compare-and-swap slave op).
- **HLS codegen** for the ring (generate `m_axi` `write`/`get`). **Unblocked:**
  m_axi kernel **and** testbench codegen now exist and are cosim-proven — see the
  `examples/increment/` increment-buffer toy (the `IncrAccel` `HwComponent` →
  generated `incr.cpp`/`incr_tb.cpp`, validated through Vitis C-synth + RTL
  co-sim + AXI-MM burst extraction). The pieces to build the ring on:
  `MMArrayReadStmt`/`MMArrayWriteStmt` in the synthesizable statement IR
  (`hwcodegen.py`/`hwstmt.py`), the m_axi paths in `hwgen.py` (`_discover_mm_masters`,
  `kernel_signature` m_axi pragma + local-buffer sizing, `_emit_mm_array_read/write`
  lowering to `array_utils`), and the TB emitters (`MemComponent`→array+`MemMgr`,
  `alloc_array`/`read_array`, mem-pointer `KernelCallStmt`). The ring's `write`/`get`
  generate on top of these. The `aximm_codegen` plan that delivered this has been
  retired (all phases done).
- **Interrupt/doorbell** instead of polling for the blocking path — would let a
  consumer sleep until a doorbell write rather than poll `tail`.
