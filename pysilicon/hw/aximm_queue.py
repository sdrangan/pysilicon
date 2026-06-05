"""
aximm_queue.py — memory-backed AXI-MM ring buffer (SPSC FIFO).

An :class:`AXIMMQueue` is a producer/consumer FIFO whose storage lives in an
AXI-MM memory region.  It is built entirely on the existing ``MMIFMaster`` /
``MMIFSlave`` primitives in :mod:`pysilicon.hw.memif`: the ring lives in
ordinary MM memory and the head/tail pointers are ordinary MM words, so the
queue works over *any* MM interconnect (``AXIMMCrossBarIF`` or ``DirectMMIF``).

Design (see ``plans/aximm_queue.md`` for the full rationale)
------------------------------------------------------------
* **Pointers live in MM, not in Python.**  Two independent masters share state
  purely through memory; there is no shared Python object.
* **SPSC, lock-free.**  Exactly one producer (writes ``tail`` + data, reads
  ``head``) and one consumer (writes ``head`` + reads data, reads ``tail``).
  No pointer is written by both sides, so no lock is needed.
* **Word-index pointers, one reserved slot.**  ``head`` and ``tail`` are slot
  indices in ``[0, capacity)``; ``empty ⇔ head == tail`` and
  ``full ⇔ (tail + 1) % capacity == head``.  Usable depth is ``capacity - 1``.
* **Byte-addressed AXI-MM only.**  Every offset is computed from ``mem_bw``;
  nothing is hard-coded.

This module exposes two classes:

``AXIMMQueueLayout``
    The memory map.  Owns all address math given ``mem_bw``, ``capacity`` and
    ``elem_words``.
``MMMemory``
    A reusable word-addressed RAM ``SimObj`` backing an MM slave (a cleaned-up,
    generalized version of the ``MemBank`` defined locally in
    ``examples/interface/aximm_demo.py``).  Lives here (rather than in
    ``memif.py``) so the queue's dependency on a concrete memory model stays
    local; any MM slave with read/write callbacks works — ``MMMemory`` is just
    the convenient default.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from pysilicon.hw.memif import MMIFMaster, MMIFSlave, Words
from pysilicon.simulation.simobj import ProcessGen, SimObj


# ---------------------------------------------------------------------------
# Memory map
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AXIMMQueueLayout:
    """Memory map for a ring buffer in an AXI-MM region.

    Every offset is derived from ``mem_bw``; one control field occupies one
    memory word.  Pointers (head/tail) are slot indices in ``[0, capacity)``.
    Usable depth is ``capacity - 1`` (one slot reserved to distinguish full
    from empty).

    Layout (control words precede the data slots)::

        word 0 : head      (consumer-owned slot index)
        word 1 : tail      (producer-owned slot index)
        word 2 : capacity  (number of slots; informational)
        word 3 : reserved  (0; room for status/flags later)
        data   : capacity slots × elem_words words

    Addressing is byte-addressed (AXI-MM / DDR convention): a word spans
    ``mem_bw // 8`` byte addresses, so word *i* sits at ``base + i * word_bytes``.
    """

    # One control field per memory word: head, tail, capacity, reserved.
    NUM_CONTROL_WORDS: ClassVar[int] = 4

    base_addr: int
    capacity: int            # number of slots
    elem_words: int = 1      # words per slot
    mem_bw: int = 32         # memory data width in bits (AXI-MM: byte-addressed)

    def __post_init__(self) -> None:
        if self.capacity < 2:
            raise ValueError(
                f"capacity must be >= 2 (one slot is reserved), got {self.capacity}"
            )
        if self.elem_words < 1:
            raise ValueError(f"elem_words must be >= 1, got {self.elem_words}")
        if self.mem_bw not in (8, 16, 32, 64):
            raise ValueError(
                f"mem_bw must be one of (8, 16, 32, 64), got {self.mem_bw}"
            )

    # -- word geometry ------------------------------------------------------

    @property
    def word_bytes(self) -> int:
        # AXI-MM is byte-addressed: a word spans mem_bw // 8 byte addresses.
        return self.mem_bw // 8

    @property
    def control_bytes(self) -> int:
        return self.NUM_CONTROL_WORDS * self.word_bytes

    # -- control-word addresses --------------------------------------------

    @property
    def head_addr(self) -> int:
        return self.base_addr + 0 * self.word_bytes

    @property
    def tail_addr(self) -> int:
        return self.base_addr + 1 * self.word_bytes

    @property
    def capacity_addr(self) -> int:
        return self.base_addr + 2 * self.word_bytes

    # -- data region --------------------------------------------------------

    @property
    def data_base(self) -> int:
        return self.base_addr + self.control_bytes

    def slot_addr(self, idx: int) -> int:
        """Byte address of slot *idx* (0 <= idx < capacity)."""
        return self.data_base + idx * self.elem_words * self.word_bytes

    @property
    def total_bytes(self) -> int:
        """Size of the whole region, for ``assign_address_ranges``."""
        return self.control_bytes + self.capacity * self.elem_words * self.word_bytes


# ---------------------------------------------------------------------------
# Reusable memory slave
# ---------------------------------------------------------------------------

@dataclass
class MMMemory(SimObj):
    """Word-addressed RAM backing an MM slave (generalized aximm_demo MemBank).

    Backed by a dict keyed by byte address.  In the FULL protocol the slave
    callback receives the whole burst at the starting ``local_addr`` and is
    responsible for striding words within it; the stride per word is
    ``bitwidth // 8`` bytes, never a hard-coded 4.

    Parameters
    ----------
    bitwidth : int
        Data word width in bits.  Must match the queue layout's ``mem_bw`` and
        the interconnect ``bitwidth``.
    access_latency : float
        Simulated read access time (seconds).  Writes are non-blocking.
    """

    bitwidth: int = 32
    access_latency: float = 0.0

    slave_ep: MMIFSlave = field(init=False)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._mem: dict[int, int] = {}
        self.slave_ep = MMIFSlave(
            sim=self.sim,
            bitwidth=self.bitwidth,
            rx_write_proc=self.rx_write,
            rx_read_proc=self.rx_read,
        )

    @property
    def _word_bytes(self) -> int:
        return self.bitwidth // 8

    def rx_write(self, words: Words, local_addr: int) -> ProcessGen[None]:
        word_bytes = self._word_bytes
        for i, w in enumerate(words):
            self._mem[local_addr + i * word_bytes] = int(w)
        yield self.timeout(0)   # write is non-blocking for the caller

    def rx_read(self, nwords: int, local_addr: int) -> ProcessGen[Words]:
        if self.access_latency > 0:
            yield self.timeout(self.access_latency)
        else:
            yield self.timeout(0)
        word_bytes = self._word_bytes
        dtype = np.uint32 if self.bitwidth <= 32 else np.uint64
        return np.array(
            [self._mem.get(local_addr + i * word_bytes, 0) for i in range(nwords)],
            dtype=dtype,
        )


# ---------------------------------------------------------------------------
# Queue proxy
# ---------------------------------------------------------------------------

#: Default poll interval (simulation seconds) for the blocking write/get loops.
DEFAULT_POLL_INTERVAL: float = 1.0


def _split(idx: int, nslots: int, capacity: int) -> list[tuple[int, int]]:
    """Split a run of *nslots* slots starting at *idx* across the ring wrap.

    Returns one or two ``(start_idx, count)`` runs so a ``write``/``get`` issues
    at most two MM transfers (decision 10) — never a per-element modular loop.
    """
    first = min(nslots, capacity - idx)
    runs = [(idx, first)]
    if nslots > first:
        runs.append((0, nslots - first))
    return runs


@dataclass
class AXIMMQueue:
    """Proxy a master uses to ``write``/``get`` a memory-backed ring buffer.

    Wraps an :class:`MMIFMaster` and an :class:`AXIMMQueueLayout`.  The producer
    side calls ``write*`` (writes data + ``tail``, reads ``head``); the consumer
    side calls ``get*`` (reads data + ``tail``, writes ``head``).  Two instances
    over the same layout/base — one per master endpoint — form the full SPSC
    queue (decision 7).

    The non-blocking ``try_write`` / ``try_get`` are the primitives; the public
    blocking ``write`` / ``get`` (Phase 3) poll them.
    """

    master: MMIFMaster
    layout: AXIMMQueueLayout

    def __post_init__(self) -> None:
        # Decision 11: the layout's mem_bw is the single source of word width
        # and must match the bound master/interconnect bitwidth, else every
        # computed byte offset would be wrong.
        if self.master.bitwidth != self.layout.mem_bw:
            raise ValueError(
                f"AXIMMQueue: master bitwidth {self.master.bitwidth} does not "
                f"match layout mem_bw {self.layout.mem_bw}"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _dtype(self) -> np.dtype:
        return np.dtype(np.uint32) if self.layout.mem_bw <= 32 else np.dtype(np.uint64)

    def _read_ptrs(self) -> ProcessGen[tuple[int, int]]:
        """Read (head, tail) in one transaction (they are adjacent words)."""
        w = yield from self.master.read(2, self.layout.head_addr)
        return int(w[0]), int(w[1])

    # ------------------------------------------------------------------
    # Setup / status
    # ------------------------------------------------------------------

    def reset(self) -> ProcessGen[None]:
        """Zero head and tail and record capacity (call once, by whichever side
        owns setup)."""
        ctrl = np.zeros(self.layout.NUM_CONTROL_WORDS, dtype=self._dtype)
        ctrl[2] = self.layout.capacity   # informational capacity word
        yield from self.master.write(ctrl, self.layout.head_addr)

    def count(self) -> ProcessGen[int]:
        """Number of occupied slots."""
        head, tail = yield from self._read_ptrs()
        return (tail - head) % self.layout.capacity

    def space(self) -> ProcessGen[int]:
        """Number of free (usable) slots; usable depth is capacity - 1."""
        c = yield from self.count()
        return (self.layout.capacity - 1) - c

    # ------------------------------------------------------------------
    # Non-blocking primitives
    # ------------------------------------------------------------------

    def try_write(self, words: Words) -> ProcessGen[bool]:
        """Try to enqueue ``len(words) // elem_words`` slots.

        Returns ``False`` (a no-op) if the whole batch will not fit.  Data is
        written **before** ``tail`` is advanced so the consumer can never see a
        tail pointing at unwritten data (the SPSC ordering crux, decision 2).
        """
        ew = self.layout.elem_words
        cap = self.layout.capacity
        nslots = len(words) // ew
        if nslots * ew != len(words):
            raise ValueError(
                f"try_write: {len(words)} words is not a multiple of "
                f"elem_words={ew}"
            )
        if nslots == 0:
            return True

        head, tail = yield from self._read_ptrs()
        free = (cap - 1) - (tail - head) % cap
        if nslots > free:
            return False

        # 1) write data (possibly split across the wrap)
        offset = 0
        for start, cnt in _split(tail, nslots, cap):
            nwords = cnt * ew
            yield from self.master.write(
                words[offset:offset + nwords], self.layout.slot_addr(start)
            )
            offset += nwords

        # 2) only now advance tail
        new_tail = (tail + nslots) % cap
        yield from self.master.write(
            np.array([new_tail], dtype=self._dtype), self.layout.tail_addr
        )
        return True

    def try_get(self, max_slots: int) -> ProcessGen[Words]:
        """Dequeue up to *max_slots* slots; returns the words actually read
        (possibly short or empty).

        Data is read **before** ``head`` is advanced so the producer can never
        reclaim a slot still being read (the SPSC ordering crux, decision 2).
        """
        ew = self.layout.elem_words
        cap = self.layout.capacity

        head, tail = yield from self._read_ptrs()
        avail = (tail - head) % cap
        nslots = min(max_slots, avail)
        if nslots == 0:
            return np.empty(0, dtype=self._dtype)

        # 1) read data (possibly split across the wrap)
        chunks = []
        for start, cnt in _split(head, nslots, cap):
            chunk = yield from self.master.read(cnt * ew, self.layout.slot_addr(start))
            chunks.append(chunk)

        # 2) only now advance head
        new_head = (head + nslots) % cap
        yield from self.master.write(
            np.array([new_head], dtype=self._dtype), self.layout.head_addr
        )
        return np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

    # ------------------------------------------------------------------
    # Blocking public API (decision 8) — poll the try_* primitives.
    # ------------------------------------------------------------------

    def write(
        self, words: Words, poll_interval: float = DEFAULT_POLL_INTERVAL
    ) -> ProcessGen[None]:
        """Enqueue *words* as one atomic batch, blocking until it fits.

        The batch is enqueued all-or-nothing, so it must fit in the usable depth
        (``capacity - 1`` slots); a producer that wants to stream more than that
        chunks the data into multiple ``write`` calls.  Polls :meth:`try_write`,
        sleeping *poll_interval* simulation seconds between attempts; the loop is
        cancelled by simulation end like any other SimObj process.
        """
        nslots = len(words) // self.layout.elem_words
        if nslots > self.layout.capacity - 1:
            raise ValueError(
                f"write: batch of {nslots} slots can never fit in usable depth "
                f"{self.layout.capacity - 1}; chunk the data into smaller writes"
            )
        while not (yield from self.try_write(words)):
            yield self.master.timeout(poll_interval)

    def get(
        self, nslots: int, poll_interval: float = DEFAULT_POLL_INTERVAL
    ) -> ProcessGen[Words]:
        """Dequeue exactly *nslots* slots, blocking until they are available.

        Polls :meth:`try_get`, sleeping *poll_interval* simulation seconds when
        the queue is empty; the loop is cancelled by simulation end like any
        other SimObj process.
        """
        ew = self.layout.elem_words
        out: list[Words] = []
        collected = 0
        while collected < nslots:
            chunk = yield from self.try_get(nslots - collected)
            if len(chunk) == 0:
                yield self.master.timeout(poll_interval)
            else:
                out.append(chunk)
                collected += len(chunk) // ew
        return np.concatenate(out) if len(out) > 1 else out[0]
