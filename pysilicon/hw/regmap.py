"""
regmap.py — Register-map abstraction for PySilicon.

Provides a Python-native register map that bridges a DataSchema-typed backing
store to an AXI-Lite-compatible MMIFSlave endpoint.

Public API
----------
RegAccess            — Enum of host access modes
RegField             — Dataclass declaring one field (schema, access, hooks)
RegMap               — Ordered collection of RegFields with numpy word buffers
RegMapAccessError    — Raised on access-mode violation or bad offset
RegMapMMIFSlave      — MMIFSlave subclass dispatching to a RegMap
VitisRegMap          — RegMap with auto-prepended ap_start at 0x00
VitisRegMapMMIFSlave — RegMapMMIFSlave with Vitis kernel launch lifecycle
Bit                  — IntField.specialize(bitwidth=1, signed=False)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal

import numpy as np

from pysilicon.hw.dataschema import DataSchema, IntField, Words
from pysilicon.hw.hwstmt import SynthCallStmt
from pysilicon.hw.memif import MMIFMaster, MMIFSlave
from pysilicon.hw.synth import synthesizable
from pysilicon.simulation.simobj import ProcessGen

# Single-bit unsigned integer field alias.
# Could become a real class in a future version.
Bit: type[IntField] = IntField.specialize(bitwidth=1, signed=False)


# ---------------------------------------------------------------------------
# Synthesizable IR nodes for regmap accesses
# ---------------------------------------------------------------------------


@dataclass
class RegMapGetStmt(SynthCallStmt):
    """Synthesizable read of a regmap field — emits an AXI-Lite scalar read."""


@dataclass
class RegMapSetStmt(SynthCallStmt):
    """Synthesizable write to a regmap field — emits an AXI-Lite scalar write."""


# ---------------------------------------------------------------------------
# Access mode enum
# ---------------------------------------------------------------------------


class RegAccess(Enum):
    """Host access mode for a register field.

    | Mode | Host read | Host write | Owner read | Owner write |
    |------|-----------|------------|------------|-------------|
    | R    | OK        | rejected   | OK         | OK          |
    | W    | rejected  | OK         | OK         | OK          |
    | RW   | OK        | OK         | OK         | OK          |
    | W1C  | OK        | clear bits | OK         | OK          |
    | W1S  | OK        | set+clear  | OK         | OK          |
    """

    R   = "R"
    W   = "W"
    RW  = "RW"
    W1C = "W1C"
    W1S = "W1S"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RegMapAccessError(RuntimeError):
    """Raised on host access-mode violation, offset miss, or unaligned address."""


# ---------------------------------------------------------------------------
# RegField — one field declaration
# ---------------------------------------------------------------------------


@dataclass
class RegField:
    """Declaration of one register map field.

    Parameters
    ----------
    schema:      DataSchema subclass (IntField, EnumField, DataArray, …)
    access:      Host access mode
    description: Free-text; included in generated documentation
    on_write:    Hook fired per-word after the backing store update
    on_read:     Hook fired per-word after reading the backing store
    offset:      Manual byte offset; None = auto-assign in declaration order
    """

    schema:      type[DataSchema]
    access:      RegAccess
    description: str = ""
    on_write:    Callable[[str, int, int], None] | None = None
    on_read:     Callable[[str, int, int], None] | None = None
    offset:      int | None = None
    is_vitis_auto: bool = False   # True for fields Vitis auto-generates (ap_start, etc.)
                                  # — present in PySim but skipped in C++ codegen.


# ---------------------------------------------------------------------------
# RegMap — ordered field collection with numpy backing buffers
# ---------------------------------------------------------------------------


class RegMap:
    """
    Ordered collection of RegFields with per-field numpy word buffers.

    The backing store is word-aligned; each field's words are consecutive.
    Host-side access goes through write_word / read_word (called by
    RegMapMMIFSlave). Owner-side access goes through get / set.
    """

    def __init__(self, fields: dict[str, RegField], bitwidth: int = 32) -> None:
        self.bitwidth = bitwidth
        self._fields: dict[str, RegField] = dict(fields)
        word_bytes = bitwidth // 8

        # W1C / W1S require single-word scalar fields.
        for name, f in fields.items():
            if f.access in (RegAccess.W1C, RegAccess.W1S):
                nw = f.schema.nwords_per_inst(bitwidth)
                if nw != 1:
                    raise ValueError(
                        f"Field '{name}': {f.access.value} requires a single-word "
                        f"field but schema occupies {nw} words."
                    )

        # ------------------------------------------------------------------
        # Offset assignment
        # ------------------------------------------------------------------
        occupied: set[int] = set()
        self._offsets: dict[str, int] = {}

        # Pass 1: register manually-placed offsets and detect overlaps.
        for name, f in fields.items():
            if f.offset is not None:
                nwords = f.schema.nwords_per_inst(bitwidth)
                for k in range(nwords):
                    pos = f.offset + k * word_bytes
                    if pos in occupied:
                        raise ValueError(
                            f"Field '{name}' at offset 0x{f.offset:x} overlaps "
                            f"with another field at byte 0x{pos:x}."
                        )
                    occupied.add(pos)
                self._offsets[name] = f.offset

        # Pass 2: auto-assign remaining fields in declaration order.
        next_free = 0
        for name, f in fields.items():
            if f.offset is not None:
                continue
            nwords = f.schema.nwords_per_inst(bitwidth)
            while not all(
                (next_free + k * word_bytes) not in occupied for k in range(nwords)
            ):
                next_free += word_bytes
            self._offsets[name] = next_free
            for k in range(nwords):
                occupied.add(next_free + k * word_bytes)
            next_free += nwords * word_bytes

        # ------------------------------------------------------------------
        # Backing store — zero-initialised, one numpy array per field.
        # ------------------------------------------------------------------
        dtype = np.uint32 if bitwidth <= 32 else np.uint64
        self._buffers: dict[str, np.ndarray] = {
            name: np.zeros(f.schema.nwords_per_inst(bitwidth), dtype=dtype)
            for name, f in fields.items()
        }

    # ------------------------------------------------------------------
    # Layout queries
    # ------------------------------------------------------------------

    def offset_of(self, name: str) -> int:
        """Return the byte offset of field *name*."""
        try:
            return self._offsets[name]
        except KeyError:
            raise KeyError(f"No field '{name}' in this RegMap.") from None

    def nwords_of(self, name: str) -> int:
        """Return the number of bus words occupied by field *name*."""
        return int(self._fields[name].schema.nwords_per_inst(self.bitwidth))

    def total_size_bytes(self) -> int:
        """Return the total byte span of this RegMap."""
        if not self._offsets:
            return 0
        word_bytes = self.bitwidth // 8
        return max(
            off + self.nwords_of(n) * word_bytes
            for n, off in self._offsets.items()
        )

    # ------------------------------------------------------------------
    # Owner-side value access
    # ------------------------------------------------------------------

    @synthesizable(stmt_class=RegMapGetStmt)
    def get(self, name: str) -> Any:
        """Deserialize the backing buffer and return a schema instance."""
        f = self._fields[name]
        return f.schema().deserialize(self._buffers[name], word_bw=self.bitwidth)

    @synthesizable(stmt_class=RegMapSetStmt)
    def set(self, name: str, value: Any) -> None:
        """Overwrite the field's backing buffer from a schema instance or raw value.

        Raw values are wrapped via ``schema(value)`` before serialization.
        W1C / W1S semantics are NOT applied; the buffer is overwritten directly.
        """
        f = self._fields[name]
        nwords = self.nwords_of(name)
        if not isinstance(value, f.schema):
            value = f.schema(value)  # type: ignore[call-arg]
        words = value.serialize(word_bw=self.bitwidth)
        if len(words) != nwords:
            raise ValueError(
                f"Serialized length {len(words)} != expected {nwords} for '{name}'."
            )
        self._buffers[name][:] = words

    # ------------------------------------------------------------------
    # Bus-level word access (used by RegMapMMIFSlave)
    # ------------------------------------------------------------------

    def field_name_at_offset(self, byte_offset: int) -> tuple[str, int]:
        """Return (field_name, sub_word_index) for *byte_offset*, or raise."""
        word_bytes = self.bitwidth // 8
        for name, off in self._offsets.items():
            for k in range(self.nwords_of(name)):
                if off + k * word_bytes == byte_offset:
                    return name, k
        raise RegMapAccessError(
            f"No field at byte offset 0x{byte_offset:x} in this RegMap."
        )

    def read_word(self, name: str, sub_word: int) -> int:
        """Return one word from the field's backing buffer."""
        return int(self._buffers[name][sub_word])

    def write_word(
        self,
        name: str,
        sub_word: int,
        value: int,
        *,
        source: Literal["host", "owner"],
    ) -> int:
        """Write one word, applying W1C masking for host writes.

        Returns the post-write buffer value (before any subsequent auto-clear).
        W1S auto-clear is handled by RegMapMMIFSlave, not here.
        """
        buf = self._buffers[name]
        f = self._fields[name]
        if source == "host" and f.access == RegAccess.W1C:
            # Clear bits set in 'value' from the backing store.
            buf[sub_word] = int(buf[sub_word]) & (~value)
        else:
            buf[sub_word] = value
        return int(buf[sub_word])

    # ------------------------------------------------------------------
    # Host-side bound access
    # ------------------------------------------------------------------

    def bind_master(self, master: MMIFMaster, base_addr: int = 0) -> "BoundRegMap":
        """Return a :class:`BoundRegMap` proxying host-side bus access at
        ``base_addr`` via ``master``.  The proxy looks up each field's
        schema and offset from this regmap, so callers issue reads/writes
        by field name instead of recomputing addresses and schema types
        at every call site.
        """
        return BoundRegMap(self, master, base_addr)


# ---------------------------------------------------------------------------
# BoundRegMap — host-side proxy mirroring RegMap.get/set over an MMIFMaster
# ---------------------------------------------------------------------------


class BoundRegMap:
    """Host-side proxy binding a :class:`RegMap` to an :class:`MMIFMaster`
    plus a base address.  Exposes ``get`` / ``set`` / ``start`` methods
    whose names mirror :meth:`RegMap.get` / :meth:`RegMap.set` (the
    kernel-side, in-process API), but route through the master bus.

    ``get`` returns a native Python value:

    - ``IntField`` (including ``Bit``) → ``int``
    - ``FloatField`` → ``float``
    - ``EnumField`` → the matching ``IntEnum`` member
    - ``DataArray`` / ``DataList`` / other → the schema instance

    All methods are coroutines — call sites use ``yield from``.
    """

    def __init__(self, regmap: RegMap, master: MMIFMaster, base_addr: int = 0) -> None:
        self._regmap = regmap
        self._master = master
        self._base_addr = base_addr

    @property
    def base_addr(self) -> int:
        return self._base_addr

    def set(self, name: str, value: Any) -> ProcessGen[None]:
        """Write ``value`` to field ``name`` over the master bus.

        Raw values are wrapped via ``schema(value)`` to match the kernel
        side's :meth:`RegMap.set` ergonomics.
        """
        f = self._regmap._fields[name]
        if not isinstance(value, f.schema):
            value = f.schema(value)
        addr = self._base_addr + self._regmap.offset_of(name)
        yield from self._master.write_schema(value, addr=addr)

    def get(self, name: str) -> ProcessGen[Any]:
        """Read field ``name`` over the master bus and return a native value."""
        f = self._regmap._fields[name]
        addr = self._base_addr + self._regmap.offset_of(name)
        obj = yield from self._master.read_schema(f.schema, addr=addr)
        return self._to_native(obj, f.schema)

    def start(self) -> ProcessGen[None]:
        """Write 1 to ``ap_start`` (only valid on a :class:`VitisRegMap`)."""
        yield from self._master.write_schema(
            Bit(1),
            addr=self._base_addr + self._regmap.offset_of("ap_start"),
        )

    @staticmethod
    def _to_native(obj: Any, schema_cls: type) -> Any:
        from pysilicon.hw.dataschema import FloatField, IntField
        enum_type = getattr(schema_cls, "enum_type", None)
        if enum_type is not None:
            return enum_type(int(obj.val))
        if isinstance(schema_cls, type) and issubclass(schema_cls, IntField):
            return int(obj.val)
        if isinstance(schema_cls, type) and issubclass(schema_cls, FloatField):
            return float(obj.val)
        return obj


# ---------------------------------------------------------------------------
# RegMapMMIFSlave — wires MMIFSlave callbacks to a RegMap
# ---------------------------------------------------------------------------


@dataclass
class RegMapMMIFSlave(MMIFSlave):
    """MMIFSlave subclass that dispatches reads/writes to a RegMap.

    Wires ``rx_write_proc`` and ``rx_read_proc`` automatically; callers should
    not pass these kwargs.
    """

    regmap: RegMap = field(kw_only=True)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        if self.bitwidth != self.regmap.bitwidth:
            raise ValueError(
                f"Slave bitwidth ({self.bitwidth}) != RegMap bitwidth "
                f"({self.regmap.bitwidth})."
            )
        self.rx_write_proc = self._rx_write
        self.rx_read_proc = self._rx_read
        super().__post_init__()

    def _rx_write(self, words: Words, local_addr: int) -> ProcessGen[None]:
        """Per-word write dispatch: validate access, update buffer, fire hooks."""
        word_bytes = self.bitwidth // 8
        for i in range(len(words)):
            byte_addr = local_addr + i * word_bytes
            name, sub_word = self.regmap.field_name_at_offset(byte_addr)
            f = self.regmap._fields[name]
            if f.access == RegAccess.R:
                raise RegMapAccessError(
                    f"Host write to read-only field '{name}' at 0x{byte_addr:x}."
                )
            raw_val = int(words[i])
            # Update backing store (W1C masking applied inside write_word).
            self.regmap.write_word(name, sub_word, raw_val, source="host")
            # Hook fires after backing-store update, before W1S auto-clear.
            if f.on_write is not None:
                f.on_write(name, sub_word, raw_val)
            # W1S: auto-clear the bit after the hook returns.
            if f.access == RegAccess.W1S:
                self.regmap._buffers[name][sub_word] = 0
        yield self.timeout(0)

    def _rx_read(self, nwords: int, local_addr: int) -> ProcessGen[Words]:
        """Per-word read dispatch: validate access, read buffer, fire hooks."""
        word_bytes = self.bitwidth // 8
        dtype = np.uint32 if self.bitwidth <= 32 else np.uint64
        result = np.zeros(nwords, dtype=dtype)
        for i in range(nwords):
            byte_addr = local_addr + i * word_bytes
            name, sub_word = self.regmap.field_name_at_offset(byte_addr)
            f = self.regmap._fields[name]
            if f.access == RegAccess.W:
                raise RegMapAccessError(
                    f"Host read from write-only field '{name}' at 0x{byte_addr:x}."
                )
            word_val = self.regmap.read_word(name, sub_word)
            # Hook fires after read, before return.
            if f.on_read is not None:
                f.on_read(name, sub_word, word_val)
            result[i] = word_val
        yield self.timeout(0)
        return result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# VitisRegMap — RegMap with Vitis ap_ctrl_hs conventions (v1)
# ---------------------------------------------------------------------------


class VitisRegMap(RegMap):
    """RegMap with Vitis ap_ctrl_hs control conventions auto-applied.

    v1: prepends ``ap_start`` (W1S) at offset 0x00.  User fields start at 0x04.
    v2: expands to the full bit-packed control word (ap_done, ap_idle, ap_ready,
        auto_restart) plus optional GIE/IER/ISR interrupt registers. (Not in v1.)
    """

    def __init__(self, fields: dict[str, RegField], bitwidth: int = 32) -> None:
        for name, f in fields.items():
            if name.startswith("ap_"):
                raise ValueError(
                    f"Field name '{name}' begins with reserved prefix 'ap_'."
                )
            if f.offset == 0:
                raise ValueError(
                    f"Field '{name}' specifies offset=0, which collides with "
                    "the auto-prepended ap_start register."
                )
        ctrl: dict[str, RegField] = {
            "ap_start": RegField(
                Bit,
                RegAccess.W1S,
                offset=0x00,
                description="Start the kernel (Vitis ap_ctrl_hs)",
                is_vitis_auto=True,
            )
        }
        super().__init__({**ctrl, **fields}, bitwidth=bitwidth)

    def start(self, master: MMIFMaster, base_addr: int = 0) -> ProcessGen[None]:
        """Convenience host-side launch: write 1 to ``ap_start``."""
        yield from master.write_schema(Bit(1), addr=base_addr + self.offset_of("ap_start"))


# ---------------------------------------------------------------------------
# VitisRegMapMMIFSlave — owns the Vitis kernel launch lifecycle
# ---------------------------------------------------------------------------


@dataclass
class VitisRegMapMMIFSlave(RegMapMMIFSlave):
    """RegMapMMIFSlave that invokes an ``on_start`` generator on ap_start writes.

    When the host writes 1 to ``ap_start``:

    1. If ``on_start`` is already running, the write is silently ignored
       (mirrors Vitis ap_ctrl_hs gating by ap_idle).  The W1S auto-clear
       of ap_start still fires.
    2. Otherwise the slave spawns ``env.process(on_start())`` and marks
       itself busy.
    3. When ``on_start`` returns, the slave marks itself idle.
    """

    regmap:   VitisRegMap                           = field(kw_only=True)
    on_start: Callable[[], ProcessGen[None]] | None = field(default=None, kw_only=True)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    def __post_init__(self) -> None:
        ap_field = self.regmap._fields["ap_start"]
        if ap_field.on_write is not None:
            raise ValueError(
                "ap_start on_write hook is reserved for VitisRegMapMMIFSlave."
            )
        self._busy: bool = False
        ap_field.on_write = self._on_ap_start
        super().__post_init__()

    def _on_ap_start(self, name: str, sub_word: int, value: int) -> None:
        """Hook installed on ap_start; spawns _launch() if not busy."""
        if self._busy:
            return
        self._busy = True
        self.env.process(self._launch())

    def _launch(self) -> ProcessGen[None]:
        """Runs on_start() and clears _busy when it returns."""
        try:
            if self.on_start is not None:
                result = self.on_start()
                if result is not None:
                    yield from result
        finally:
            self._busy = False
