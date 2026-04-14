import numpy as np
import pytest

from pysilicon.hw.memory import AddrUnit, Memory


def test_alloc_returns_byte_address_and_reuses_freed_gap() -> None:
    memory = Memory(word_size=32, nwords_tot=8, addr_unit=AddrUnit.byte)

    addr0 = memory.alloc(2)
    addr1 = memory.alloc(3)

    assert addr0 == 0
    assert addr1 == 8
    assert memory.segments[addr0].dtype == np.uint32
    assert memory.segments[addr0].shape == (2,)

    memory.free(addr0)
    addr2 = memory.alloc(2)

    assert addr2 == addr0


def test_alloc_uses_uint64_chunks_for_wide_words() -> None:
    memory = Memory(word_size=96, addr_unit=AddrUnit.word)

    addr = memory.alloc(3)

    assert addr == 0
    assert memory.segments[addr].dtype == np.uint64
    assert memory.segments[addr].shape == (3, 2)


def test_alloc_raises_when_no_contiguous_segment_fits() -> None:
    memory = Memory(word_size=32, nwords_tot=5, addr_unit=AddrUnit.word)

    addr0 = memory.alloc(2)
    memory.alloc(2)
    memory.free(addr0)

    with pytest.raises(MemoryError, match="Unable to allocate 3 contiguous words"):
        memory.alloc(3)


def test_free_requires_segment_start_address() -> None:
    memory = Memory(word_size=32, addr_unit=AddrUnit.byte)
    addr = memory.alloc(2)

    with pytest.raises(KeyError, match=f"No allocated segment starts at address {addr + 4}"):
        memory.free(addr + 4)


def test_alloc_rejects_non_positive_sizes() -> None:
    memory = Memory()

    with pytest.raises(ValueError, match="nwords must be a positive integer"):
        memory.alloc(0)


def test_write_and_read_round_trip_with_byte_addresses() -> None:
    memory = Memory(word_size=32, addr_unit=AddrUnit.byte)
    addr = memory.alloc(4)
    expected = np.array([11, 22, 33], dtype=np.uint32)

    memory.write(addr + 4, expected)
    got = memory.read(addr + 4, nwords=3)

    assert np.array_equal(got, expected)
    assert got.dtype == np.uint32


def test_write_rejects_range_that_exceeds_segment() -> None:
    memory = Memory(word_size=32, addr_unit=AddrUnit.word)
    addr = memory.alloc(2)

    with pytest.raises(ValueError, match="Write of 3 words from address 0 exceeds the bounds"):
        memory.write(addr, np.array([1, 2, 3], dtype=np.uint32))


def test_read_rejects_address_outside_allocated_segments() -> None:
    memory = Memory(word_size=32, addr_unit=AddrUnit.word)
    memory.alloc(2)

    with pytest.raises(ValueError, match="Address 5 does not fall within any allocated segment"):
        memory.read(5, nwords=1)


def test_write_handles_wide_word_chunked_layout() -> None:
    memory = Memory(word_size=96, addr_unit=AddrUnit.word)
    addr = memory.alloc(2)
    expected = np.array([[1, 2], [3, 4]], dtype=np.uint64)

    memory.write(addr, expected)
    got = memory.read(addr, nwords=2)

    assert np.array_equal(got, expected)
    assert got.dtype == np.uint64


def test_write_rejects_incompatible_wide_word_shape() -> None:
    memory = Memory(word_size=96, addr_unit=AddrUnit.word)
    addr = memory.alloc(2)

    with pytest.raises(ValueError, match="Data shape is incompatible with the memory word layout"):
        memory.write(addr, np.array([1, 2, 3], dtype=np.uint64))