import numpy as np
from enum import Enum


class AddrUnit(Enum):
    byte = 0
    word = 1


class Memory(object):
    """
    Sparse memory model for PySilicon.

    Attributes
    ----------
    
    segments : dict
        A dictionary of the allocated memory segments.  The key is the starting address.
        The value is a numpy array representing the memory segment.  Consistent with the 
        DataSchema.serialize method, if the word size is <= 32 bits, memory segments are 
        stored as uint32 arrays.  Else if the word size is <= 64 bits, 
        memory segments are stored as uint64 arrays.  If the word size is > 64 bits, 
        memory as [nelem, nwords64] where nwords64 = ceil(word_size / 64) and 
        each word is stored as a uint64.  
    """

    def __init__(
            self, 
            word_size : int =32, 
            addr_size : int =32,
            nwords_tot : int | None = None,
            addr_unit : AddrUnit = AddrUnit.byte):
        """
        Parameters
        ----------
        word_size : int
            The size of a word in bits.  Default is 32 bits.
        addr_size : int
            The size of an address in bits.  Default is 32 bits.
        nwords_tot : int | None
            The total number of words in the memory.  If None, the memory is unbounded.  
            Default is None.
        addr_unit : AddrUnit
            Specifies the *meaning* of an address:
        
            - AddrUnit.byte:
                The address is a byte address (AXI4 / Pynq / DDR style).
                The simulator will convert byte addresses to word indices
                using (addr // (word_size // 8)).

            - AddrUnit.word:
                The address is a word index (HLS-local array style).
                The simulator will treat the address directly as an index
                into the underlying numpy word array.

            Default is byte-addressable.
        """
        self.word_size = word_size
        self.addr_size = addr_size
        self.addr_unit = addr_unit
        self.nwords_tot = nwords_tot
        self.segments = {}

    def _addr_to_index(self, addr : int) -> int:
        if self.addr_unit == AddrUnit.byte:
            word_nbytes = self.word_size // 8
            if self.word_size % 8 != 0:
                raise ValueError("Byte addressing requires word_size to be a multiple of 8 bits.")
            if addr % word_nbytes != 0:
                raise ValueError(
                    f"Address {addr} is not aligned to the word size of {word_nbytes} bytes."
                )
            return addr // word_nbytes
        elif self.addr_unit == AddrUnit.word:
            return addr
        else:
            raise ValueError(f"Unsupported address unit: {self.addr_unit}")

    def _index_to_addr(self, index: int) -> int:
        if self.addr_unit == AddrUnit.byte:
            if self.word_size % 8 != 0:
                raise ValueError("Byte addressing requires word_size to be a multiple of 8 bits.")
            return index * (self.word_size // 8)
        if self.addr_unit == AddrUnit.word:
            return index
        raise ValueError(f"Unsupported address unit: {self.addr_unit}")

    def _segment_shape(self, nwords: int) -> tuple[int, ...]:
        if self.word_size <= 32:
            return (nwords,)
        if self.word_size <= 64:
            return (nwords,)
        return (nwords, (self.word_size + 63) // 64)

    def _segment_dtype(self):
        if self.word_size <= 32:
            return np.uint32
        return np.uint64

    def _segment_bounds(self) -> list[tuple[int, int]]:
        bounds = []
        for start_addr, segment in self.segments.items():
            start_index = self._addr_to_index(start_addr)
            end_index = start_index + int(segment.shape[0])
            bounds.append((start_index, end_index))
        bounds.sort()
        return bounds

    def _locate_segment(self, addr: int) -> tuple[np.ndarray, int, int, int]:
        addr_index = self._addr_to_index(addr)
        for start_addr, segment in self.segments.items():
            start_index = self._addr_to_index(start_addr)
            end_index = start_index + int(segment.shape[0])
            if start_index <= addr_index < end_index:
                return segment, addr_index - start_index, start_index, end_index
        raise ValueError(f"Address {addr} does not fall within any allocated segment.")
    
    def alloc(
            self,
            nwords : int
    ) -> int:
        """
        Allocate a contiguous memory segment.

        Parameters
        ----------
        nwords : int
            Number of contiguous words to allocate.

        Returns
        -------
        int
            Starting address of the allocated segment. The returned address uses
            the configured ``addr_unit`` semantics, so it is either a byte address
            or a word index.

        Raises
        ------
        ValueError
            If ``nwords`` is not positive.
        MemoryError
            If a contiguous segment of ``nwords`` words cannot be found within
            the configured memory bounds.

        Notes
        -----
        Allocation uses a first-fit search over the existing sparse segments.
        The newly allocated segment is zero-initialized and stored in
        ``self.segments`` under its starting address.
        """
        if nwords <= 0:
            raise ValueError("nwords must be a positive integer.")

        next_free = 0
        for start_index, end_index in self._segment_bounds():
            if next_free + nwords <= start_index:
                break
            next_free = max(next_free, end_index)

        if self.nwords_tot is not None and next_free + nwords > self.nwords_tot:
            raise MemoryError(
                f"Unable to allocate {nwords} contiguous words in memory of size {self.nwords_tot}."
            )

        addr = self._index_to_addr(next_free)
        self.segments[addr] = np.zeros(
            self._segment_shape(nwords),
            dtype=self._segment_dtype(),
        )
        return addr

    def free(
            self,
            addr : int
    ):
        """
        Free a previously allocated memory segment.

        Parameters
        ----------
        addr : int
            Starting address of the segment to free, expressed in the configured
            ``addr_unit``.

        Raises
        ------
        KeyError
            If ``addr`` is not the starting address of an allocated segment.

        Notes
        -----
        Only full segments can be freed. Passing an address inside an allocated
        segment, rather than its start address, raises ``KeyError``.
        """
        if addr not in self.segments:
            raise KeyError(f"No allocated segment starts at address {addr}.")
        del self.segments[addr]

    def read(
            self,
            addr : int,
            nwords : int = 1
    ) -> np.ndarray:
        """
        Read a contiguous slice from an allocated memory segment.

        Parameters
        ----------
        addr : int
            Starting address of the read, expressed in the configured
            ``addr_unit``.
        nwords : int
            Number of words to read. Default is 1.

        Returns
        -------
        np.ndarray
            Copy of the requested data. For word sizes above 64 bits, the result
            has shape ``(nwords, ceil(word_size / 64))``.

        Raises
        ------
        ValueError
            If ``nwords`` is not positive, if ``addr`` is not within an allocated
            segment, or if the requested range extends past the end of the segment.
        """
        if nwords <= 0:
            raise ValueError("nwords must be a positive integer.")

        segment, offset, _, end_index = self._locate_segment(addr)
        addr_index = self._addr_to_index(addr)
        if addr_index + nwords > end_index:
            raise ValueError(
                f"Read of {nwords} words from address {addr} exceeds the bounds of its segment."
            )

        return np.array(segment[offset:offset + nwords], copy=True)

    def write(
            self,
            addr : int,
            data : np.ndarray
    ):
        """
        Write a contiguous slice into an allocated memory segment.

        Parameters
        ----------
        addr : int
            Starting address of the write, expressed in the configured
            ``addr_unit``.
        data : np.ndarray
            Array of words to write. The leading dimension is interpreted as the
            number of words. For word sizes above 64 bits, each word must be
            represented by ``ceil(word_size / 64)`` uint64 chunks.

        Raises
        ------
        ValueError
            If ``data`` is empty, if ``addr`` is not within an allocated segment,
            if the write would extend past the end of the segment, or if ``data``
            does not match the segment word layout.
        """
        segment, offset, _, end_index = self._locate_segment(addr)

        data_arr = np.asarray(data, dtype=segment.dtype)
        if segment.ndim == 1:
            if data_arr.ndim == 0:
                data_arr = data_arr.reshape(1)
            elif data_arr.ndim != 1:
                raise ValueError("Data shape is incompatible with the memory word layout.")
        else:
            if data_arr.ndim == 1 and data_arr.shape[0] == segment.shape[1]:
                data_arr = data_arr.reshape(1, -1)
            elif data_arr.ndim != segment.ndim or data_arr.shape[1:] != segment.shape[1:]:
                raise ValueError("Data shape is incompatible with the memory word layout.")

        nwords = int(data_arr.shape[0])
        if nwords <= 0:
            raise ValueError("data must contain at least one word.")

        addr_index = self._addr_to_index(addr)
        if addr_index + nwords > end_index:
            raise ValueError(
                f"Write of {nwords} words from address {addr} exceeds the bounds of its segment."
            )

        segment[offset:offset + nwords] = data_arr
    