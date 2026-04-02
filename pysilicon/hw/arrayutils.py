"""Generate Vitis HLS array read and write helpers for packed element arrays.

Example
-------
```python
from pysilicon.hw.arrayutils import gen_array_utils, read_array, write_array
from pysilicon.build.build import CodeGenConfig
from pysilicon.hw.dataschema import IntField

Int16 = IntField.specialize(16, signed=True)
path = gen_array_utils(Int16, [32, 64], cfg=CodeGenConfig(root_dir="include"))
print(path)

packed = write_array([1, 2, 3, 4], elem_type=Int16, word_bw=32)
unpacked = read_array(packed, elem_type=Int16, word_bw=32, shape=4)
```
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import posixpath
import re
from typing import Any

import numpy as np

from pysilicon.build.build import CodeGenConfig
from pysilicon.hw.dataschema import DataArray, DataList, DataSchema


def write_array(arr: Any, elem_type: type[DataSchema], word_bw: int) -> np.ndarray:
    """Pack a Python array of schema elements into hardware words.

    Parameters
    ----------
    arr : Any
        Input array-like value. For scalar field element types this is typically a
        NumPy array or Python sequence. The full input shape is used as the array
        schema shape for serialization.
    elem_type : type[DataSchema]
        Element schema class describing each array entry.
    word_bw : int
        Packed output word width in bits.

    Returns
    -------
    numpy.ndarray
        Packed hardware words as returned by ``DataSchema.serialize()``. For
        ``word_bw <= 32`` the dtype is ``np.uint32``; for ``word_bw <= 64`` it is
        ``np.uint64``; larger widths follow the existing chunked ``serialize``
        behavior.
    """
    if not isinstance(elem_type, type) or not issubclass(elem_type, DataSchema):
        raise TypeError("elem_type must be a DataSchema subclass.")
    if word_bw <= 0:
        raise ValueError("word_bw must be positive.")

    np_arr = np.asarray(arr)
    shape = tuple(int(dim) for dim in np_arr.shape)

    array_cls = DataArray.specialize(
        element_type=elem_type,
        max_shape=shape,
        static=True,
    )
    array_obj = array_cls()
    array_obj.val = arr
    return array_obj.serialize(word_bw=word_bw)


def write_uint32_file(
    arr: Any,
    elem_type: type[DataSchema],
    file_path: str | Path,
    write_slice: Any = None,
    nwrite: int | None = None,
) -> Path:
    """Pack an array into 32-bit words and write it to a binary file.

    Parameters
    ----------
    arr : Any
        Input array-like value.
    elem_type : type[DataSchema]
        Element schema class describing each array entry.
    file_path : str | Path
        Destination binary file path.
    write_slice : Any, optional
        Optional NumPy-style slice used to select a subset of ``arr`` before
        packing. This matches the behavior of ``DataArray.write_uint32_file``.
    nwrite : int | None, optional
        Convenience argument that selects the first ``nwrite`` entries along the
        leading dimension. Mutually exclusive with ``write_slice``.

    Returns
    -------
    pathlib.Path
        The written file path.
    """
    if write_slice is not None and nwrite is not None:
        raise ValueError("Specify only one of write_slice or nwrite.")
    if nwrite is not None and nwrite < 0:
        raise ValueError("nwrite must be non-negative.")

    np_arr = np.asarray(arr)
    if np_arr.ndim > 0:
        if nwrite is not None:
            write_slice = (slice(0, int(nwrite)),) + (slice(None),) * (np_arr.ndim - 1)
    else:
        if nwrite is not None:
            if int(nwrite) == 0:
                write_slice = np.s_[:0]
            elif int(nwrite) == 1:
                write_slice = ()
            else:
                raise ValueError("nwrite > 1 is invalid for scalar-valued arrays.")

    selected = np_arr if write_slice is None else np_arr[write_slice]
    words = np.asarray(write_array(selected, elem_type=elem_type, word_bw=32), dtype="<u4")

    out_path = Path(file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    words.tofile(out_path)
    return out_path


def read_array(
    packed: Any,
    elem_type: type[DataSchema],
    word_bw: int,
    shape: int | tuple[int, ...] | list[int],
) -> Any:
    """Unpack hardware words into a Python array of schema elements.

    Parameters
    ----------
    packed : Any
        Packed hardware words. This may be any array-like object accepted by
        ``DataSchema.deserialize()``.
    elem_type : type[DataSchema]
        Element schema class describing each unpacked array entry.
    word_bw : int
        Packed input word width in bits.
    shape : int | tuple[int, ...] | list[int]
        Expected unpacked array shape. A scalar integer is treated as a 1D shape.

    Returns
    -------
    Any
        The unpacked Python-side array value held in the temporary ``DataArray``
        instance after deserialization.
    """
    if not isinstance(elem_type, type) or not issubclass(elem_type, DataSchema):
        raise TypeError("elem_type must be a DataSchema subclass.")
    if word_bw <= 0:
        raise ValueError("word_bw must be positive.")

    if isinstance(shape, int):
        norm_shape = (int(shape),)
    else:
        norm_shape = tuple(int(dim) for dim in shape)

    if any(dim < 0 for dim in norm_shape):
        raise ValueError("shape dimensions must be non-negative.")

    array_cls = DataArray.specialize(
        element_type=elem_type,
        max_shape=norm_shape,
        static=True,
    )
    array_obj = array_cls()
    array_obj.deserialize(np.asarray(packed), word_bw=word_bw)
    return array_obj.val


def read_uint32_file(
    file_path: str | Path,
    elem_type: type[DataSchema],
    shape: int | tuple[int, ...] | list[int],
) -> Any:
    """Read packed 32-bit words from a binary file into a Python array.

    Parameters
    ----------
    file_path : str | Path
        Source binary file path containing packed little-endian uint32 words.
    elem_type : type[DataSchema]
        Element schema class describing each unpacked array entry.
    shape : int | tuple[int, ...] | list[int]
        Expected unpacked array shape.

    Returns
    -------
    Any
        The unpacked Python-side array value.
    """
    in_path = Path(file_path)
    words = np.fromfile(in_path, dtype="<u4")
    return read_array(words, elem_type=elem_type, word_bw=32, shape=shape)


def _array_utils_stem(elem_type: type[DataSchema]) -> str:
    name = elem_type.__name__

    int_match = re.fullmatch(r"Int(\d+)", name)
    if int_match is not None:
        return f"int{int_match.group(1)}"

    uint_match = re.fullmatch(r"UInt(\d+)", name)
    if uint_match is not None:
        return f"uint{uint_match.group(1)}"

    return elem_type._camel_to_snake(name)


def _array_utils_filename(elem_type: type[DataSchema]) -> str:
    return f"{_array_utils_stem(elem_type)}_array_utils.h"


def _array_utils_tb_filename(elem_type: type[DataSchema]) -> str:
    return f"{_array_utils_stem(elem_type)}_array_utils_tb.h"


def _array_utils_include_path(elem_type: type[DataSchema]) -> str:
    include_dir = (elem_type.include_dir or ".").replace("\\", "/")
    include_root = PurePosixPath(include_dir)
    filename = _array_utils_filename(elem_type)
    if include_root.as_posix() == ".":
        return filename
    return f"{include_root.as_posix()}/{filename}"


def _array_utils_tb_include_path(elem_type: type[DataSchema]) -> str:
    include_dir = (elem_type.include_dir or ".").replace("\\", "/")
    include_root = PurePosixPath(include_dir)
    filename = _array_utils_tb_filename(elem_type)
    if include_root.as_posix() == ".":
        return filename
    return f"{include_root.as_posix()}/{filename}"


def _array_utils_include_guard(elem_type: type[DataSchema]) -> str:
    guard = re.sub(r"[^A-Za-z0-9]+", "_", _array_utils_include_path(elem_type)).strip("_").upper()
    return re.sub(r"_+", "_", guard)


def _array_utils_tb_include_guard(elem_type: type[DataSchema]) -> str:
    guard = re.sub(r"[^A-Za-z0-9]+", "_", _array_utils_tb_include_path(elem_type)).strip("_").upper()
    return re.sub(r"_+", "_", guard)


def _array_utils_namespace(elem_type: type[DataSchema]) -> str:
    return f"{_array_utils_stem(elem_type)}_array_utils"


def _relative_synth_include_from_tb(elem_type: type[DataSchema]) -> str:
    current_dir = posixpath.dirname(_array_utils_tb_include_path(elem_type)) or "."
    return posixpath.relpath(_array_utils_include_path(elem_type), start=current_dir)


def _relative_streamutils_tb_include(elem_type: type[DataSchema], cfg: CodeGenConfig) -> str:
    tb_out_path = cfg.root_dir / _array_utils_tb_include_path(elem_type)
    util_path = cfg.root_dir / cfg.util_dir / "streamutils_tb.h"
    include_path = os.path.relpath(util_path, start=tb_out_path.parent)
    return include_path.replace("\\", "/")


def _relative_include_for_elem(elem_type: type[DataSchema]) -> str | None:
    if not elem_type.can_gen_include:
        return None
    current_dir = posixpath.dirname(_array_utils_include_path(elem_type)) or "."
    return posixpath.relpath(elem_type.include_path(), start=current_dir)


def _relative_streamutils_include(elem_type: type[DataSchema], cfg: CodeGenConfig) -> str:
    out_path = cfg.root_dir / _array_utils_include_path(elem_type)
    util_path = cfg.root_dir / cfg.util_dir / "streamutils_hls.h"
    include_path = os.path.relpath(util_path, start=out_path.parent)
    return include_path.replace("\\", "/")


def _needs_streamutils_include(elem_type: type[DataSchema]) -> bool:
    if elem_type.can_gen_include:
        return False
    read_expr = elem_type.from_uint_expr("packed_bits")
    write_expr = elem_type.to_uint_value_expr("value")
    return "streamutils::" in read_expr or "streamutils::" in write_expr


def _get_read_recursive_lines(
    elem_type: type[DataSchema],
    word_bw: int,
    dst_expr: str,
    source_expr: str,
) -> list[str]:
    prefix = ""
    member_name: str | None = dst_expr
    if issubclass(elem_type, DataList):
        prefix = f"{dst_expr}."
        member_name = None

    kwargs = {
        "word_bw": word_bw,
        "src_type": "array",
        "source": source_expr,
        "ipos0": 0,
        "iword0": 0,
        "prefix": prefix,
        "member_name": member_name,
    }

    method = getattr(elem_type, "gen_read_recursive", None)
    if callable(method):
        result = method(**kwargs)
    else:
        result = elem_type._gen_read_recursive(**kwargs)

    if isinstance(result, tuple):
        lines = result[0]
    else:
        lines = result

    return [str(line) for line in lines]


def _get_write_recursive_lines(
    elem_type: type[DataSchema],
    word_bw: int,
    src_expr: str,
    target_expr: str,
) -> list[str]:
    prefix = ""
    member_name: str | None = src_expr
    if issubclass(elem_type, DataList):
        prefix = f"{src_expr}."
        member_name = None

    result = elem_type._gen_write_recursive(
        word_bw=word_bw,
        dst_type="array",
        target=target_expr,
        ipos0=0,
        iword0=0,
        prefix=prefix,
        member_name=member_name,
    )
    lines = result[0] if isinstance(result, tuple) else result
    return [str(line) for line in lines]


def _gen_reader_body(elem_type: type[DataSchema], word_bw: int, indent_level: int = 1) -> str:
    indent = elem_type._get_indent(indent_level)
    i1 = elem_type._get_indent(indent_level + 1)
    i2 = elem_type._get_indent(indent_level + 2)
    i3 = elem_type._get_indent(indent_level + 3)

    elem_bw = elem_type.get_bitwidth()
    pf = word_bw // elem_bw if elem_bw > 0 else 0
    words_per_elem = elem_type.nwords_per_inst(word_bw)
    elem_cpp = elem_type.cpp_class_name()
    assign_expr = elem_type.from_uint_expr("src[in_idx]")

    lines = [
        f"{indent}if (src == nullptr || dst == nullptr || len <= 0) {{",
        f"{i1}return;",
        f"{indent}}}",
        "",
        f"{indent}int in_idx = 0;",
    ]

    if pf >= 2:
        lines.extend([
            f"{indent}for (int i = 0; i < len; i += {pf}) {{",
            f"{i1}#pragma HLS PIPELINE II=1",
            f"{i1}ap_uint<{word_bw}> w = src[in_idx++];",
            f"{i1}for (int j = 0; j < {pf}; ++j) {{",
            f"{i2}#pragma HLS UNROLL",
            f"{i2}if (i + j < len) {{",
        ])
        for j in range(pf):
            lo = j * elem_bw
            hi = lo + elem_bw - 1
            cond = "if" if j == 0 else "else if"
            rhs_expr = elem_type.from_uint_expr(f"w.range({hi}, {lo})")
            lines.append(f"{i3}{cond} (j == {j}) {{")
            lines.append(f"{i3}    dst[i + j] = {rhs_expr};")
            lines.append(f"{i3}}}")
        lines.extend([
            f"{i2}}}",
            f"{i1}}}",
            f"{indent}}}",
        ])
        return "\n".join(lines)

    if elem_bw <= word_bw:
        lines.extend([
            f"{indent}for (int i = 0; i < len; ++i) {{",
            f"{i1}#pragma HLS PIPELINE II=1",
            f"{i1}dst[i] = {assign_expr};",
            f"{i1}++in_idx;",
            f"{indent}}}",
        ])
        return "\n".join(lines)

    lines.extend([
        f"{indent}constexpr int words_per_elem = {words_per_elem};",
        f"{indent}for (int i = 0; i < len; ++i) {{",
        f"{i1}#pragma HLS PIPELINE",
    ])
    recursive_lines = _get_read_recursive_lines(
        elem_type=elem_type,
        word_bw=word_bw,
        dst_expr="dst[i]",
        source_expr="(src + in_idx)",
    )
    for line in recursive_lines:
        stripped = line[4:] if line.startswith("    ") else line
        lines.append(f"{i1}{stripped}" if stripped else "")
    lines.extend([
        f"{i1}in_idx += words_per_elem;",
        f"{indent}}}",
    ])
    return "\n".join(lines)


def _gen_writer_body(elem_type: type[DataSchema], word_bw: int, indent_level: int = 1) -> str:
    indent = elem_type._get_indent(indent_level)
    i1 = elem_type._get_indent(indent_level + 1)
    i2 = elem_type._get_indent(indent_level + 2)
    i3 = elem_type._get_indent(indent_level + 3)

    elem_bw = elem_type.get_bitwidth()
    pf = word_bw // elem_bw if elem_bw > 0 else 0
    words_per_elem = elem_type.nwords_per_inst(word_bw)
    elem_uint_expr = elem_type.to_uint_value_expr("src[in_idx]")

    lines = [
        f"{indent}if (src == nullptr || dst == nullptr || len <= 0) {{",
        f"{i1}return;",
        f"{indent}}}",
        "",
        f"{indent}int out_idx = 0;",
    ]

    if pf >= 2:
        lines.extend([
            f"{indent}for (int i = 0; i < len; i += {pf}) {{",
            f"{i1}#pragma HLS PIPELINE II=1",
            f"{i1}ap_uint<{word_bw}> w = 0;",
            f"{i1}for (int j = 0; j < {pf}; ++j) {{",
            f"{i2}#pragma HLS UNROLL",
            f"{i2}if (i + j < len) {{",
        ])
        for j in range(pf):
            lo = j * elem_bw
            hi = lo + elem_bw - 1
            cond = "if" if j == 0 else "else if"
            rhs_expr = elem_type.to_uint_value_expr(f"src[i + {j}]")
            lines.append(f"{i3}{cond} (j == {j}) {{")
            lines.append(f"{i3}    w.range({hi}, {lo}) = {rhs_expr};")
            lines.append(f"{i3}}}")
        lines.extend([
            f"{i2}}}",
            f"{i1}}}",
            f"{i1}dst[out_idx++] = w;",
            f"{indent}}}",
        ])
        return "\n".join(lines)

    if elem_bw <= word_bw:
        lines.extend([
            f"{indent}for (int in_idx = 0; in_idx < len; ++in_idx) {{",
            f"{i1}#pragma HLS PIPELINE II=1",
            f"{i1}dst[out_idx++] = {elem_uint_expr};",
            f"{indent}}}",
        ])
        return "\n".join(lines)

    lines.extend([
        f"{indent}constexpr int words_per_elem = {words_per_elem};",
        f"{indent}for (int i = 0; i < len; ++i) {{",
        f"{i1}#pragma HLS PIPELINE",
    ])
    recursive_lines = _get_write_recursive_lines(
        elem_type=elem_type,
        word_bw=word_bw,
        src_expr="src[i]",
        target_expr="(dst + out_idx)",
    )
    for line in recursive_lines:
        stripped = line[4:] if line.startswith("    ") else line
        lines.append(f"{i1}{stripped}" if stripped else "")
    lines.extend([
        f"{i1}out_idx += words_per_elem;",
        f"{indent}}}",
    ])
    return "\n".join(lines)


def _gen_specialization(elem_type: type[DataSchema], word_bw: int, indent_level: int = 0) -> str:
    indent = elem_type._get_indent(indent_level)
    elem_cpp = elem_type.cpp_class_name()
    lines = [
        "/**",
        f" * @brief Read an array of {elem_cpp} values from packed {word_bw}-bit words.",
        " *",
        " * The packed input uses greedy LSB-first packing with no inter-element padding.",
        f" * This specialization is optimized for word_bw = {word_bw}.",
        " *",
        " * @param src Pointer to packed source words.",
        " * @param dst Pointer to the destination array.",
        " * @param len Number of elements to decode.",
        " */",
        f"{indent}template<>",
        f"{indent}inline void read_array<{word_bw}>(const ap_uint<{word_bw}>* src, value_type* dst, int len) {{",
        f"{indent}    #pragma HLS INLINE",
        _gen_reader_body(elem_type=elem_type, word_bw=word_bw, indent_level=1),
        f"{indent}}}",
    ]
    return "\n".join(lines)


def _gen_write_specialization(elem_type: type[DataSchema], word_bw: int, indent_level: int = 0) -> str:
    indent = elem_type._get_indent(indent_level)
    elem_cpp = elem_type.cpp_class_name()
    lines = [
        "/**",
        f" * @brief Write an array of {elem_cpp} values into packed {word_bw}-bit words.",
        " *",
        " * The packed output uses greedy LSB-first packing with no inter-element padding.",
        f" * This specialization is optimized for word_bw = {word_bw}.",
        " *",
        " * @param src Pointer to the source array.",
        " * @param dst Pointer to the packed destination words.",
        " * @param len Number of elements to encode.",
        " */",
        f"{indent}template<>",
        f"{indent}inline void write_array<{word_bw}>(const value_type* src, ap_uint<{word_bw}>* dst, int len) {{",
        f"{indent}    #pragma HLS INLINE",
        _gen_writer_body(elem_type=elem_type, word_bw=word_bw, indent_level=1),
        f"{indent}}}",
    ]
    return "\n".join(lines)


def _gen_stream_elem_helpers(
    elem_type: type[DataSchema],
    word_bw_supported: list[int],
    indent_level: int = 0,
) -> str:
    indent = elem_type._get_indent(indent_level)
    i1 = elem_type._get_indent(indent_level + 1)
    i2 = elem_type._get_indent(indent_level + 2)
    i3 = elem_type._get_indent(indent_level + 3)

    elem_bw = elem_type.get_bitwidth()
    elem_cpp = elem_type.cpp_class_name()

    lines = [
        "template<int word_bw>",
        f"{indent}static constexpr int pf() {{",
        f"{i1}return word_bw / {elem_bw};",
        f"{indent}}}",
        "",
        "template<int word_bw>",
        f"{indent}inline void read_array_elem(const ap_uint<word_bw>* src, value_type out[pf<word_bw>()], int n = pf<word_bw>()) {{",
        f"{i1}#pragma HLS INLINE",
    ]

    for idx, bw in enumerate(word_bw_supported):
        pfv = bw // elem_bw if elem_bw > 0 else 0
        kw = "if" if idx == 0 else "else if"
        lines.append(f"{i1}{kw} constexpr (word_bw == {bw}) {{")
        if pfv >= 2:
            lines.append(f"{i2}if (src == nullptr) {{")
            lines.append(f"{i3}return;")
            lines.append(f"{i2}}}")
            lines.append(f"{i2}ap_uint<{bw}> w = src[0];")
            for j in range(pfv):
                lo = j * elem_bw
                hi = lo + elem_bw - 1
                rhs_expr = elem_type.from_uint_expr(f"w.range({hi}, {lo})")
                lines.append(f"{i2}if (n > {j}) {{")
                lines.append(f"{i3}out[{j}] = {rhs_expr};")
                lines.append(f"{i2}}}")
        else:
            if elem_bw <= bw:
                lines.append(f"{i2}if (n > 0 && src != nullptr) {{")
                lines.append(f"{i3}out[0] = {elem_type.from_uint_expr('src[0]')};")
                lines.append(f"{i2}}}")
            else:
                lines.append(f"{i2}if (n > 0 && src != nullptr) {{")
                recursive_lines = _get_read_recursive_lines(
                    elem_type=elem_type,
                    word_bw=bw,
                    dst_expr="out[0]",
                    source_expr="src",
                )
                for line in recursive_lines:
                    stripped = line[4:] if line.startswith("    ") else line
                    lines.append(f"{i3}{stripped}" if stripped else "")
                lines.append(f"{i2}}}")
        lines.append(f"{i1}}}")

    lines.extend([
        f"{i1}else {{",
        f'{i2}static_assert(word_bw > 0, "Unsupported word_bw for read_array_elem");',
        f"{i1}}}",
        f"{indent}}}",
        "",
        "template<int word_bw>",
        f"{indent}inline void write_array_elem(const value_type in[pf<word_bw>()], ap_uint<word_bw>* dst, int n = pf<word_bw>()) {{",
        f"{i1}#pragma HLS INLINE",
    ])

    for idx, bw in enumerate(word_bw_supported):
        pfv = bw // elem_bw if elem_bw > 0 else 0
        kw = "if" if idx == 0 else "else if"
        lines.append(f"{i1}{kw} constexpr (word_bw == {bw}) {{")
        if pfv >= 2:
            lines.append(f"{i2}if (dst == nullptr) {{")
            lines.append(f"{i3}return;")
            lines.append(f"{i2}}}")
            lines.append(f"{i2}ap_uint<{bw}> w = 0;")
            for j in range(pfv):
                lo = j * elem_bw
                hi = lo + elem_bw - 1
                rhs_expr = elem_type.to_uint_value_expr(f"in[{j}]")
                lines.append(f"{i2}if (n > {j}) {{")
                lines.append(f"{i3}w.range({hi}, {lo}) = {rhs_expr};")
                lines.append(f"{i2}}}")
            lines.append(f"{i2}dst[0] = w;")
        else:
            if elem_bw <= bw:
                lines.append(f"{i2}if (n > 0 && dst != nullptr) {{")
                lines.append(f"{i3}dst[0] = {elem_type.to_uint_value_expr('in[0]')};")
                lines.append(f"{i2}}}")
            else:
                lines.append(f"{i2}if (n > 0 && dst != nullptr) {{")
                recursive_lines = _get_write_recursive_lines(
                    elem_type=elem_type,
                    word_bw=bw,
                    src_expr="in[0]",
                    target_expr="dst",
                )
                for line in recursive_lines:
                    stripped = line[4:] if line.startswith("    ") else line
                    lines.append(f"{i3}{stripped}" if stripped else "")
                lines.append(f"{i2}}}")
        lines.append(f"{i1}}}")

    lines.extend([
        f"{i1}else {{",
        f'{i2}static_assert(word_bw > 0, "Unsupported word_bw for write_array_elem");',
        f"{i1}}}",
        f"{indent}}}",
        "",
        "template<int word_bw>",
        f"{indent}inline void read_stream_elem(hls::stream<ap_uint<word_bw>>& s, value_type out[pf<word_bw>()], int n = pf<word_bw>()) {{",
        f"{i1}#pragma HLS INLINE",
    ])

    for idx, bw in enumerate(word_bw_supported):
        pfv = bw // elem_bw if elem_bw > 0 else 0
        kw = "if" if idx == 0 else "else if"
        lines.append(f"{i1}{kw} constexpr (word_bw == {bw}) {{")
        if pfv >= 2:
            lines.append(f"{i2}ap_uint<{bw}> w = s.read();")
            for j in range(pfv):
                lo = j * elem_bw
                hi = lo + elem_bw - 1
                rhs_expr = elem_type.from_uint_expr(f"w.range({hi}, {lo})")
                lines.append(f"{i2}if (n > {j}) {{")
                lines.append(f"{i3}out[{j}] = {rhs_expr};")
                lines.append(f"{i2}}}")
        else:
            if elem_bw <= bw:
                lines.append(f"{i2}if (n > 0) {{")
                lines.append(f"{i3}ap_uint<{bw}> w = s.read();")
                lines.append(f"{i3}out[0] = {elem_type.from_uint_expr('w')};")
                lines.append(f"{i2}}}")
            else:
                lines.append(f"{i2}if (n > 0) {{")
                lines.append(f"{i3}out[0].template read_stream<{bw}>(s);")
                lines.append(f"{i2}}}")
        lines.append(f"{i1}}}")

    lines.extend([
        f"{i1}else {{",
        f'{i2}static_assert(word_bw > 0, "Unsupported word_bw for read_stream_elem");',
        f"{i1}}}",
        f"{indent}}}",
        "",
        "template<int word_bw>",
        f"{indent}inline void read_axi4_stream_elem(hls::stream<hls::axis<ap_uint<word_bw>, 0, 0, 0>>& s, value_type out[pf<word_bw>()], int n = pf<word_bw>()) {{",
        f"{i1}#pragma HLS INLINE",
    ])

    for idx, bw in enumerate(word_bw_supported):
        pfv = bw // elem_bw if elem_bw > 0 else 0
        kw = "if" if idx == 0 else "else if"
        lines.append(f"{i1}{kw} constexpr (word_bw == {bw}) {{")
        if pfv >= 2:
            lines.append(f"{i2}ap_uint<{bw}> w = s.read().data;")
            for j in range(pfv):
                lo = j * elem_bw
                hi = lo + elem_bw - 1
                rhs_expr = elem_type.from_uint_expr(f"w.range({hi}, {lo})")
                lines.append(f"{i2}if (n > {j}) {{")
                lines.append(f"{i3}out[{j}] = {rhs_expr};")
                lines.append(f"{i2}}}")
        else:
            if elem_bw <= bw:
                lines.append(f"{i2}if (n > 0) {{")
                lines.append(f"{i3}ap_uint<{bw}> w = s.read().data;")
                lines.append(f"{i3}out[0] = {elem_type.from_uint_expr('w')};")
                lines.append(f"{i2}}}")
            else:
                lines.append(f"{i2}if (n > 0) {{")
                lines.append(f"{i3}out[0].template read_axi4_stream<{bw}>(s);")
                lines.append(f"{i2}}}")
        lines.append(f"{i1}}}")

    lines.extend([
        f"{i1}else {{",
        f'{i2}static_assert(word_bw > 0, "Unsupported word_bw for read_axi4_stream_elem");',
        f"{i1}}}",
        f"{indent}}}",
        "",
        "template<int word_bw>",
        f"{indent}inline void write_stream_elem(hls::stream<ap_uint<word_bw>>& s, const value_type in[pf<word_bw>()], int n = pf<word_bw>()) {{",
        f"{i1}#pragma HLS INLINE",
    ])

    for idx, bw in enumerate(word_bw_supported):
        pfv = bw // elem_bw if elem_bw > 0 else 0
        kw = "if" if idx == 0 else "else if"
        lines.append(f"{i1}{kw} constexpr (word_bw == {bw}) {{")
        if pfv >= 2:
            lines.append(f"{i2}ap_uint<{bw}> w = 0;")
            for j in range(pfv):
                lo = j * elem_bw
                hi = lo + elem_bw - 1
                rhs_expr = elem_type.to_uint_value_expr(f"in[{j}]")
                lines.append(f"{i2}if (n > {j}) {{")
                lines.append(f"{i3}w.range({hi}, {lo}) = {rhs_expr};")
                lines.append(f"{i2}}}")
            lines.append(f"{i2}s.write(w);")
        else:
            if elem_bw <= bw:
                lines.append(f"{i2}if (n > 0) {{")
                lines.append(f"{i3}ap_uint<{bw}> w = {elem_type.to_uint_value_expr('in[0]')};")
                lines.append(f"{i3}s.write(w);")
                lines.append(f"{i2}}}")
            else:
                lines.append(f"{i2}if (n > 0) {{")
                lines.append(f"{i3}in[0].template write_stream<{bw}>(s);")
                lines.append(f"{i2}}}")
        lines.append(f"{i1}}}")

    lines.extend([
        f"{i1}else {{",
        f'{i2}static_assert(word_bw > 0, "Unsupported word_bw for write_stream_elem");',
        f"{i1}}}",
        f"{indent}}}",
        "",
        "template<int word_bw>",
        f"{indent}inline void write_axi4_stream_elem(hls::stream<hls::axis<ap_uint<word_bw>, 0, 0, 0>>& s, const value_type in[pf<word_bw>()], bool tlast = false, int n = pf<word_bw>()) {{",
        f"{i1}#pragma HLS INLINE",
    ])

    for idx, bw in enumerate(word_bw_supported):
        pfv = bw // elem_bw if elem_bw > 0 else 0
        kw = "if" if idx == 0 else "else if"
        lines.append(f"{i1}{kw} constexpr (word_bw == {bw}) {{")
        if pfv >= 2:
            lines.append(f"{i2}ap_uint<{bw}> w = 0;")
            for j in range(pfv):
                lo = j * elem_bw
                hi = lo + elem_bw - 1
                rhs_expr = elem_type.to_uint_value_expr(f"in[{j}]")
                lines.append(f"{i2}if (n > {j}) {{")
                lines.append(f"{i3}w.range({hi}, {lo}) = {rhs_expr};")
                lines.append(f"{i2}}}")
            lines.append(f"{i2}streamutils::write_axi4_word<{bw}>(s, w, tlast);")
        else:
            if elem_bw <= bw:
                lines.append(f"{i2}if (n > 0) {{")
                lines.append(f"{i3}ap_uint<{bw}> w = {elem_type.to_uint_value_expr('in[0]')};")
                lines.append(f"{i3}streamutils::write_axi4_word<{bw}>(s, w, tlast);")
                lines.append(f"{i2}}}")
            else:
                lines.append(f"{i2}if (n > 0) {{")
                lines.append(f"{i3}in[0].template write_axi4_stream<{bw}>(s, tlast);")
                lines.append(f"{i2}}}")
        lines.append(f"{i1}}}")

    lines.extend([
        f"{i1}else {{",
        f'{i2}static_assert(word_bw > 0, "Unsupported word_bw for write_axi4_stream_elem");',
        f"{i1}}}",
        f"{indent}}}",
        "",
        "template<int word_bw>",
        f"{indent}inline void read_stream(hls::stream<ap_uint<word_bw>>& s, value_type* dst, int len) {{",
        f"{i1}#pragma HLS INLINE",
        f"{i1}if (dst == nullptr || len <= 0) {{",
        f"{i2}return;",
        f"{i1}}}",
        f"{i1}for (int i = 0; i < len; i += pf<word_bw>()) {{",
        f"{i2}read_stream_elem<word_bw>(s, dst + i, len - i);",
        f"{i1}}}",
        f"{indent}}}",
        "",
        "template<int word_bw>",
        f"{indent}inline void read_axi4_stream(hls::stream<hls::axis<ap_uint<word_bw>, 0, 0, 0>>& s, value_type* dst, int len) {{",
        f"{i1}#pragma HLS INLINE",
        f"{i1}if (dst == nullptr || len <= 0) {{",
        f"{i2}return;",
        f"{i1}}}",
        f"{i1}for (int i = 0; i < len; i += pf<word_bw>()) {{",
        f"{i2}read_axi4_stream_elem<word_bw>(s, dst + i, len - i);",
        f"{i1}}}",
        f"{indent}}}",
        "",
        "template<int word_bw>",
        f"{indent}inline void write_stream(hls::stream<ap_uint<word_bw>>& s, const value_type* src, int len) {{",
        f"{i1}#pragma HLS INLINE",
        f"{i1}if (src == nullptr || len <= 0) {{",
        f"{i2}return;",
        f"{i1}}}",
        f"{i1}for (int i = 0; i < len; i += pf<word_bw>()) {{",
        f"{i2}write_stream_elem<word_bw>(s, src + i, len - i);",
        f"{i1}}}",
        f"{indent}}}",
        "",
        "template<int word_bw>",
        f"{indent}inline void write_axi4_stream(hls::stream<hls::axis<ap_uint<word_bw>, 0, 0, 0>>& s, const value_type* src, bool tlast = true, int len = pf<word_bw>()) {{",
        f"{i1}#pragma HLS INLINE",
        f"{i1}if (src == nullptr || len <= 0) {{",
        f"{i2}return;",
        f"{i1}}}",
        f"{i1}for (int i = 0; i < len; i += pf<word_bw>()) {{",
        f"{i2}const bool lane_tlast = (i + pf<word_bw>() >= len) ? tlast : false;",
        f"{i2}write_axi4_stream_elem<word_bw>(s, src + i, lane_tlast, len - i);",
        f"{i1}}}",
        f"{indent}}}",
    ])
    return "\n".join(lines)


def _gen_tb_helpers(elem_type: type[DataSchema], indent_level: int = 0) -> str:
    indent = elem_type._get_indent(indent_level)
    i1 = elem_type._get_indent(indent_level + 1)
    elem_bw = elem_type.get_bitwidth()

    lines = [
        "/**",
        " * @brief Read a packed uint32 binary file into an array of value_type.",
        " *",
        " * The input file must contain little-endian uint32 words produced by the",
        " * Python arrayutils.write_uint32_file helper.",
        " *",
        " * @param dst Pointer to the destination array.",
        " * @param file_path Path to the input binary file.",
        " * @param n0 Number of array elements to decode.",
        " */",
        f"{indent}inline void read_uint32_file_array(value_type* dst, const char* file_path, int n0) {{",
        f"{i1}if (n0 < 0) {{",
        f'{i1}    throw std::runtime_error("n0 must be non-negative.");',
        f"{i1}}}",
        f"{i1}std::ifstream ifs(file_path, std::ios::binary);",
        f"{i1}if (!ifs) {{",
        f'{i1}    throw std::runtime_error(std::string("Failed to open input file: ") + file_path);',
        f"{i1}}}",
        f"{i1}const int nwords = (n0 * {elem_bw} + 31) / 32;",
        f"{i1}std::vector<ap_uint<32>> words;",
        f"{i1}words.reserve(nwords);",
        f"{i1}for (int i = 0; i < nwords; ++i) {{",
        f"{i1}    words.push_back(streamutils::read_le_uint32(ifs));",
        f"{i1}}}",
        f"{i1}if (ifs.peek() != std::ifstream::traits_type::eof()) {{",
        f'{i1}    throw std::runtime_error(std::string("Unexpected trailing bytes in input file: ") + file_path);',
        f"{i1}}}",
        f"{i1}read_array<32>(words.empty() ? nullptr : words.data(), dst, n0);",
        f"{indent}}}",
        "",
        "/**",
        " * @brief Write an array of value_type to a packed uint32 binary file.",
        " *",
        " * The output file matches the little-endian uint32 format consumed by the",
        " * Python arrayutils.read_uint32_file helper.",
        " *",
        " * @param src Pointer to the source array.",
        " * @param file_path Path to the output binary file.",
        " * @param n0 Number of array elements to encode.",
        " */",
        f"{indent}inline void write_uint32_file_array(const value_type* src, const char* file_path, int n0) {{",
        f"{i1}if (n0 < 0) {{",
        f'{i1}    throw std::runtime_error("n0 must be non-negative.");',
        f"{i1}}}",
        f"{i1}std::ofstream ofs(file_path, std::ios::binary);",
        f"{i1}if (!ofs) {{",
        f'{i1}    throw std::runtime_error(std::string("Failed to open output file: ") + file_path);',
        f"{i1}}}",
        f"{i1}const int nwords = (n0 * {elem_bw} + 31) / 32;",
        f"{i1}std::vector<ap_uint<32>> words(nwords);",
        f"{i1}write_array<32>(src, words.empty() ? nullptr : words.data(), n0);",
        f"{i1}for (const auto& word : words) {{",
        f"{i1}    streamutils::write_le_uint32(ofs, static_cast<uint32_t>(word));",
        f"{i1}}}",
        f"{indent}}}",
    ]
    return "\n".join(lines)


def gen_array_utils(
    elem_type: type[DataSchema],
    word_bw_supported: list[int],
    cfg: CodeGenConfig | None = None,
) -> Path:
    """Generate a Vitis HLS header that reads and writes packed arrays of one element type.

    Parameters
    ----------
    elem_type : type[DataSchema]
        Element schema class to decode.
    word_bw_supported : list[int]
        Word widths to specialize in the generated header.
    cfg : CodeGenConfig | None, optional
        Output configuration. If omitted, uses ``CodeGenConfig()``.

    Returns
    -------
    pathlib.Path
        The generated header path.
    """
    if not isinstance(elem_type, type) or not issubclass(elem_type, DataSchema):
        raise TypeError("elem_type must be a DataSchema subclass.")

    if cfg is None:
        cfg = CodeGenConfig()

    widths = sorted({int(bw) for bw in word_bw_supported})
    if not widths:
        raise ValueError("word_bw_supported must contain at least one positive width.")

    for bw in widths:
        if bw <= 0:
            raise ValueError(f"word_bw values must be positive. Got {bw}.")

    out_path = cfg.root_dir / _array_utils_include_path(elem_type)
    tb_out_path = cfg.root_dir / _array_utils_tb_include_path(elem_type)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    elem_include = _relative_include_for_elem(elem_type)
    include_guard = _array_utils_include_guard(elem_type)
    tb_include_guard = _array_utils_tb_include_guard(elem_type)
    namespace = _array_utils_namespace(elem_type)
    elem_cpp = elem_type.cpp_class_name()

    lines = [
        f"#ifndef {include_guard}",
        f"#define {include_guard}",
        "",
        "#include <ap_int.h>",
        "#include <hls_stream.h>",
        "#if __has_include(<hls_axi_stream.h>)",
        "#include <hls_axi_stream.h>",
        "#else",
        "#include <ap_axi_sdata.h>",
        "#endif",
    ]

    if True:
        lines.append(f'#include "{_relative_streamutils_include(elem_type, cfg)}"')

    if elem_include is not None:
        lines.append(f'#include "{elem_include}"')

    lines.extend([
        "",
        f"namespace {namespace} {{",
        "",
        f"using value_type = {elem_cpp};",
        "",
        "template<int>",
        "struct unsupported_word_bw { static constexpr bool value = false; };",
        "",
        _gen_stream_elem_helpers(elem_type=elem_type, word_bw_supported=widths),
        "",
        "/**",
        f" * @brief Read an array of {elem_cpp} values from packed ap_uint words.",
        " *",
        " * Elements are unpacked greedily from least-significant bits first with no",
        " * padding between adjacent elements, matching the PySilicon DataSchema array",
        " * packing convention.",
        " *",
        " * @tparam word_bw Packed source word width in bits.",
        " * @param src Pointer to packed source words.",
        " * @param dst Pointer to the destination array.",
        " * @param len Number of elements to decode.",
        " */",
        "template<int word_bw>",
        "inline void read_array(const ap_uint<word_bw>* src, value_type* dst, int len) {",
        f"    static_assert(unsupported_word_bw<word_bw>::value, \"Unsupported word_bw for {namespace}::read_array\");",
        "    (void)src;",
        "    (void)dst;",
        "    (void)len;",
        "}",
        "",
        "/**",
        f" * @brief Write an array of {elem_cpp} values into packed ap_uint words.",
        " *",
        " * Elements are packed greedily from least-significant bits first with no",
        " * padding between adjacent elements, matching the PySilicon DataSchema array",
        " * packing convention.",
        " *",
        " * @tparam word_bw Packed destination word width in bits.",
        " * @param src Pointer to the source array.",
        " * @param dst Pointer to the packed destination words.",
        " * @param len Number of elements to encode.",
        " */",
        "template<int word_bw>",
        "inline void write_array(const value_type* src, ap_uint<word_bw>* dst, int len) {",
        f"    static_assert(unsupported_word_bw<word_bw>::value, \"Unsupported word_bw for {namespace}::write_array\");",
        "    (void)src;",
        "    (void)dst;",
        "    (void)len;",
        "}",
    ])

    for bw in widths:
        lines.extend([
            "",
            _gen_specialization(elem_type=elem_type, word_bw=bw),
            "",
            _gen_write_specialization(elem_type=elem_type, word_bw=bw),
        ])

    lines.extend([
        "",
        f"}}  // namespace {namespace}",
        "",
        f"#endif // {include_guard}",
    ])

    tb_lines = [
        f"#ifndef {tb_include_guard}",
        f"#define {tb_include_guard}",
        "",
        f'#include "{_relative_streamutils_tb_include(elem_type, cfg)}"',
        f'#include "{_relative_synth_include_from_tb(elem_type)}"',
        "",
        f"namespace {namespace} {{",
        "",
        _gen_tb_helpers(elem_type=elem_type),
        "",
        f"}}  // namespace {namespace}",
        "",
        f"#endif // {tb_include_guard}",
    ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tb_out_path.parent.mkdir(parents=True, exist_ok=True)
    tb_out_path.write_text("\n".join(tb_lines) + "\n", encoding="utf-8")
    return out_path