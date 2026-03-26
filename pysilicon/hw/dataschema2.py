"""Experimental class-driven schema architecture.

This module is intentionally separate from dataschema.py. The key design shift is:

- schema structure lives on the class
- runtime values live on the instance

That allows structural code-generation APIs to operate directly on schema classes,
for example ``Instruction.gen_include()``.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import IntEnum
import math
import posixpath
from pathlib import Path, PurePosixPath
import re
from typing import Any, ClassVar

import numpy as np

from pysilicon.codegen.build import CodeGenConfig


class DataSchema(ABC):
    """Abstract base class for schema nodes.

    Subclasses define structural metadata at the class level. Instances hold only
    runtime values.
    """

    include_dir: ClassVar[str] = "."
    include_filename: ClassVar[str | None] = None
    cpp_repr: ClassVar[str | None] = None
    can_gen_include: ClassVar[bool] = True
    allowed_specialize_kwargs: ClassVar[set[str]] = {
        "include_dir",
        "include_filename",
        "cpp_repr",
    }

    @classmethod
    @abstractmethod
    def get_bitwidth(cls) -> int:
        """Return the packed hardware bitwidth for this schema node."""

    @classmethod
    def cpp_class_name(cls) -> str:
        """Return the C++ type name used for code generation."""
        return cls.cpp_repr or cls.__name__

    @classmethod
    @abstractmethod
    def init_value(cls) -> Any:
        """Return the initial Python-side runtime representation for this schema."""

    @classmethod
    def get_dependencies(cls) -> list[type[DataSchema]]:
        """Return generated-schema dependencies required by this schema header."""
        return []

    @staticmethod
    def _get_indent(level: int) -> str:
        """Return consistent indentation for generated C++ code."""
        return "    " * level

    @classmethod
    def to_uint_expr(cls, value_expr: str) -> str:
        """Return a C++ expression that packs a value into unsigned bits."""
        return f"{cls.cpp_class_name()}::pack_to_uint({value_expr})"

    @classmethod
    def to_uint_value_expr(cls, value_expr: str) -> str:
        """Return a C++ expression that packs a value expression to unsigned bits."""
        return cls.to_uint_expr(value_expr)

    def serialize(self, word_bw: int = 32) -> np.ndarray:
        """Serialize this runtime value into packed hardware words."""
        if word_bw <= 0:
            raise ValueError("word_bw must be positive.")

        words: list[int] = [0]
        final_ipos, final_iword = self._serialize_recursive(
            word_bw=word_bw,
            words=words,
            ipos0=0,
            iword0=0,
        )

        n_words = final_iword + (1 if final_ipos > 0 else 0)
        if n_words == 0:
            n_words = 1

        words = words[:n_words]

        if word_bw <= 32:
            return np.array(
                [np.uint32(word & ((1 << min(word_bw, 32)) - 1)) for word in words],
                dtype=np.uint32,
            )

        if word_bw <= 64:
            return np.array(
                [np.uint64(word & ((1 << word_bw) - 1)) for word in words],
                dtype=np.uint64,
            )

        chunks_per_word = math.ceil(word_bw / 64)
        out = np.zeros((n_words, chunks_per_word), dtype=np.uint64)
        mask64 = (1 << 64) - 1
        for row_idx, word in enumerate(words):
            for chunk_idx in range(chunks_per_word):
                out[row_idx, chunk_idx] = np.uint64((word >> (64 * chunk_idx)) & mask64)
        return out

    def _serialize_recursive(
        self,
        word_bw: int,
        words: list[int],
        ipos0: int = 0,
        iword0: int = 0,
    ) -> tuple[int, int]:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement serialization.")

    def deserialize(self, packed: np.ndarray, word_bw: int = 32) -> DataSchema:
        """Deserialize packed hardware words into this runtime value."""
        if word_bw <= 0:
            raise ValueError("word_bw must be positive.")

        arr = np.asarray(packed)
        words: list[int] = []

        if word_bw <= 64:
            if arr.ndim == 0:
                arr = arr.reshape(1)
            elif arr.ndim != 1:
                raise ValueError("For word_bw <= 64, packed must be a 1D array-like.")

            mask = (1 << word_bw) - 1
            words = [int(value) & mask for value in arr]
        else:
            chunks_per_word = math.ceil(word_bw / 64)
            if arr.ndim != 2:
                raise ValueError("For word_bw > 64, packed must be a 2D array-like.")
            if arr.shape[1] != chunks_per_word:
                raise ValueError(
                    f"For word_bw={word_bw}, packed must have shape (n_words, {chunks_per_word})."
                )

            mask = (1 << word_bw) - 1
            for row in arr:
                word = 0
                for chunk_idx, chunk in enumerate(row):
                    word |= int(np.uint64(chunk)) << (64 * chunk_idx)
                words.append(word & mask)

        if not words:
            words = [0]

        self._deserialize_recursive(
            word_bw=word_bw,
            words=words,
            ipos0=0,
            iword0=0,
        )
        return self

    def write_uint32_file(self, file_path: str | Path) -> Path:
        """Serialize this schema to 32-bit words and write them to a binary file."""
        out_path = Path(file_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        words = np.asarray(self.serialize(word_bw=32), dtype="<u4")
        words.tofile(out_path)
        return out_path

    def read_uint32_file(self, file_path: str | Path) -> DataSchema:
        """Read 32-bit packed words from a binary file and deserialize this schema."""
        in_path = Path(file_path)
        words = np.fromfile(in_path, dtype="<u4")
        return self.deserialize(words, word_bw=32)

    def _deserialize_recursive(
        self,
        word_bw: int,
        words: list[int],
        ipos0: int = 0,
        iword0: int = 0,
    ) -> tuple[int, int]:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement deserialization.")

    @classmethod
    def from_uint_expr(cls, uint_expr: str) -> str:
        """Return a C++ expression that reconstructs a value from unsigned bits."""
        return f"{cls.cpp_class_name()}::unpack_from_uint({uint_expr})"

    @classmethod
    def gen_pack(cls, indent_level: int = 0) -> str:
        """Return the C++ pack helper emitted for this schema."""
        raise NotImplementedError(f"{cls.__name__} does not implement pack generation.")

    @classmethod
    def gen_unpack(cls, indent_level: int = 0) -> str:
        """Return the C++ unpack helper emitted for this schema."""
        raise NotImplementedError(f"{cls.__name__} does not implement unpack generation.")

    @classmethod
    def gen_write(
        cls,
        word_bw: int | None = None,
        dst_type: str = "array",
        word_bw_supported: list[int] | None = None,
        indent_level: int = 0,
    ) -> str:
        """Return C++ code that writes this schema to a hardware destination."""
        if word_bw_supported is None:
            if word_bw is None:
                raise ValueError("word_bw must be provided when word_bw_supported is omitted.")
            word_bw_supported = [word_bw]

        if not word_bw_supported:
            raise ValueError("word_bw_supported must contain at least one value.")

        for bw in word_bw_supported:
            if bw <= 0:
                raise ValueError(f"word_bw values must be positive. Got {bw}.")

        if dst_type not in {"array", "stream", "axi4_stream"}:
            raise ValueError(f"Unsupported dst_type: {dst_type}.")

        indent = cls._get_indent(indent_level)
        i1 = cls._get_indent(indent_level + 1)
        i2 = cls._get_indent(indent_level + 2)

        if dst_type == "array":
            signature = f"{indent}template<int word_bw>\n{indent}void write_array(ap_uint<word_bw> x[]) const {{"
            target = "x"
            unsupported_msg = "Unsupported word_bw for write_array"
        elif dst_type == "stream":
            signature = (
                f"{indent}template<int word_bw>\n"
                f"{indent}void write_stream(hls::stream<ap_uint<word_bw>> &s) const {{"
            )
            target = "s"
            unsupported_msg = "Unsupported word_bw for write_stream"
        else:
            signature = (
                f"{indent}template<int word_bw>\n"
                f"{indent}void write_axi4_stream("
                f"hls::stream<hls::axis<ap_uint<word_bw>, 0, 0, 0>> &s, bool tlast = true) const {{"
            )
            target = "s"
            unsupported_msg = "Unsupported word_bw for write_axi4_stream"

        lines = signature.splitlines()

        for idx, bw in enumerate(word_bw_supported):
            cond = "if constexpr" if idx == 0 else "else if constexpr"
            lines.append(f"{i1}{cond} (word_bw == {bw}) {{")

            if dst_type != "array":
                lines.append(f"{i2}ap_uint<{bw}> w = 0;")

            final_lines, final_ipos, _ = cls._gen_write_recursive(
                word_bw=bw,
                dst_type=dst_type,
                target=target,
                ipos0=0,
                iword0=0,
                prefix="this->",
            )

            if dst_type == "axi4_stream" and final_ipos == 0:
                marker = f"streamutils::write_axi4_word<{bw}>({target}, w, "
                for line_idx in range(len(final_lines) - 1, -1, -1):
                    if marker in final_lines[line_idx]:
                        final_lines[line_idx] = final_lines[line_idx].replace(
                            ", false);",
                            ", tlast);",
                        )
                        break

            for line in final_lines:
                if line.startswith("    "):
                    line = line[4:]
                lines.append(f"{i2}{line}" if line else "")

            if dst_type != "array" and final_ipos > 0:
                if dst_type == "stream":
                    lines.append(f"{i2}{target}.write(w);")
                else:
                    lines.append(f"{i2}streamutils::write_axi4_word<{bw}>({target}, w, tlast);")

            lines.append(f"{i1}}}")

        lines.extend([
            f"{i1}else {{",
            f"{i2}static_assert(word_bw > 0, \"{unsupported_msg}\");",
            f"{i1}}}",
            f"{indent}}}",
        ])
        return "\n".join(lines)

    @classmethod
    def _gen_write_recursive(
        cls,
        word_bw: int,
        dst_type: str = "array",
        target: str = "x",
        ipos0: int = 0,
        iword0: int = 0,
        prefix: str = "",
        member_name: str | None = None,
    ) -> tuple[list[str], int, int]:
        """Return recursive write statements and final packing state."""
        raise NotImplementedError(f"{cls.__name__} does not implement write generation.")

    @classmethod
    def gen_read(
        cls,
        word_bw: int | None = None,
        src_type: str = "array",
        word_bw_supported: list[int] | None = None,
        indent_level: int = 0,
    ) -> str:
        """Return C++ code that reads this schema from a hardware source."""
        if word_bw_supported is None:
            if word_bw is None:
                raise ValueError("word_bw must be provided when word_bw_supported is omitted.")
            word_bw_supported = [word_bw]

        if not word_bw_supported:
            raise ValueError("word_bw_supported must contain at least one value.")

        for bw in word_bw_supported:
            if bw <= 0:
                raise ValueError(f"word_bw values must be positive. Got {bw}.")

        if src_type not in {"array", "stream", "axi4_stream"}:
            raise ValueError(f"Unsupported src_type: {src_type}.")

        indent = cls._get_indent(indent_level)
        i1 = cls._get_indent(indent_level + 1)
        i2 = cls._get_indent(indent_level + 2)

        if src_type == "array":
            signature = f"{indent}template<int word_bw>\n{indent}void read_array(const ap_uint<word_bw> x[]) {{"
            source = "x"
            unsupported_msg = "Unsupported word_bw for read_array"
        elif src_type == "stream":
            signature = (
                f"{indent}template<int word_bw>\n"
                f"{indent}void read_stream(hls::stream<ap_uint<word_bw>> &s) {{"
            )
            source = "s"
            unsupported_msg = "Unsupported word_bw for read_stream"
        else:
            signature = (
                f"{indent}template<int word_bw>\n"
                f"{indent}void read_axi4_stream(" 
                f"hls::stream<hls::axis<ap_uint<word_bw>, 0, 0, 0>> &s) {{"
            )
            source = "s"
            unsupported_msg = "Unsupported word_bw for read_axi4_stream"

        lines = signature.splitlines()

        for idx, bw in enumerate(word_bw_supported):
            cond = "if constexpr" if idx == 0 else "else if constexpr"
            lines.append(f"{i1}{cond} (word_bw == {bw}) {{")

            if src_type in {"stream", "axi4_stream"}:
                lines.append(f"{i2}ap_uint<{bw}> w = 0;")

            final_lines, _, _ = cls._gen_read_recursive(
                word_bw=bw,
                src_type=src_type,
                source=source,
                ipos0=0,
                iword0=0,
                prefix="this->",
            )

            for line in final_lines:
                if line.startswith("    "):
                    line = line[4:]
                lines.append(f"{i2}{line}" if line else "")

            lines.append(f"{i1}}}")

        lines.extend([
            f"{i1}else {{",
            f"{i2}static_assert(word_bw > 0, \"{unsupported_msg}\");",
            f"{i1}}}",
            f"{indent}}}",
        ])
        return "\n".join(lines)

    @classmethod
    def _gen_read_recursive(
        cls,
        word_bw: int,
        src_type: str = "array",
        source: str = "x",
        ipos0: int = 0,
        iword0: int = 0,
        prefix: str = "",
        member_name: str | None = None,
    ) -> tuple[list[str], int, int]:
        """Return recursive read statements and final packing state."""
        raise NotImplementedError(f"{cls.__name__} does not implement read generation.")

    def is_close(
        self,
        other: DataSchema,
        rel_tol: float | None = None,
        abs_tol: float | None = 1e-8,
    ) -> bool:
        """Compare runtime values with another schema instance."""
        _ = rel_tol, abs_tol
        if not isinstance(other, self.__class__):
            return False
        return self.val == other.val

    @staticmethod
    def _camel_to_snake(name: str) -> str:
        """Convert a class name to a snake_case stem for generated filenames."""
        return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

    @classmethod
    def default_include_filename(cls) -> str:
        """Return the default generated include filename for this schema class."""
        return f"{cls._camel_to_snake(cls.__name__)}.h"

    @classmethod
    def resolved_include_filename(cls) -> str:
        """Return the configured include filename or the default derived name."""
        return cls.include_filename or cls.default_include_filename()

    @classmethod
    def include_path(cls) -> str:
        """Return the schema header path relative to the code-generation root."""
        include_dir = (cls.include_dir or ".").replace("\\", "/")
        include_root = PurePosixPath(include_dir)
        filename = cls.resolved_include_filename()
        if include_root.as_posix() == ".":
            return filename
        return f"{include_root.as_posix()}/{filename}"

    @classmethod
    def relative_include_path_to(cls, dependency: type[DataSchema]) -> str:
        """Return the include path from this schema header to a dependency header."""
        current_dir = posixpath.dirname(cls.include_path()) or "."
        return posixpath.relpath(dependency.include_path(), start=current_dir)

    @classmethod
    def include_guard(cls) -> str:
        """Return a deterministic include guard derived from the include path."""
        guard = re.sub(r"[^A-Za-z0-9]+", "_", cls.include_path()).strip("_").upper()
        return re.sub(r"_+", "_", guard)

    @classmethod
    def validate_specialize_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Validate structural overrides accepted by ``specialize()`` methods."""
        unknown = sorted(set(kwargs) - cls.allowed_specialize_kwargs)
        if unknown:
            allowed = ", ".join(sorted(cls.allowed_specialize_kwargs))
            unknown_str = ", ".join(unknown)
            raise TypeError(
                f"Unknown specialization keyword(s) for {cls.__name__}: {unknown_str}. "
                f"Allowed keys: {allowed}."
            )

        validated = dict(kwargs)
        if "include_dir" in validated and not isinstance(validated["include_dir"], str):
            raise TypeError("include_dir must be a string.")
        if "include_filename" in validated:
            include_filename = validated["include_filename"]
            if include_filename is not None and not isinstance(include_filename, str):
                raise TypeError("include_filename must be a string or None.")
        if "cpp_repr" in validated:
            cpp_repr = validated["cpp_repr"]
            if cpp_repr is not None and not isinstance(cpp_repr, str):
                raise TypeError("cpp_repr must be a string or None.")
        return validated

    @classmethod
    def merge_specialize_attrs(
        cls,
        base_attrs: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge validated structural overrides into subclass attributes."""
        merged = dict(base_attrs)
        merged.update(cls.validate_specialize_kwargs(kwargs))
        return merged

    @classmethod
    def _gen_include_decl(cls, word_bw_supported: list[int] | None = None) -> str:
        """Return the declaration body emitted inside a generated header."""
        raise NotImplementedError(f"{cls.__name__} does not implement generated includes.")

    @classmethod
    def gen_include(
        cls,
        cfg: CodeGenConfig | None = None,
        word_bw_supported: list[int] | None = None,
    ) -> Path:
        """Render and write the generated header for this schema class.

        Parameters
        ----------
        cfg : CodeGenConfig | None, optional
            Code-generation configuration describing the root output directory.
            If omitted, a default ``CodeGenConfig()`` is created, which uses the
            current working directory as ``root_dir``.
        word_bw_supported : list[int] | None, optional
            Optional list of supported interface word widths to emit into the
            generated header. When provided for ``DataList`` schemas, the header
            includes generated read helpers for each width.

        Returns
        -------
        pathlib.Path
            The written header path, computed as
            ``cfg.root_dir / cls.include_path()``.

        Notes
        -----
        Parent directories are created automatically if they do not already
        exist. If the destination header already exists, it is overwritten.
        """
        if not cls.can_gen_include:
            raise ValueError(f"{cls.__name__} does not support standalone include generation.")

        if cfg is None:
            cfg = CodeGenConfig()

        if word_bw_supported is None:
            word_bw_supported = []

        for bw in word_bw_supported:
            if bw <= 0:
                raise ValueError(f"word_bw values must be positive. Got {bw}.")

        out_path = cfg.root_dir / cls.include_path()
        streamutils_path = cfg.root_dir / cfg.util_dir / "streamutils.h"
        streamutils_include = os.path.relpath(streamutils_path, start=out_path.parent)
        streamutils_include = streamutils_include.replace("\\", "/")

        lines = [
            f"#ifndef {cls.include_guard()}",
            f"#define {cls.include_guard()}",
            "",
            "#include <ap_int.h>",
            "#include <cctype>",
            "#include <cstdlib>",
            "#include <fstream>",
            "#include <hls_stream.h>",
            "#include <iterator>",
            "#include <stdexcept>",
            "#include <string>",
            "#if __has_include(<hls_axi_stream.h>)",
            "#include <hls_axi_stream.h>",
            "#else",
            "#include <ap_axi_sdata.h>",
            "#endif",
            "#include <iostream>",
            f'#include "{streamutils_include}"',
            "",
        ]

        dependency_lines = [
            f'#include "{cls.relative_include_path_to(dependency)}"'
            for dependency in cls.get_dependencies()
        ]
        if dependency_lines:
            lines.extend(dependency_lines)
            lines.append("")

        lines.append(cls._gen_include_decl(word_bw_supported=word_bw_supported))
        lines.extend([
            "",
            f"#endif // {cls.include_guard()}",
        ])

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path


class DataField(DataSchema):
    """Base class for scalar fields.

    Structural properties such as bitwidth and C++ type are defined on the class.
    Instances hold the runtime value in ``self.val``.
    """

    bitwidth: ClassVar[int | None] = None
    cpp_type: ClassVar[str | None] = None

    def __init__(self, value: Any = None):
        self._val = self.__class__.init_value()
        if value is not None:
            self.val = value

    @property
    def val(self) -> Any:
        """Return the runtime value stored in this field instance."""
        return self._val

    @val.setter
    def val(self, value: Any) -> None:
        self._val = self._convert(value)

    def _convert(self, value: Any) -> Any:
        """Convert a runtime value into this field's Python representation."""
        return value

    def _value_to_field_bits(self, current_val: Any) -> int:
        """Convert the runtime value to a raw integer bit pattern."""
        return int(current_val)

    def _field_bits_to_value(self, field_bits: int) -> Any:
        """Convert a raw integer bit pattern to a runtime value."""
        return int(field_bits)

    @classmethod
    def to_uint_expr(cls, value_expr: str) -> str:
        return f"(ap_uint<{cls.get_bitwidth()}>)({value_expr})"

    @classmethod
    def to_uint_value_expr(cls, value_expr: str) -> str:
        return f"(ap_uint<{cls.get_bitwidth()}>)({value_expr})"

    @classmethod
    def from_uint_expr(cls, uint_expr: str) -> str:
        return f"({cls.cpp_class_name()})({uint_expr})"

    @classmethod
    def gen_pack(cls, indent_level: int = 0) -> str:
        _ = indent_level
        return ""

    @classmethod
    def gen_unpack(cls, indent_level: int = 0) -> str:
        _ = indent_level
        return ""

    @classmethod
    def _gen_write_recursive(
        cls,
        word_bw: int,
        dst_type: str = "array",
        target: str = "x",
        ipos0: int = 0,
        iword0: int = 0,
        prefix: str = "",
        member_name: str | None = None,
    ) -> tuple[list[str], int, int]:
        bitwidth = cls.get_bitwidth()
        if member_name is None:
            raise ValueError(f"{cls.__name__} write generation requires a member_name.")
        if bitwidth > word_bw:
            raise ValueError(
                f"Field '{member_name}' with bitwidth {bitwidth} cannot fit into word_bw={word_bw}."
            )

        lines: list[str] = []
        curr_ipos = ipos0
        curr_iword = iword0

        if curr_ipos + bitwidth > word_bw:
            if dst_type == "stream":
                lines.append(f"    {target}.write(w);")
                lines.append("    w = 0;")
            elif dst_type == "axi4_stream":
                lines.append(f"    streamutils::write_axi4_word<{word_bw}>({target}, w, false);")
                lines.append("    w = 0;")

            curr_iword += 1
            curr_ipos = 0

        lhs = "w" if dst_type != "array" else f"{target}[{curr_iword}]"
        val_expr = cls.to_uint_expr(f"{prefix}{member_name}")

        if curr_ipos == 0 and bitwidth == word_bw:
            lines.append(f"    {lhs} = {val_expr};")
        else:
            if dst_type == "array" and curr_ipos == 0:
                lines.append(f"    {target}[{curr_iword}] = 0;")

            high = curr_ipos + bitwidth - 1
            low = curr_ipos
            lines.append(f"    {lhs}.range({high}, {low}) = {val_expr};")

        curr_ipos += bitwidth

        if curr_ipos == word_bw:
            if dst_type == "stream":
                lines.append(f"    {target}.write(w);")
                lines.append("    w = 0;")
            elif dst_type == "axi4_stream":
                lines.append(f"    streamutils::write_axi4_word<{word_bw}>({target}, w, false);")
                lines.append("    w = 0;")

            curr_iword += 1
            curr_ipos = 0

        return lines, curr_ipos, curr_iword

    @classmethod
    def _gen_read_recursive(
        cls,
        word_bw: int,
        src_type: str = "array",
        source: str = "x",
        ipos0: int = 0,
        iword0: int = 0,
        prefix: str = "",
        member_name: str | None = None,
    ) -> tuple[list[str], int, int]:
        bitwidth = cls.get_bitwidth()
        if member_name is None:
            raise ValueError(f"{cls.__name__} read generation requires a member_name.")
        if bitwidth > word_bw:
            raise ValueError(
                f"Field '{member_name}' with bitwidth {bitwidth} cannot fit into word_bw={word_bw}."
            )

        lines: list[str] = []
        curr_ipos = ipos0
        curr_iword = iword0

        if curr_ipos + bitwidth > word_bw:
            curr_iword += 1
            curr_ipos = 0

        if src_type == "stream" and curr_ipos == 0:
            lines.append(f"    w = {source}.read();")
        elif src_type == "axi4_stream" and curr_ipos == 0:
            lines.append(f"    w = {source}.read().data;")

        word_expr = f"{source}[{curr_iword}]" if src_type == "array" else "w"

        if curr_ipos == 0 and bitwidth == word_bw:
            rhs_expr = word_expr
        else:
            high = curr_ipos + bitwidth - 1
            low = curr_ipos
            rhs_expr = f"{word_expr}.range({high}, {low})"

        assign_expr = cls.from_uint_expr(rhs_expr)
        lines.append(f"    {prefix}{member_name} = {assign_expr};")

        curr_ipos += bitwidth
        if curr_ipos == word_bw:
            curr_iword += 1
            curr_ipos = 0

        return lines, curr_ipos, curr_iword

    def _serialize_recursive(
        self,
        word_bw: int,
        words: list[int],
        ipos0: int = 0,
        iword0: int = 0,
    ) -> tuple[int, int]:
        bitwidth = self.__class__.get_bitwidth()
        if bitwidth > word_bw:
            raise ValueError(
                f"Field '{self.__class__.__name__}' with bitwidth {bitwidth} cannot fit into word_bw={word_bw}."
            )

        curr_ipos = ipos0
        curr_iword = iword0

        if curr_ipos + bitwidth > word_bw:
            curr_iword += 1
            curr_ipos = 0

        while len(words) <= curr_iword:
            words.append(0)

        current_val = self.val
        mask = (1 << bitwidth) - 1
        field_bits = self._value_to_field_bits(current_val) & mask
        words[curr_iword] |= field_bits << curr_ipos

        curr_ipos += bitwidth
        if curr_ipos == word_bw:
            curr_iword += 1
            curr_ipos = 0

        return curr_ipos, curr_iword

    def _deserialize_recursive(
        self,
        word_bw: int,
        words: list[int],
        ipos0: int = 0,
        iword0: int = 0,
    ) -> tuple[int, int]:
        bitwidth = self.__class__.get_bitwidth()
        if bitwidth > word_bw:
            raise ValueError(
                f"Field '{self.__class__.__name__}' with bitwidth {bitwidth} cannot fit into word_bw={word_bw}."
            )

        curr_ipos = ipos0
        curr_iword = iword0

        if curr_ipos + bitwidth > word_bw:
            curr_iword += 1
            curr_ipos = 0

        word = 0 if curr_iword >= len(words) else words[curr_iword]
        mask = (1 << bitwidth) - 1
        field_bits = (word >> curr_ipos) & mask
        self.val = self._field_bits_to_value(field_bits)

        curr_ipos += bitwidth
        if curr_ipos == word_bw:
            curr_iword += 1
            curr_ipos = 0

        return curr_ipos, curr_iword

    @classmethod
    def get_bitwidth(cls) -> int:
        if cls.bitwidth is None:
            raise TypeError(f"{cls.__name__} does not define a class-level bitwidth.")
        return cls.bitwidth

    @classmethod
    def cpp_class_name(cls) -> str:
        if cls.cpp_repr is not None:
            return cls.cpp_repr
        if cls.cpp_type is None:
            raise TypeError(f"{cls.__name__} does not define a class-level cpp_type.")
        return cls.cpp_type

    def is_close(
        self,
        other: DataSchema,
        rel_tol: float | None = None,
        abs_tol: float | None = 1e-8,
    ) -> bool:
        _ = rel_tol, abs_tol
        if not isinstance(other, self.__class__):
            return False

        v1 = self.val
        v2 = other.val

        if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
            return bool(np.array_equal(np.asarray(v1), np.asarray(v2)))

        return v1 == v2


class IntField(DataField):
    """
    Integer scalar field for arbitrary bitwidth and signedness.

    ``specialize()`` creates a new subclass whose structural properties are fixed at
    the class level. Those subclasses can then be used directly in schema
    definitions.
    """

    bitwidth: ClassVar[int] = 32
    signed: ClassVar[bool] = True
    cpp_type: ClassVar[str] = "ap_int<32>"
    can_gen_include: ClassVar[bool] = False
    _specializations: ClassVar[dict[tuple[Any, ...], type[IntField]]] = {}

    @classmethod
    def specialize(
        cls,
        bitwidth: int,
        signed: bool = True,
        **kwargs: Any,
    ) -> type[IntField]:
        """Return a cached specialized ``IntField`` subclass.

        Parameters
        ----------
        bitwidth : int
            Number of bits in the integer representation. Must be positive.
        signed : bool, default=True
            Whether the integer type is signed. If False, the specialized class
            represents an unsigned integer.
        **kwargs : Any
            Optional structural metadata overrides such as ``include_dir`` and
            ``include_filename``.

        Returns
        -------
        type[IntField]
            A specialized ``IntField`` subclass with class-level attributes such as
            ``bitwidth``, ``signed``, and ``cpp_type``.

        Notes
        -----
        Repeated calls with the same ``bitwidth`` and ``signed`` values return the
        same cached subclass.
        """
        if bitwidth <= 0:
            raise ValueError("bitwidth must be positive.")

        overrides = cls.validate_specialize_kwargs(kwargs)
        override_items = tuple(sorted(overrides.items()))

        key = (cls, int(bitwidth), bool(signed), override_items)
        cached = cls._specializations.get(key)
        if cached is not None:
            return cached

        sign_prefix = "Int" if signed else "UInt"
        cpp_type = f"ap_int<{bitwidth}>" if signed else f"ap_uint<{bitwidth}>"
        subclass_name = f"{sign_prefix}{bitwidth}"

        specialized_attrs = cls.merge_specialize_attrs(
            {
                "bitwidth": int(bitwidth),
                "signed": bool(signed),
                "cpp_type": cpp_type,
                "__module__": cls.__module__,
                "__doc__": (
                    f"Specialized integer field: bitwidth={bitwidth}, signed={signed}."
                ),
            },
            overrides,
        )
        specialized = type(subclass_name, (cls,), specialized_attrs)
        cls._specializations[key] = specialized
        return specialized

    def _convert(self, value: Any) -> Any:
        if isinstance(value, (float, np.floating)) and not float(value).is_integer():
            raise ValueError(
                f"Cannot assign non-integer value {value!r} to {self.__class__.__name__}."
            )

        bitwidth = self.__class__.get_bitwidth()
        signed = self.__class__.signed

        if bitwidth <= 64:
            mask = (1 << bitwidth) - 1
            wrapped = int(value) & mask

            if signed:
                sign_bit = 1 << (bitwidth - 1)
                if wrapped & sign_bit:
                    wrapped -= 1 << bitwidth
                return np.int32(wrapped) if bitwidth <= 32 else np.int64(wrapped)

            return np.uint32(wrapped) if bitwidth <= 32 else np.uint64(wrapped)

        num_words = math.ceil(bitwidth / 64)

        if isinstance(value, np.ndarray):
            arr = np.array(value, dtype=np.uint64)
            if arr.size != num_words:
                raise ValueError(
                    f"Array size mismatch for {self.__class__.__name__}: "
                    f"expected {num_words} words, got {arr.size}."
                )
            return arr

        if isinstance(value, (list, tuple)):
            arr = np.array(value, dtype=np.uint64)
            if arr.size != num_words:
                raise ValueError(
                    f"Array size mismatch for {self.__class__.__name__}: "
                    f"expected {num_words} words, got {arr.size}."
                )
            return arr

        mask = (1 << bitwidth) - 1
        wrapped = int(value) & mask
        arr = np.zeros(num_words, dtype=np.uint64)
        for idx in range(num_words):
            arr[idx] = np.uint64((wrapped >> (64 * idx)) & 0xFFFFFFFFFFFFFFFF)
        return arr

    @classmethod
    def init_value(cls) -> Any:
        """Return the initial Python-side integer representation for this field."""
        bitwidth = cls.get_bitwidth()
        if bitwidth > 64:
            num_words = math.ceil(bitwidth / 64)
            return np.zeros(num_words, dtype=np.uint64)

        if cls.signed:
            return np.int32(0) if bitwidth <= 32 else np.int64(0)

        return np.uint32(0) if bitwidth <= 32 else np.uint64(0)

    @classmethod
    def to_uint_expr(cls, value_expr: str) -> str:
        return value_expr

    @classmethod
    def to_uint_value_expr(cls, value_expr: str) -> str:
        return value_expr

    def _value_to_field_bits(self, current_val: Any) -> int:
        bitwidth = self.__class__.get_bitwidth()
        if bitwidth > 64 and isinstance(current_val, np.ndarray):
            field_bits = 0
            for idx, word in enumerate(current_val.astype(np.uint64)):
                field_bits |= int(word) << (64 * idx)
            return field_bits
        return int(current_val)


class FloatField(DataField):
    """Floating-point scalar field specialized by bitwidth."""

    bitwidth: ClassVar[int] = 32
    cpp_type: ClassVar[str] = "float"
    can_gen_include: ClassVar[bool] = False
    _specializations: ClassVar[dict[tuple[Any, ...], type[FloatField]]] = {}

    @classmethod
    def specialize(cls, bitwidth: int = 32, **kwargs: Any) -> type[FloatField]:
        """Return a cached specialized ``FloatField`` subclass.

        Parameters
        ----------
        bitwidth : int, default=32
            Floating-point bitwidth. Supported values are ``32`` and ``64``.
        **kwargs : Any
            Optional structural metadata overrides such as ``include_dir`` and
            ``include_filename``.

        Returns
        -------
        type[FloatField]
            A specialized ``FloatField`` subclass with class-level attributes such
            as ``bitwidth`` and ``cpp_type``.

        Notes
        -----
        Repeated calls with the same ``bitwidth`` return the same cached subclass.
        """
        if bitwidth not in (32, 64):
            raise ValueError("FloatField only supports 32 or 64 bit widths.")

        overrides = cls.validate_specialize_kwargs(kwargs)
        override_items = tuple(sorted(overrides.items()))

        key = (cls, int(bitwidth), override_items)
        cached = cls._specializations.get(key)
        if cached is not None:
            return cached

        cpp_type = "double" if bitwidth == 64 else "float"
        subclass_name = f"Float{bitwidth}"
        specialized_attrs = cls.merge_specialize_attrs(
            {
                "bitwidth": int(bitwidth),
                "cpp_type": cpp_type,
                "__module__": cls.__module__,
                "__doc__": f"Specialized floating-point field: bitwidth={bitwidth}.",
            },
            overrides,
        )
        specialized = type(subclass_name, (cls,), specialized_attrs)
        cls._specializations[key] = specialized
        return specialized

    def _convert(self, value: Any) -> Any:
        return np.float64(value) if self.__class__.get_bitwidth() == 64 else np.float32(value)

    @classmethod
    def init_value(cls) -> Any:
        """Return the initial Python-side floating-point representation."""
        return np.float64(0.0) if cls.get_bitwidth() == 64 else np.float32(0.0)

    @classmethod
    def to_uint_expr(cls, value_expr: str) -> str:
        return f"streamutils::float_to_uint({value_expr})"

    @classmethod
    def to_uint_value_expr(cls, value_expr: str) -> str:
        return f"streamutils::float_to_uint({value_expr})"

    @classmethod
    def from_uint_expr(cls, uint_expr: str) -> str:
        if cls.get_bitwidth() != 32:
            raise ValueError("FloatField unpack currently supports only bitwidth=32.")
        return f"streamutils::uint_to_float((uint32_t)({uint_expr}))"

    def _value_to_field_bits(self, current_val: Any) -> int:
        bitwidth = self.__class__.get_bitwidth()
        if bitwidth == 32:
            return int(np.asarray(np.float32(current_val), dtype=np.float32).view(np.uint32))
        if bitwidth == 64:
            return int(np.asarray(np.float64(current_val), dtype=np.float64).view(np.uint64))
        raise ValueError(f"Unsupported FloatField bitwidth={bitwidth} for serialization.")

    def _field_bits_to_value(self, field_bits: int) -> Any:
        bitwidth = self.__class__.get_bitwidth()
        if bitwidth == 32:
            return np.asarray(np.uint32(field_bits), dtype=np.uint32).view(np.float32).item()
        if bitwidth == 64:
            return np.asarray(np.uint64(field_bits), dtype=np.uint64).view(np.float64).item()
        raise ValueError(f"Unsupported FloatField bitwidth={bitwidth} for deserialization.")

    def is_close(
        self,
        other: DataSchema,
        rel_tol: float | None = None,
        abs_tol: float | None = 1e-8,
    ) -> bool:
        if not isinstance(other, FloatField):
            return False

        if self.__class__.get_bitwidth() != other.__class__.get_bitwidth():
            return False

        v1 = float(self.val)
        v2 = float(other.val)

        if rel_tol is None and abs_tol is None:
            return v1 == v2

        kwargs: dict[str, float] = {}
        if rel_tol is not None:
            kwargs["rtol"] = rel_tol
        if abs_tol is not None:
            kwargs["atol"] = abs_tol
        return bool(np.isclose(v1, v2, **kwargs))


class EnumField(DataField):
    """Enum scalar field specialized by IntEnum type and optional bitwidth/default."""

    enum_type: ClassVar[type[IntEnum] | None] = None
    default_member: ClassVar[IntEnum | None] = None
    bitwidth: ClassVar[int | None] = None
    cpp_type: ClassVar[str | None] = None
    _specializations: ClassVar[dict[tuple[Any, ...], type[EnumField]]] = {}

    @classmethod
    def get_dependencies(cls) -> list[type[DataSchema]]:
        return []

    @classmethod
    def specialize(
        cls,
        enum_type: type[IntEnum],
        bitwidth: int | None = None,
        default: IntEnum | None = None,
        **kwargs: Any,
    ) -> type[EnumField]:
        """Return a cached specialized ``EnumField`` subclass.

        Parameters
        ----------
        enum_type : type[IntEnum]
            Enum type used by this field specialization. The enum must derive from
            ``IntEnum`` and currently must have non-negative values.
        bitwidth : int | None, optional
            Storage bitwidth for the enum field. If omitted, the minimum bitwidth
            required to represent the enum values is used.
        default : IntEnum | None, optional
            Default enum member used by ``init_value()``. If omitted, the first
            enum member is used.
        **kwargs : Any
            Optional structural metadata overrides such as ``include_dir`` and
            ``include_filename``.

        Returns
        -------
        type[EnumField]
            A specialized ``EnumField`` subclass with class-level attributes such
            as ``enum_type``, ``bitwidth``, ``cpp_type``, and ``default_member``.

        Notes
        -----
        Repeated calls with the same ``enum_type``, resolved ``bitwidth``, and
        default member return the same cached subclass.
        """
        if not issubclass(enum_type, IntEnum):
            raise TypeError("EnumField requires enum_type to derive from IntEnum.")

        enum_values = [int(member.value) for member in enum_type]
        if any(value < 0 for value in enum_values):
            raise ValueError("EnumField currently supports only non-negative IntEnum values.")

        min_width = (max(enum_values).bit_length() or 1) if enum_values else 1
        resolved_bitwidth = min_width if bitwidth is None else int(bitwidth)
        if resolved_bitwidth < min_width:
            raise ValueError(
                f"bitwidth={resolved_bitwidth} is too small for enum {enum_type.__name__}; "
                f"needs at least {min_width} bits."
            )

        resolved_default = list(enum_type)[0] if default is None else enum_type(default)
        overrides = cls.validate_specialize_kwargs(kwargs)
        enum_defaults = {
            "cpp_repr": enum_type.__name__,
            "include_filename": f"{cls._camel_to_snake(enum_type.__name__)}.h",
        }
        resolved_overrides = dict(enum_defaults)
        resolved_overrides.update(overrides)
        override_items = tuple(sorted(overrides.items()))
        resolved_override_items = tuple(sorted(resolved_overrides.items()))
        key = (
            cls,
            enum_type,
            resolved_bitwidth,
            int(resolved_default.value),
            resolved_override_items,
        )
        cached = cls._specializations.get(key)
        if cached is not None:
            return cached

        subclass_name = f"{enum_type.__name__}EnumField"
        specialized_attrs = cls.merge_specialize_attrs(
            {
                "enum_type": enum_type,
                "default_member": resolved_default,
                "bitwidth": resolved_bitwidth,
                "cpp_type": enum_type.__name__,
                "__module__": cls.__module__,
                "__doc__": (
                    f"Specialized enum field: enum_type={enum_type.__name__}, "
                    f"bitwidth={resolved_bitwidth}."
                ),
            },
            resolved_overrides,
        )
        specialized = type(subclass_name, (cls,), specialized_attrs)
        cls._specializations[key] = specialized
        return specialized

    def _convert(self, value: Any) -> IntEnum:
        enum_type = self.__class__.enum_type
        if enum_type is None:
            raise TypeError(f"{self.__class__.__name__} does not define enum_type.")
        try:
            return enum_type(value)
        except ValueError as exc:
            raise ValueError(
                f"Value {value!r} is not a valid member of {enum_type.__name__}."
            ) from exc

    @classmethod
    def init_value(cls) -> IntEnum:
        """Return the initial enum member for this field."""
        if cls.default_member is None:
            raise TypeError(f"{cls.__name__} does not define a default enum member.")
        return cls.default_member

    @classmethod
    def to_uint_expr(cls, value_expr: str) -> str:
        return f"(ap_uint<{cls.get_bitwidth()}>)({value_expr})"

    @classmethod
    def to_uint_value_expr(cls, value_expr: str) -> str:
        return f"(ap_uint<{cls.get_bitwidth()}>)({value_expr})"

    @classmethod
    def from_uint_expr(cls, uint_expr: str) -> str:
        return f"({cls.cpp_class_name()})({uint_expr})"

    @classmethod
    def _gen_include_decl(cls, word_bw_supported: list[int] | None = None) -> str:
        enum_type = cls.enum_type
        if enum_type is None:
            raise TypeError(f"{cls.__name__} does not define enum_type.")
        _ = word_bw_supported
        lines = [f"enum class {cls.cpp_class_name()} {{"]
        for member in enum_type:
            lines.append(f"    {member.name} = {member.value},")
        lines.append("};")
        return "\n".join(lines)

    def is_close(
        self,
        other: DataSchema,
        rel_tol: float | None = None,
        abs_tol: float | None = 1e-8,
    ) -> bool:
        _ = rel_tol, abs_tol
        if not isinstance(other, EnumField):
            return False

        if self.__class__.enum_type is not other.__class__.enum_type:
            return False

        return self.val == other.val


class DataList(DataSchema):
    """Structured aggregate with schema defined entirely at the class level.

    ``elements`` maps member names either to ``DataSchema`` subclasses or to
    metadata dictionaries containing a ``schema`` entry and optional metadata
    such as ``description``. Instances create child runtime objects from that
    class-level declaration.
    """

    element_inline_comment_threshold: ClassVar[int] = 56
    elements: ClassVar[dict[str, type[DataSchema] | Mapping[str, Any]]] = {}

    def __init__(self, **values: Any):
        object.__setattr__(self, "_children", {})

        for name, schema_cls in self._iter_element_schemas():
            child = schema_cls()
            self._children[name] = child

        for name, value in values.items():
            setattr(self, name, value)

    @classmethod
    def _normalize_element_definition(cls, name: str) -> dict[str, type[DataSchema] | str | None]:
        raw_definition = cls.elements[name]

        if isinstance(raw_definition, Mapping):
            unknown = sorted(set(raw_definition) - {"schema", "description"})
            if unknown:
                unknown_str = ", ".join(unknown)
                raise TypeError(
                    f"{cls.__name__}.elements['{name}'] has unsupported metadata key(s): "
                    f"{unknown_str}. Allowed keys: description, schema."
                )
            if "schema" not in raw_definition:
                raise TypeError(
                    f"{cls.__name__}.elements['{name}'] metadata must define a 'schema' entry."
                )
            schema_cls = raw_definition["schema"]
            description = raw_definition.get("description")
        else:
            schema_cls = raw_definition
            description = None

        if not isinstance(schema_cls, type) or not issubclass(schema_cls, DataSchema):
            raise TypeError(
                f"{cls.__name__}.elements['{name}']['schema'] must be a DataSchema subclass."
            )

        if description is not None:
            if not isinstance(description, str):
                raise TypeError(
                    f"{cls.__name__}.elements['{name}']['description'] must be a string if provided."
                )
            description = description.strip() or None

        return {
            "schema": schema_cls,
            "description": description,
        }

    @classmethod
    def _iter_elements(cls) -> list[tuple[str, dict[str, type[DataSchema] | str | None]]]:
        items = list(cls.elements.items())
        normalized: list[tuple[str, dict[str, type[DataSchema] | str | None]]] = []
        for name, _ in items:
            if not isinstance(name, str):
                raise TypeError(f"{cls.__name__}.elements keys must be strings.")
            normalized.append((name, cls._normalize_element_definition(name)))
        return normalized

    @classmethod
    def _iter_element_schemas(cls) -> list[tuple[str, type[DataSchema]]]:
        return [
            (name, definition["schema"])
            for name, definition in cls._iter_elements()
        ]

    @classmethod
    def get_element_schema(cls, name: str) -> type[DataSchema]:
        """Return the schema class for a named element."""
        try:
            return cls._normalize_element_definition(name)["schema"]
        except KeyError as exc:
            raise KeyError(f"{cls.__name__}.elements has no entry named '{name}'.") from exc

    @classmethod
    def get_element_description(cls, name: str) -> str | None:
        """Return the normalized description for a named element."""
        try:
            return cls._normalize_element_definition(name)["description"]
        except KeyError as exc:
            raise KeyError(f"{cls.__name__}.elements has no entry named '{name}'.") from exc

    @classmethod
    def get_element_definition(cls, name: str) -> dict[str, type[DataSchema] | str | None]:
        """Return the normalized metadata dictionary for a named element."""
        try:
            return dict(cls._normalize_element_definition(name))
        except KeyError as exc:
            raise KeyError(f"{cls.__name__}.elements has no entry named '{name}'.") from exc

    @classmethod
    def _format_member_comment(cls, description: str | None) -> list[str]:
        if description is None:
            return []
        if len(description) <= cls.element_inline_comment_threshold:
            return [f"//INLINE// {description}"]
        return [f"// {description}"]

    @classmethod
    def get_bitwidth(cls) -> int:
        return sum(schema_cls.get_bitwidth() for _, schema_cls in cls._iter_element_schemas())

    @classmethod
    def get_dependencies(cls) -> list[type[DataSchema]]:
        deps: list[type[DataSchema]] = []
        seen: set[type[DataSchema]] = set()
        for _, schema_cls in cls._iter_element_schemas():
            if not schema_cls.can_gen_include or schema_cls is cls or schema_cls in seen:
                continue
            seen.add(schema_cls)
            deps.append(schema_cls)
        return deps

    @classmethod
    def gen_pack(cls, indent_level: int = 0) -> str:
        indent = cls._get_indent(indent_level)
        inner_indent = cls._get_indent(indent_level + 1)

        lines = [
            f"{indent}static ap_uint<bitwidth> pack_to_uint(const {cls.cpp_class_name()}& data) {{",
            f"{inner_indent}ap_uint<bitwidth> res = 0;",
        ]

        current_lsb = 0
        for name, schema_cls in cls._iter_element_schemas():
            width = schema_cls.get_bitwidth()
            high = current_lsb + width - 1
            low = current_lsb
            expr = schema_cls.to_uint_expr(f"data.{name}")
            lines.append(f"{inner_indent}res.range({high}, {low}) = {expr};")
            current_lsb += width

        lines.append(f"{inner_indent}return res;")
        lines.append(f"{indent}}}")
        return "\n".join(lines)

    @classmethod
    def gen_unpack(cls, indent_level: int = 0) -> str:
        indent = cls._get_indent(indent_level)
        inner_indent = cls._get_indent(indent_level + 1)

        lines = [
            f"{indent}static {cls.cpp_class_name()} unpack_from_uint(const ap_uint<bitwidth>& packed) {{",
            f"{inner_indent}{cls.cpp_class_name()} data;",
        ]

        current_lsb = 0
        for name, schema_cls in cls._iter_element_schemas():
            width = schema_cls.get_bitwidth()
            high = current_lsb + width - 1
            low = current_lsb
            rhs_expr = schema_cls.from_uint_expr(f"packed.range({high}, {low})")
            lines.append(f"{inner_indent}data.{name} = {rhs_expr};")
            current_lsb += width

        lines.append(f"{inner_indent}return data;")
        lines.append(f"{indent}}}")
        return "\n".join(lines)

    @classmethod
    def _gen_write_recursive(
        cls,
        word_bw: int,
        dst_type: str = "array",
        target: str = "x",
        ipos0: int = 0,
        iword0: int = 0,
        prefix: str = "",
        member_name: str | None = None,
    ) -> tuple[list[str], int, int]:
        _ = member_name
        lines: list[str] = []
        curr_ipos = ipos0
        curr_iword = iword0

        for name, schema_cls in cls._iter_element_schemas():
            if issubclass(schema_cls, DataList):
                elem_prefix = f"{prefix}{name}."
                elem_member_name = None
            else:
                elem_prefix = prefix
                elem_member_name = name

            elem_lines, curr_ipos, curr_iword = schema_cls._gen_write_recursive(
                word_bw=word_bw,
                dst_type=dst_type,
                target=target,
                ipos0=curr_ipos,
                iword0=curr_iword,
                prefix=elem_prefix,
                member_name=elem_member_name,
            )
            lines.extend(elem_lines)

        return lines, curr_ipos, curr_iword

    @classmethod
    def _gen_read_recursive(
        cls,
        word_bw: int,
        src_type: str = "array",
        source: str = "x",
        ipos0: int = 0,
        iword0: int = 0,
        prefix: str = "",
        member_name: str | None = None,
    ) -> tuple[list[str], int, int]:
        _ = member_name
        lines: list[str] = []
        curr_ipos = ipos0
        curr_iword = iword0

        for name, schema_cls in cls._iter_element_schemas():
            if issubclass(schema_cls, DataList):
                elem_prefix = f"{prefix}{name}."
                elem_member_name = None
            else:
                elem_prefix = prefix
                elem_member_name = name

            elem_lines, curr_ipos, curr_iword = schema_cls._gen_read_recursive(
                word_bw=word_bw,
                src_type=src_type,
                source=source,
                ipos0=curr_ipos,
                iword0=curr_iword,
                prefix=elem_prefix,
                member_name=elem_member_name,
            )
            lines.extend(elem_lines)

        return lines, curr_ipos, curr_iword

    @classmethod
    def _gen_include_decl(cls, word_bw_supported: list[int] | None = None) -> str:
        lines = [f"struct {cls.cpp_class_name()} {{"]
        for name, definition in cls._iter_elements():
            schema_cls = definition["schema"]
            description = definition["description"]
            comment_lines = cls._format_member_comment(description)
            if comment_lines and comment_lines[0].startswith("//INLINE// "):
                inline_comment = comment_lines[0][11:]
                lines.append(f"    {schema_cls.cpp_class_name()} {name};  // {inline_comment}")
            else:
                lines.extend(f"    {line}" for line in comment_lines)
                lines.append(f"    {schema_cls.cpp_class_name()} {name};")
        lines.append("")
        lines.append(f"    static constexpr int bitwidth = {cls.get_bitwidth()};")

        pack_decl = cls.gen_pack(indent_level=1)
        if pack_decl:
            lines.append("")
            lines.extend(pack_decl.splitlines())

        unpack_decl = cls.gen_unpack(indent_level=1)
        if unpack_decl:
            lines.append("")
            lines.extend(unpack_decl.splitlines())

        if word_bw_supported:
            for dst_type in ("array", "stream", "axi4_stream"):
                lines.append("")
                lines.extend(
                    cls.gen_write(
                        dst_type=dst_type,
                        word_bw_supported=word_bw_supported,
                        indent_level=1,
                    ).splitlines()
                )

            for src_type in ("array", "stream", "axi4_stream"):
                lines.append("")
                lines.extend(
                    cls.gen_read(
                        src_type=src_type,
                        word_bw_supported=word_bw_supported,
                        indent_level=1,
                    ).splitlines()
                )

        lines.append("};")
        return "\n".join(lines)

    @classmethod
    def init_value(cls) -> dict[str, Any]:
        """Return the initial nested Python representation for this aggregate."""
        return {name: schema_cls.init_value() for name, schema_cls in cls._iter_element_schemas()}

    def _serialize_recursive(
        self,
        word_bw: int,
        words: list[int],
        ipos0: int = 0,
        iword0: int = 0,
    ) -> tuple[int, int]:
        curr_ipos = ipos0
        curr_iword = iword0

        for name, _ in self.__class__._iter_element_schemas():
            curr_ipos, curr_iword = self._children[name]._serialize_recursive(
                word_bw=word_bw,
                words=words,
                ipos0=curr_ipos,
                iword0=curr_iword,
            )

        return curr_ipos, curr_iword

    def _deserialize_recursive(
        self,
        word_bw: int,
        words: list[int],
        ipos0: int = 0,
        iword0: int = 0,
    ) -> tuple[int, int]:
        curr_ipos = ipos0
        curr_iword = iword0

        for name, _ in self.__class__._iter_element_schemas():
            curr_ipos, curr_iword = self._children[name]._deserialize_recursive(
                word_bw=word_bw,
                words=words,
                ipos0=curr_ipos,
                iword0=curr_iword,
            )

        return curr_ipos, curr_iword

    @property
    def val(self) -> dict[str, Any]:
        """Return a nested runtime snapshot of child values."""
        out: dict[str, Any] = {}
        for name, child in self._children.items():
            out[name] = child.val
        return out

    @val.setter
    def val(self, values: Mapping[str, Any]) -> None:
        if not isinstance(values, Mapping):
            raise TypeError(f"{self.__class__.__name__}.val expects a mapping.")
        for name, value in values.items():
            setattr(self, name, value)

    def __getattr__(self, name: str) -> Any:
        children = self.__dict__.get("_children")
        if children is not None and name in children:
            child = children[name]
            if isinstance(child, DataField):
                return child.val
            if isinstance(child, DataList):
                return child
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        children = self.__dict__.get("_children")
        if children is not None and name in children:
            child = children[name]
            if isinstance(child, DataField):
                child.val = value
            elif isinstance(child, DataList):
                if isinstance(value, child.__class__):
                    child.val = value.val
                else:
                    child.val = value
            else:
                raise TypeError(f"Unsupported child schema type for attribute '{name}'.")
            return
        object.__setattr__(self, name, value)

    def is_close(
        self,
        other: DataSchema,
        rel_tol: float | None = None,
        abs_tol: float | None = 1e-8,
    ) -> bool:
        if not isinstance(other, self.__class__):
            return False

        if set(self._children) != set(other._children):
            return False

        for name, child in self._children.items():
            if not child.is_close(other._children[name], rel_tol=rel_tol, abs_tol=abs_tol):
                return False

        return True


__all__ = [
    "DataSchema",
    "DataField",
    "IntField",
    "FloatField",
    "EnumField",
    "DataList",
]