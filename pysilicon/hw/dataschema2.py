"""Experimental class-driven schema architecture.

This module is intentionally separate from dataschema.py. The key design shift is:

- schema structure lives on the class
- runtime values live on the instance

That allows structural code-generation APIs to operate directly on schema classes,
for example ``Instruction.gen_include()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import IntEnum
import math
import posixpath
from pathlib import PurePosixPath
import re
from typing import Any, ClassVar

import numpy as np


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
    def _gen_include_decl(cls) -> str:
        """Return the declaration body emitted inside a generated header."""
        raise NotImplementedError(f"{cls.__name__} does not implement generated includes.")

    @classmethod
    def gen_include(cls) -> str:
        """Return the generated header contents for this schema class."""
        if not cls.can_gen_include:
            raise ValueError(f"{cls.__name__} does not support standalone include generation.")

        lines = [
            f"#ifndef {cls.include_guard()}",
            f"#define {cls.include_guard()}",
            "",
        ]

        dependency_lines = [
            f'#include "{cls.relative_include_path_to(dependency)}"'
            for dependency in cls.get_dependencies()
        ]
        if dependency_lines:
            lines.extend(dependency_lines)
            lines.append("")

        lines.append(cls._gen_include_decl())
        lines.extend([
            "",
            f"#endif // {cls.include_guard()}",
        ])
        return "\n".join(lines)


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
    def _gen_include_decl(cls) -> str:
        enum_type = cls.enum_type
        if enum_type is None:
            raise TypeError(f"{cls.__name__} does not define enum_type.")
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

    ``elements`` maps member names to ``DataSchema`` subclasses. Instances create
    child runtime objects from that class-level declaration.
    """

    elements: ClassVar[dict[str, type[DataSchema]]] = {}

    def __init__(self, **values: Any):
        object.__setattr__(self, "_children", {})

        for name, schema_cls in self._iter_elements():
            child = schema_cls()
            self._children[name] = child

        for name, value in values.items():
            setattr(self, name, value)

    @classmethod
    def _iter_elements(cls) -> list[tuple[str, type[DataSchema]]]:
        items = list(cls.elements.items())
        for name, schema_cls in items:
            if not isinstance(name, str):
                raise TypeError(f"{cls.__name__}.elements keys must be strings.")
            if not isinstance(schema_cls, type) or not issubclass(schema_cls, DataSchema):
                raise TypeError(
                    f"{cls.__name__}.elements['{name}'] must be a DataSchema subclass."
                )
        return items

    @classmethod
    def get_bitwidth(cls) -> int:
        return sum(schema_cls.get_bitwidth() for _, schema_cls in cls._iter_elements())

    @classmethod
    def get_dependencies(cls) -> list[type[DataSchema]]:
        deps: list[type[DataSchema]] = []
        seen: set[type[DataSchema]] = set()
        for _, schema_cls in cls._iter_elements():
            if not schema_cls.can_gen_include or schema_cls is cls or schema_cls in seen:
                continue
            seen.add(schema_cls)
            deps.append(schema_cls)
        return deps

    @classmethod
    def _gen_include_decl(cls) -> str:
        lines = [f"struct {cls.cpp_class_name()} {{"]
        for name, schema_cls in cls._iter_elements():
            lines.append(f"    {schema_cls.cpp_class_name()} {name};")
        lines.append("};")
        return "\n".join(lines)

    @classmethod
    def init_value(cls) -> dict[str, Any]:
        """Return the initial nested Python representation for this aggregate."""
        return {name: schema_cls.init_value() for name, schema_cls in cls._iter_elements()}

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