"""Infer constrained schema specs from Python symbols."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Annotated, Any, TypedDict, get_args, get_origin, get_type_hints, is_typeddict

from pysilicon.ai.schema_spec import normalize_module_spec, pascal_case, singularize, snake_case


@dataclass(frozen=True)
class IntHint:
    bitwidth: int = 32
    signed: bool = True
    description: str | None = None


@dataclass(frozen=True)
class FloatHint:
    bitwidth: int = 32
    description: str | None = None


@dataclass(frozen=True)
class ArrayHint:
    max_shape: tuple[int, ...]
    static: bool = True
    element_name: str | None = None
    type_name: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class EnumHint:
    bitwidth: int | None = None
    description: str | None = None


def load_python_symbol(module_path: str | Path, symbol_name: str) -> Any:
    module_path = Path(module_path)
    module_name = f"_pysilicon_input_{module_path.stem}_{abs(hash(str(module_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Python module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise AttributeError(f"Symbol '{symbol_name}' was not found in {module_path}.") from exc


def infer_schema_spec_from_symbol(symbol: Any, *, module_name: str | None = None) -> dict[str, Any]:
    if not isinstance(symbol, type):
        raise TypeError("infer_schema_spec_from_symbol expects a class symbol.")

    if dataclasses.is_dataclass(symbol):
        root = _infer_struct_from_dataclass(symbol)
    elif is_typeddict(symbol):
        root = _infer_struct_from_typeddict(symbol)
    else:
        raise TypeError(
            "Initial Python-type inference supports dataclasses and TypedDict classes only."
        )

    return normalize_module_spec(
        {
            "module_name": module_name or snake_case(symbol.__name__),
            "root": root,
        }
    )


def _infer_struct_from_dataclass(cls: type[Any], *, field_name: str | None = None) -> dict[str, Any]:
    hints = get_type_hints(cls, include_extras=True)
    fields: list[dict[str, Any]] = []
    for dataclass_field in dataclasses.fields(cls):
        annotation = hints[dataclass_field.name]
        metadata = dict(dataclass_field.metadata)
        fields.append(_infer_node(annotation, name=dataclass_field.name, metadata=metadata))
    return {
        "kind": "struct",
        "name": field_name,
        "type_name": cls.__name__,
        "fields": fields,
    }


def _infer_struct_from_typeddict(cls: type[TypedDict], *, field_name: str | None = None) -> dict[str, Any]:
    hints = get_type_hints(cls, include_extras=True)
    fields = [_infer_node(annotation, name=name, metadata={}) for name, annotation in hints.items()]
    return {
        "kind": "struct",
        "name": field_name,
        "type_name": cls.__name__,
        "fields": fields,
    }


def _infer_node(annotation: Any, *, name: str | None, metadata: dict[str, Any]) -> dict[str, Any]:
    annotation, extras = _unwrap_annotated(annotation)
    merged = dict(metadata)
    for extra in extras:
        merged.update(_hint_to_metadata(extra))

    origin = get_origin(annotation)
    if origin in {list, tuple}:
        array_meta = {key: merged.get(key) for key in ("max_shape", "static", "element_name", "type_name", "description")}
        if array_meta["max_shape"] is None:
            raise ValueError(f"Array field '{name}' must declare max_shape via ArrayHint or metadata.")
        element_annotation = get_args(annotation)[0]
        element_node = _infer_node(element_annotation, name=None, metadata={})
        return {
            "kind": "array",
            "name": name,
            "type_name": array_meta["type_name"] or pascal_case(f"{name or 'item'}_array"),
            "description": array_meta["description"],
            "max_shape": tuple(int(dim) for dim in array_meta["max_shape"]),
            "static": bool(array_meta.get("static", True)),
            "element_name": array_meta["element_name"] or singularize(name or "item"),
            "element": element_node,
        }

    if annotation is int:
        return {
            "kind": "int",
            "name": name,
            "description": merged.get("description"),
            "bitwidth": int(merged.get("bitwidth", 32)),
            "signed": bool(merged.get("signed", True)),
        }

    if annotation is float:
        return {
            "kind": "float",
            "name": name,
            "description": merged.get("description"),
            "bitwidth": int(merged.get("bitwidth", 32)),
        }

    if isinstance(annotation, type) and issubclass(annotation, IntEnum):
        return {
            "kind": "enum",
            "name": name,
            "description": merged.get("description"),
            "type_name": merged.get("type_name") or annotation.__name__,
            "bitwidth": merged.get("bitwidth"),
            "values": [{"name": member.name, "value": int(member.value)} for member in annotation],
        }

    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        struct_node = _infer_struct_from_dataclass(annotation, field_name=name)
        if merged.get("description") is not None:
            struct_node["description"] = merged["description"]
        return struct_node

    if isinstance(annotation, type) and is_typeddict(annotation):
        struct_node = _infer_struct_from_typeddict(annotation, field_name=name)
        if merged.get("description") is not None:
            struct_node["description"] = merged["description"]
        return struct_node

    raise TypeError(f"Unsupported annotation for field '{name}': {annotation!r}")


def _unwrap_annotated(annotation: Any) -> tuple[Any, list[Any]]:
    extras: list[Any] = []
    while get_origin(annotation) is Annotated:
        args = get_args(annotation)
        annotation = args[0]
        extras.extend(args[1:])
    return annotation, extras


def _hint_to_metadata(hint: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(hint):
        return {key: value for key, value in dataclasses.asdict(hint).items() if value is not None}
    if isinstance(hint, dict):
        return dict(hint)
    raise TypeError(f"Unsupported Annotated metadata payload: {hint!r}")
