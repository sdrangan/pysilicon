"""Infer constrained schema specs and interface bundles from Python symbols."""

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import json
import sys
import types
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Annotated, Any, TypedDict, get_args, get_origin, get_type_hints, is_typeddict

import numpy as np

from pysilicon.ai.schema_spec import normalize_module_spec, pascal_case, singularize, snake_case

try:  # pragma: no cover - import is covered indirectly when pydantic is installed
    from pydantic import BaseModel
except ImportError:  # pragma: no cover - optional at runtime
    BaseModel = None


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
    module = _load_module_from_path(module_path)
    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise AttributeError(f"Symbol '{symbol_name}' was not found in {module_path}.") from exc


def infer_schema_spec_from_symbol(
    symbol: Any,
    *,
    module_name: str | None = None,
    root_type_name: str | None = None,
) -> dict[str, Any]:
    root = _infer_root_node_from_symbol(symbol, root_type_name=root_type_name)

    inferred_module_name = module_name or snake_case(root["type_name"])
    return normalize_module_spec(
        {
            "module_name": inferred_module_name,
            "root": root,
        }
    )


def infer_module_spec_from_annotation(
    annotation: Any,
    *,
    member_name: str,
    module_name: str | None = None,
    root_type_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    node = _infer_node(annotation, name=member_name, metadata=metadata or {})

    if node["kind"] in {"int", "float", "enum"}:
        root = {
            "kind": "struct",
            "type_name": root_type_name or pascal_case(member_name),
            "fields": [node],
        }
    else:
        root = dict(node)
        root["name"] = None
        if root_type_name is not None:
            root["type_name"] = root_type_name

    return normalize_module_spec(
        {
            "module_name": module_name or snake_case(root["type_name"]),
            "root": root,
        }
    )


def infer_interface_bundle_from_callable(
    symbol: Any,
    *,
    interface_name: str | None = None,
    description: str | None = None,
    assumptions: list[str] | None = None,
    word_bw_supported: list[int] | None = None,
) -> dict[str, Any]:
    if not callable(symbol):
        raise TypeError("infer_interface_bundle_from_callable expects a callable symbol.")

    func_name = getattr(symbol, "__name__", "generated_interface")
    interface_name = interface_name or pascal_case(func_name)
    signature = inspect.signature(symbol)
    hints = get_type_hints(symbol, include_extras=True)
    members: list[dict[str, Any]] = []

    for parameter in signature.parameters.values():
        if parameter.kind not in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            raise TypeError("Variadic callable parameters are not supported by the AI bundle inference layer.")

        annotation = hints.get(parameter.name, parameter.annotation)
        if annotation is inspect.Signature.empty:
            raise TypeError(f"Callable parameter '{parameter.name}' must have a type annotation.")

        spec = infer_module_spec_from_annotation(
            annotation,
            member_name=parameter.name,
            module_name=snake_case(f"{interface_name}_{parameter.name}"),
            root_type_name=_annotation_root_type_name(annotation, fallback=pascal_case(f"{interface_name}_{parameter.name}")),
        )
        members.append(
            {
                "name": parameter.name,
                "direction": "input",
                "role": _default_role_for_member(parameter.name, direction="input"),
                "source": {
                    "kind": "callable_parameter",
                    "callable_name": func_name,
                    "parameter_name": parameter.name,
                },
                "spec": spec,
            }
        )

    return_annotation = hints.get("return", signature.return_annotation)
    if return_annotation not in {inspect.Signature.empty, None}:
        used_names = {member["name"] for member in members}
        for index, annotation in enumerate(_split_return_annotations(return_annotation)):
            suggested_name = _annotation_member_name(annotation, index=index, fallback=f"output_{index}")
            member_name = _dedupe_member_name(suggested_name, used_names)
            used_names.add(member_name)
            spec = infer_module_spec_from_annotation(
                annotation,
                member_name=member_name,
                module_name=snake_case(f"{interface_name}_{member_name}"),
                root_type_name=_annotation_root_type_name(
                    annotation,
                    fallback=pascal_case(f"{interface_name}_{member_name}"),
                ),
            )
            members.append(
                {
                    "name": member_name,
                    "direction": "output",
                    "role": _default_role_for_member(member_name, direction="output"),
                    "source": {
                        "kind": "callable_return",
                        "callable_name": func_name,
                        "return_index": index,
                    },
                    "spec": spec,
                }
            )

    return {
        "interface_name": interface_name,
        "description": description,
        "assumptions": list(assumptions or []),
        "word_bw_supported": list(word_bw_supported or [32, 64]),
        "source": {
            "kind": "callable",
            "callable_name": func_name,
        },
        "members": members,
    }


def _load_module_from_path(module_path: Path) -> types.ModuleType:
    if module_path.suffix == ".ipynb":
        return _load_notebook_module(module_path)
    return _load_python_module(module_path)


def _load_python_module(module_path: Path) -> types.ModuleType:
    module_name = f"_pysilicon_input_{module_path.stem}_{abs(hash(str(module_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Python module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_notebook_module(notebook_path: Path) -> types.ModuleType:
    module_name = f"_pysilicon_input_{notebook_path.stem}_{abs(hash(str(notebook_path)))}"
    source = _notebook_to_python_source(notebook_path)
    module = types.ModuleType(module_name)
    module.__file__ = str(notebook_path)
    sys.modules[module_name] = module
    exec(compile(source, str(notebook_path), "exec"), module.__dict__)
    return module


def _notebook_to_python_source(notebook_path: Path) -> str:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", [])
        if isinstance(source, str):
            lines = source.splitlines(keepends=True)
        else:
            lines = list(source)

        cleaned_lines = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("%") or stripped.startswith("!"):
                continue
            cleaned_lines.append(line)

        chunk = "".join(cleaned_lines).strip()
        if chunk:
            chunks.append(chunk)

    if not chunks:
        raise ValueError(f"No executable code cells were found in notebook {notebook_path}.")
    return "\n\n".join(chunks) + "\n"


def _infer_root_node_from_symbol(symbol: Any, *, root_type_name: str | None = None) -> dict[str, Any]:
    if isinstance(symbol, type) and dataclasses.is_dataclass(symbol):
        return _infer_struct_from_dataclass(symbol, type_name=root_type_name or symbol.__name__)
    if isinstance(symbol, type) and is_typeddict(symbol):
        return _infer_struct_from_typeddict(symbol, type_name=root_type_name or symbol.__name__)
    if _is_pydantic_model_class(symbol):
        return _infer_struct_from_pydantic(symbol, type_name=root_type_name or symbol.__name__)
    if isinstance(symbol, np.dtype):
        return _infer_struct_from_dtype(symbol, type_name=root_type_name or pascal_case("structured_dtype"))

    raise TypeError(
        "AI symbol inference supports dataclasses, TypedDict classes, pydantic BaseModel classes, "
        "and NumPy structured dtypes."
    )


def _infer_struct_from_dataclass(
    cls: type[Any],
    *,
    field_name: str | None = None,
    type_name: str | None = None,
) -> dict[str, Any]:
    hints = get_type_hints(cls, include_extras=True)
    fields: list[dict[str, Any]] = []
    for dataclass_field in dataclasses.fields(cls):
        annotation = hints[dataclass_field.name]
        metadata = dict(dataclass_field.metadata)
        fields.append(_infer_node(annotation, name=dataclass_field.name, metadata=metadata))
    return {
        "kind": "struct",
        "name": field_name,
        "type_name": type_name or cls.__name__,
        "fields": fields,
    }


def _infer_struct_from_typeddict(
    cls: type[TypedDict],
    *,
    field_name: str | None = None,
    type_name: str | None = None,
) -> dict[str, Any]:
    hints = get_type_hints(cls, include_extras=True)
    fields = [_infer_node(annotation, name=name, metadata={}) for name, annotation in hints.items()]
    return {
        "kind": "struct",
        "name": field_name,
        "type_name": type_name or cls.__name__,
        "fields": fields,
    }


def _infer_struct_from_pydantic(
    cls: type[Any],
    *,
    field_name: str | None = None,
    type_name: str | None = None,
) -> dict[str, Any]:
    hints = get_type_hints(cls, include_extras=True)
    fields: list[dict[str, Any]] = []
    for name, field_info in cls.model_fields.items():
        annotation = hints.get(name, field_info.annotation)
        metadata: dict[str, Any] = {}
        if field_info.description:
            metadata["description"] = field_info.description
        fields.append(_infer_node(annotation, name=name, metadata=metadata))
    return {
        "kind": "struct",
        "name": field_name,
        "type_name": type_name or cls.__name__,
        "fields": fields,
    }


def _infer_struct_from_dtype(
    dtype: np.dtype[Any],
    *,
    field_name: str | None = None,
    type_name: str | None = None,
) -> dict[str, Any]:
    normalized_dtype = np.dtype(dtype)
    if normalized_dtype.fields is None:
        raise TypeError("NumPy dtype inference requires a structured dtype with named fields.")

    fields: list[dict[str, Any]] = []
    for name, field_info in normalized_dtype.fields.items():
        field_dtype = np.dtype(field_info[0])
        fields.append(_infer_dtype_node(field_dtype, name=name))

    return {
        "kind": "struct",
        "name": field_name,
        "type_name": type_name or pascal_case(field_name or "structured_dtype"),
        "fields": fields,
    }


def _infer_node(annotation: Any, *, name: str | None, metadata: dict[str, Any]) -> dict[str, Any]:
    annotation, extras = _unwrap_annotated(annotation)
    merged = dict(metadata)
    for extra in extras:
        merged.update(_hint_to_metadata(extra))

    origin = get_origin(annotation)
    if origin in {list, tuple}:
        array_meta = {
            key: merged.get(key)
            for key in ("max_shape", "static", "element_name", "type_name", "description")
        }
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

    if _is_pydantic_model_class(annotation):
        struct_node = _infer_struct_from_pydantic(annotation, field_name=name)
        if merged.get("description") is not None:
            struct_node["description"] = merged["description"]
        return struct_node

    if isinstance(annotation, np.dtype):
        struct_node = _infer_struct_from_dtype(annotation, field_name=name)
        if merged.get("description") is not None:
            struct_node["description"] = merged["description"]
        return struct_node

    raise TypeError(f"Unsupported annotation for field '{name}': {annotation!r}")


def _infer_dtype_node(dtype: np.dtype[Any], *, name: str | None) -> dict[str, Any]:
    normalized_dtype = np.dtype(dtype)
    if normalized_dtype.subdtype is not None:
        base_dtype, shape = normalized_dtype.subdtype
        return {
            "kind": "array",
            "name": name,
            "type_name": pascal_case(f"{name or 'item'}_array"),
            "description": None,
            "max_shape": tuple(int(dim) for dim in shape),
            "static": True,
            "element_name": singularize(name or "item"),
            "element": _infer_dtype_node(np.dtype(base_dtype), name=None),
        }

    if normalized_dtype.fields is not None:
        return _infer_struct_from_dtype(
            normalized_dtype,
            field_name=name,
            type_name=pascal_case(name or "nested_struct"),
        )

    if normalized_dtype.kind in {"i", "u"}:
        return {
            "kind": "int",
            "name": name,
            "description": None,
            "bitwidth": normalized_dtype.itemsize * 8,
            "signed": normalized_dtype.kind == "i",
        }

    if normalized_dtype.kind == "b":
        return {
            "kind": "int",
            "name": name,
            "description": None,
            "bitwidth": 1,
            "signed": False,
        }

    if normalized_dtype.kind == "f":
        bitwidth = normalized_dtype.itemsize * 8
        if bitwidth not in {32, 64}:
            raise TypeError(f"Unsupported NumPy float dtype for AI schema inference: {normalized_dtype}.")
        return {
            "kind": "float",
            "name": name,
            "description": None,
            "bitwidth": bitwidth,
        }

    raise TypeError(f"Unsupported NumPy dtype for AI schema inference: {normalized_dtype}.")


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


def _split_return_annotations(annotation: Any) -> list[Any]:
    raw_annotation, _ = _unwrap_annotated(annotation)
    origin = get_origin(raw_annotation)
    if origin is tuple:
        args = [arg for arg in get_args(raw_annotation) if arg is not Ellipsis]
        if args:
            return list(args)
    return [annotation]


def _annotation_root_type_name(annotation: Any, *, fallback: str) -> str:
    raw_annotation, extras = _unwrap_annotated(annotation)
    for extra in extras:
        try:
            metadata = _hint_to_metadata(extra)
        except TypeError:
            continue
        type_name = metadata.get("type_name")
        if isinstance(type_name, str):
            return type_name

    if isinstance(raw_annotation, type):
        return raw_annotation.__name__
    if isinstance(raw_annotation, np.dtype):
        return fallback
    return fallback


def _annotation_member_name(annotation: Any, *, index: int, fallback: str) -> str:
    raw_annotation, extras = _unwrap_annotated(annotation)
    for extra in extras:
        try:
            metadata = _hint_to_metadata(extra)
        except TypeError:
            continue
        type_name = metadata.get("type_name")
        if isinstance(type_name, str):
            return snake_case(type_name)

    if isinstance(raw_annotation, type):
        return snake_case(raw_annotation.__name__)
    return fallback


def _dedupe_member_name(name: str, used_names: set[str]) -> str:
    if name not in used_names:
        return name
    index = 2
    while f"{name}_{index}" in used_names:
        index += 1
    return f"{name}_{index}"


def _default_role_for_member(name: str, *, direction: str) -> str:
    lowered = name.lower()
    if direction == "input" and any(token in lowered for token in {"cmd", "cfg", "config", "command"}):
        return "config"
    if "state" in lowered:
        return "state"
    if any(token in lowered for token in {"resp", "response", "status"}):
        return "response"
    if any(token in lowered for token in {"sample", "samples", "frame", "stream", "data"}):
        return "stream"
    return direction


def _is_pydantic_model_class(symbol: Any) -> bool:
    return bool(BaseModel is not None and isinstance(symbol, type) and issubclass(symbol, BaseModel))
