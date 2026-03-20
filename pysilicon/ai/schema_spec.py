"""Constrained intermediate schema representation for AI-assisted codegen."""

from __future__ import annotations

import keyword
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


SCHEMA_NODE_KINDS = {"struct", "array", "int", "float", "enum"}
NAMED_NODE_KINDS = {"struct", "array", "enum"}


def snake_case(name: str) -> str:
    out: list[str] = []
    for index, char in enumerate(name):
        if char.isupper() and index > 0 and (not name[index - 1].isupper()):
            out.append("_")
        out.append(char.lower())
    return "".join(out)


def pascal_case(name: str) -> str:
    parts = [part for part in _split_name(name) if part]
    return "".join(part[:1].upper() + part[1:] for part in parts) or "SchemaType"


def singularize(name: str) -> str:
    if name.endswith("ies") and len(name) > 3:
        return name[:-3] + "y"
    if name.endswith("sses") and len(name) > 4:
        return name[:-2]
    if name.endswith("s") and not name.endswith("ss") and len(name) > 1:
        return name[:-1]
    return f"{name}_item" if name else "item"


def normalize_module_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize a dataschema module specification."""
    if not isinstance(spec, Mapping):
        raise TypeError("Schema spec must be a mapping.")

    if "root" not in spec:
        raise ValueError("Schema spec must contain a top-level 'root' node.")

    normalized_root = _normalize_node(spec["root"], path=["root"], context="root")
    if normalized_root["kind"] not in {"struct", "array"}:
        raise ValueError("Top-level root kind must be 'struct' or 'array'.")

    module_name = spec.get("module_name") or snake_case(normalized_root["type_name"])
    module_name = _require_identifier(module_name, label="module_name")

    word_bw_supported_raw = spec.get("word_bw_supported", [32, 64])
    if not isinstance(word_bw_supported_raw, Sequence) or isinstance(word_bw_supported_raw, (str, bytes)):
        raise TypeError("'word_bw_supported' must be a sequence of integers.")
    word_bw_supported = sorted({int(value) for value in word_bw_supported_raw})
    if not word_bw_supported or any(value <= 0 for value in word_bw_supported):
        raise ValueError("'word_bw_supported' must contain positive integers.")

    normalized = {
        "module_name": module_name,
        "word_bw_supported": word_bw_supported,
        "root": normalized_root,
    }

    _validate_named_type_reuse(normalized_root)
    return normalized


def collect_named_nodes(spec: Mapping[str, Any] | dict[str, Any]) -> list[dict[str, Any]]:
    """Return unique named nodes in dependency order."""
    normalized = normalize_module_spec(spec) if "root" in spec else dict(spec)
    seen: set[str] = set()
    ordered: list[dict[str, Any]] = []

    def visit(node: dict[str, Any]) -> None:
        kind = node["kind"]
        if kind == "struct":
            for child in node["fields"]:
                visit(child)
        elif kind == "array":
            visit(node["element"])

        if kind in NAMED_NODE_KINDS:
            type_name = node["type_name"]
            if type_name not in seen:
                ordered.append(node)
                seen.add(type_name)

    visit(normalized["root"])
    return ordered


def _normalize_node(raw_node: Any, *, path: list[str], context: str) -> dict[str, Any]:
    if not isinstance(raw_node, Mapping):
        raise TypeError(f"Node at {'.'.join(path)} must be a mapping.")

    raw_kind = raw_node.get("kind")
    if raw_kind not in SCHEMA_NODE_KINDS:
        raise ValueError(
            f"Node at {'.'.join(path)} must declare kind in {sorted(SCHEMA_NODE_KINDS)}."
        )

    kind = str(raw_kind)
    description = raw_node.get("description")
    if description is not None and not isinstance(description, str):
        raise TypeError(f"description at {'.'.join(path)} must be a string when provided.")

    name = raw_node.get("name")
    if name is not None:
        name = _require_identifier(name, label=f"{'.'.join(path)}.name")
    elif context == "struct_field":
        raise ValueError(f"Field node at {'.'.join(path)} must include a 'name'.")

    if kind == "int":
        bitwidth = int(raw_node.get("bitwidth", 32))
        if bitwidth <= 0:
            raise ValueError(f"bitwidth at {'.'.join(path)} must be positive.")
        signed = bool(raw_node.get("signed", True))
        return {
            "kind": "int",
            "name": name,
            "description": description,
            "bitwidth": bitwidth,
            "signed": signed,
        }

    if kind == "float":
        bitwidth = int(raw_node.get("bitwidth", 32))
        if bitwidth not in (32, 64):
            raise ValueError(f"Float node at {'.'.join(path)} must use bitwidth 32 or 64.")
        return {
            "kind": "float",
            "name": name,
            "description": description,
            "bitwidth": bitwidth,
        }

    type_name = raw_node.get("type_name")
    if type_name is None:
        if name is not None:
            suffix = "Array" if kind == "array" else ""
            type_name = pascal_case(f"{name}{suffix}")
        else:
            suffix = "Array" if kind == "array" else "Type"
            type_name = pascal_case("_".join(path[1:]) or f"root_{suffix}")
    type_name = _require_identifier(type_name, label=f"{'.'.join(path)}.type_name")

    if kind == "enum":
        values_raw = raw_node.get("values")
        if not isinstance(values_raw, Sequence) or isinstance(values_raw, (str, bytes)) or not values_raw:
            raise ValueError(f"Enum node at {'.'.join(path)} must contain non-empty 'values'.")

        values: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        seen_values: set[int] = set()
        max_value = 0
        for index, value_raw in enumerate(values_raw):
            if not isinstance(value_raw, Mapping):
                raise TypeError(f"Enum value at {'.'.join(path)}[{index}] must be a mapping.")
            member_name = _require_identifier(
                value_raw.get("name"),
                label=f"{'.'.join(path)}.values[{index}].name",
            )
            member_value = int(value_raw.get("value"))
            if member_value < 0:
                raise ValueError(f"Enum value at {'.'.join(path)}[{index}] must be non-negative.")
            if member_name in seen_names:
                raise ValueError(f"Duplicate enum member '{member_name}' at {'.'.join(path)}.")
            if member_value in seen_values:
                raise ValueError(f"Duplicate enum value '{member_value}' at {'.'.join(path)}.")
            seen_names.add(member_name)
            seen_values.add(member_value)
            max_value = max(max_value, member_value)
            values.append({"name": member_name, "value": member_value})

        bitwidth = raw_node.get("bitwidth")
        min_bitwidth = max(max_value.bit_length(), 1)
        if bitwidth is None:
            bitwidth = min_bitwidth
        else:
            bitwidth = int(bitwidth)
            if bitwidth < min_bitwidth:
                raise ValueError(
                    f"Enum '{type_name}' requires at least {min_bitwidth} bits, got {bitwidth}."
                )

        return {
            "kind": "enum",
            "name": name,
            "description": description,
            "type_name": type_name,
            "bitwidth": bitwidth,
            "values": values,
        }

    if kind == "struct":
        fields_raw = raw_node.get("fields")
        if not isinstance(fields_raw, Sequence) or isinstance(fields_raw, (str, bytes)) or not fields_raw:
            raise ValueError(f"Struct node at {'.'.join(path)} must contain non-empty 'fields'.")

        fields: list[dict[str, Any]] = []
        seen_field_names: set[str] = set()
        for field_raw in fields_raw:
            field_node = _normalize_node(
                field_raw,
                path=[*path, str(field_raw.get("name", "field"))],
                context="struct_field",
            )
            field_name = field_node["name"]
            if field_name in seen_field_names:
                raise ValueError(f"Duplicate field name '{field_name}' in struct '{type_name}'.")
            seen_field_names.add(field_name)
            fields.append(field_node)

        return {
            "kind": "struct",
            "name": name,
            "description": description,
            "type_name": type_name,
            "fields": fields,
        }

    max_shape_raw = raw_node.get("max_shape")
    if not isinstance(max_shape_raw, Sequence) or isinstance(max_shape_raw, (str, bytes)) or not max_shape_raw:
        raise ValueError(f"Array node at {'.'.join(path)} must contain non-empty 'max_shape'.")
    max_shape = tuple(int(dim) for dim in max_shape_raw)
    if any(dim <= 0 for dim in max_shape):
        raise ValueError(f"Array node at {'.'.join(path)} must use positive max_shape dimensions.")

    static = bool(raw_node.get("static", True))
    element_name = raw_node.get("element_name") or singularize(name or snake_case(type_name))
    element_name = _require_identifier(element_name, label=f"{'.'.join(path)}.element_name")

    element_raw = raw_node.get("element")
    if element_raw is None:
        raise ValueError(f"Array node at {'.'.join(path)} must contain an 'element' node.")

    element = _normalize_node(
        element_raw,
        path=[*path, "element"],
        context="array_element",
    )
    if element["kind"] == "array":
        raise ValueError("Nested arrays are not supported by the initial AI schema pipeline.")

    return {
        "kind": "array",
        "name": name,
        "description": description,
        "type_name": type_name,
        "element_name": element_name,
        "element": element,
        "max_shape": max_shape,
        "static": static,
    }


def _require_identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty identifier.")
    if not value.isidentifier() or keyword.iskeyword(value):
        raise ValueError(f"{label} must be a valid non-keyword Python identifier.")
    return value


def _split_name(name: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    for char in name.replace("-", "_"):
        if char == "_":
            if current:
                parts.append("".join(current))
                current = []
            continue
        if char.isupper() and current and not current[-1].isupper():
            parts.append("".join(current))
            current = [char.lower()]
        else:
            current.append(char.lower())
    if current:
        parts.append("".join(current))
    return parts


def _validate_named_type_reuse(root: dict[str, Any]) -> None:
    definitions: dict[str, dict[str, Any]] = {}

    def visit(node: dict[str, Any]) -> None:
        kind = node["kind"]
        if kind in NAMED_NODE_KINDS:
            signature = _node_signature(node)
            existing = definitions.get(node["type_name"])
            if existing is None:
                definitions[node["type_name"]] = signature
            elif existing != signature:
                raise ValueError(
                    f"Type name '{node['type_name']}' is defined multiple times with different shapes."
                )

        if kind == "struct":
            for child in node["fields"]:
                visit(child)
        elif kind == "array":
            visit(node["element"])

    visit(root)


def _node_signature(node: dict[str, Any]) -> dict[str, Any]:
    signature = deepcopy(node)
    signature.pop("name", None)

    if node["kind"] == "struct":
        signature["fields"] = [_node_signature(child) for child in node["fields"]]
    elif node["kind"] == "array":
        signature["element"] = _node_signature(node["element"])

    return signature
