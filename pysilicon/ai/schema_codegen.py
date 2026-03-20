"""Render constrained schema specs into PySilicon dataschema modules."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from pysilicon.ai.schema_spec import collect_named_nodes, normalize_module_spec


def render_dataschema_module(spec: dict[str, Any]) -> str:
    normalized = normalize_module_spec(spec)
    ordered_nodes = collect_named_nodes(normalized)

    uses_enum = any(node["kind"] == "enum" for node in ordered_nodes)
    imports = [
        "from pysilicon.hw.dataschema import DataArray, DataList, EnumField, FloatField, IntField",
    ]
    if uses_enum:
        imports.insert(0, "from enum import IntEnum")

    lines: list[str] = [
        '"""Auto-generated PySilicon dataschema module."""',
        "",
        *imports,
        "",
        "",
    ]

    for node in ordered_nodes:
        if node["kind"] == "enum":
            lines.extend(_render_enum(node))
        elif node["kind"] == "struct":
            lines.extend(_render_struct(node))
        elif node["kind"] == "array":
            lines.extend(_render_array(node))
        else:
            raise ValueError(f"Unsupported named node kind: {node['kind']}")
        lines.append("")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_dataschema_module(spec: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_dataschema_module(spec), encoding="utf-8")
    return output


def generate_vitis_headers(
    spec: dict[str, Any],
    include_dir: str | Path,
    *,
    module_path: str | Path | None = None,
    word_bw_supported: list[int] | None = None,
) -> dict[str, Any]:
    normalized = normalize_module_spec(spec)
    module_source = render_dataschema_module(normalized)
    include_dir = Path(include_dir)
    include_dir.mkdir(parents=True, exist_ok=True)

    if word_bw_supported is None:
        word_bw_supported = list(normalized["word_bw_supported"])

    ordered_nodes = [node for node in collect_named_nodes(normalized) if node["kind"] in {"struct", "array"}]
    header_paths: list[str] = []
    failed_headers: list[dict[str, str]] = []

    def build_headers(generated_module_path: Path) -> None:
        module = _load_generated_module(generated_module_path, normalized["module_name"])
        for node in ordered_nodes:
            cls = getattr(module, node["type_name"])
            schema = cls(name=None)
            try:
                header_paths.append(
                    schema.gen_include(
                        include_dir=include_dir,
                        word_bw_supported=word_bw_supported,
                    )
                )
            except Exception as exc:  # pragma: no cover - exercised via higher-level tests
                failed_headers.append(
                    {
                        "type_name": node["type_name"],
                        "error": str(exc),
                    }
                )

    if module_path is not None:
        output_module_path = write_dataschema_module(normalized, module_path)
        build_headers(output_module_path)
        module_output = str(output_module_path)
    else:
        with TemporaryDirectory(prefix="pysilicon_ai_codegen_") as tmpdir:
            output_module_path = Path(tmpdir) / f"{normalized['module_name']}.py"
            output_module_path.write_text(module_source, encoding="utf-8")
            build_headers(output_module_path)
        module_output = None

    return {
        "module_name": normalized["module_name"],
        "module_output_path": module_output,
        "root_type_name": normalized["root"]["type_name"],
        "generated_types": [node["type_name"] for node in ordered_nodes],
        "header_paths": header_paths,
        "failed_headers": failed_headers,
    }


def _render_enum(node: dict[str, Any]) -> list[str]:
    lines = [f"class {node['type_name']}(IntEnum):"]
    for value in node["values"]:
        lines.append(f"    {value['name']} = {value['value']}")
    return lines


def _render_struct(node: dict[str, Any]) -> list[str]:
    lines = [f"class {node['type_name']}(DataList):", "    def __init__(self, name=None):"]
    if node.get("description"):
        lines.append(f"        super().__init__(name=name, description={node['description']!r})")
    else:
        lines.append("        super().__init__(name=name)")

    for field in node["fields"]:
        lines.append(f"        self.add_elem({_render_instance(field, field_name=field['name'])})")
    return lines


def _render_array(node: dict[str, Any]) -> list[str]:
    lines = [f"class {node['type_name']}(DataArray):", "    def __init__(self, name=None):"]
    element_expr = _render_instance(node["element"], field_name=node["element_name"])
    args = [
        "name=name",
        f"element_type={element_expr}",
        f"max_shape={tuple(node['max_shape'])!r}",
        f"static={node['static']!r}",
    ]
    if node.get("description"):
        args.append(f"description={node['description']!r}")
    lines.append("        super().__init__(")
    for arg in args:
        lines.append(f"            {arg},")
    lines.append("        )")
    return lines


def _render_instance(node: dict[str, Any], *, field_name: str) -> str:
    kind = node["kind"]
    if kind == "int":
        parts = [
            f"name={field_name!r}",
            f"bitwidth={node['bitwidth']}",
            f"signed={node['signed']!r}",
        ]
        if node.get("description"):
            parts.append(f"description={node['description']!r}")
        return f"IntField({', '.join(parts)})"

    if kind == "float":
        parts = [
            f"name={field_name!r}",
            f"bitwidth={node['bitwidth']}",
        ]
        if node.get("description"):
            parts.append(f"description={node['description']!r}")
        return f"FloatField({', '.join(parts)})"

    if kind == "enum":
        parts = [
            f"name={field_name!r}",
            f"enum_type={node['type_name']}",
        ]
        if node.get("bitwidth") is not None:
            parts.append(f"bitwidth={node['bitwidth']}")
        if node.get("description"):
            parts.append(f"description={node['description']!r}")
        return f"EnumField({', '.join(parts)})"

    if kind in {"struct", "array"}:
        return f"{node['type_name']}(name={field_name!r})"

    raise ValueError(f"Unsupported node kind: {kind}")


def _load_generated_module(module_path: Path, module_name: str) -> Any:
    import_name = f"_pysilicon_ai_{module_name}_{module_path.stem}_{abs(hash(str(module_path)))}"
    spec = importlib.util.spec_from_file_location(import_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load generated module from {module_path}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[import_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(import_name, None)
    return module
