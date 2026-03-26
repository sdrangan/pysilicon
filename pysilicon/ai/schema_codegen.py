"""Render constrained schema specs into class-driven PySilicon schema modules."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from pysilicon.ai.schema_spec import collect_named_nodes, normalize_module_spec
from pysilicon.codegen.build import CodeGenConfig
from pysilicon.codegen.streamutils import copy_streamutils


def render_dataschema_module(spec: dict[str, Any]) -> str:
    normalized = normalize_module_spec(spec)
    ordered_nodes = collect_named_nodes(normalized)

    uses_enum = _spec_uses_kind(normalized["root"], "enum")
    hw_imports: list[str] = []
    if _spec_uses_kind(normalized["root"], "array"):
        hw_imports.append("DataArray")
    if _spec_uses_kind(normalized["root"], "struct"):
        hw_imports.append("DataList")
    if uses_enum:
        hw_imports.append("EnumField")
    if _spec_uses_kind(normalized["root"], "float"):
        hw_imports.append("FloatField")
    if _spec_uses_kind(normalized["root"], "int"):
        hw_imports.append("IntField")

    imports = [f"from pysilicon.hw import {', '.join(hw_imports)}"]
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
        lines.extend(["", ""])

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
    include_root = Path(include_dir)
    include_root.mkdir(parents=True, exist_ok=True)
    cfg = CodeGenConfig(root_dir=include_root)

    if word_bw_supported is None:
        word_bw_supported = list(normalized["word_bw_supported"])

    ordered_nodes = collect_named_nodes(normalized)
    generated_symbols = [_schema_symbol_name(node) for node in ordered_nodes]
    header_paths: list[str] = []
    failed_headers: list[dict[str, str]] = []

    def build_headers(generated_module_path: Path) -> None:
        module = _load_generated_module(generated_module_path, normalized["module_name"])
        copy_streamutils(cfg)

        for node in ordered_nodes:
            symbol_name = _schema_symbol_name(node)
            schema_cls = getattr(module, symbol_name)
            try:
                out_path = schema_cls.gen_include(cfg=cfg, word_bw_supported=word_bw_supported)
                header_paths.append(str(out_path))
            except Exception as exc:  # pragma: no cover - exercised via higher-level tests
                failed_headers.append(
                    {
                        "schema_symbol": symbol_name,
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
        "generated_types": generated_symbols,
        "header_paths": header_paths,
        "failed_headers": failed_headers,
    }


def _render_enum(node: dict[str, Any]) -> list[str]:
    schema_symbol = _schema_symbol_name(node)
    lines = [f"class {node['type_name']}(IntEnum):"]
    for value in node["values"]:
        lines.append(f"    {value['name']} = {value['value']}")
    lines.extend(
        [
            "",
            f"{schema_symbol} = EnumField.specialize(",
            f"    enum_type={node['type_name']},",
            f"    bitwidth={node['bitwidth']},",
            ")",
        ]
    )
    return lines


def _render_struct(node: dict[str, Any]) -> list[str]:
    lines = [f"class {node['type_name']}(DataList):"]
    if node.get("description"):
        lines.append(f'    """{node["description"]}"""')
    lines.append("    elements = {")
    for field in node["fields"]:
        lines.extend(_render_field_entry(field))
    lines.append("    }")
    return lines


def _render_array(node: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if node.get("description"):
        lines.append(f"# {node['description']}")
    lines.extend(
        [
            f"{node['type_name']} = DataArray.specialize(",
            f"    element_type={_render_schema_expr(node['element'])},",
            f"    max_shape={tuple(node['max_shape'])!r},",
            f"    static={node['static']!r},",
            f"    member_name={node['element_name']!r},",
            f"    cpp_repr={node['type_name']!r},",
            f"    include_filename={_default_include_filename(node['type_name'])!r},",
            ")",
        ]
    )
    return lines


def _render_field_entry(node: dict[str, Any]) -> list[str]:
    schema_expr = _render_schema_expr(node)
    if node.get("description"):
        return [
            f"        {node['name']!r}: {{",
            f"            'schema': {schema_expr},",
            f"            'description': {node['description']!r},",
            "        },",
        ]
    return [f"        {node['name']!r}: {schema_expr},"]


def _render_schema_expr(node: dict[str, Any]) -> str:
    kind = node["kind"]
    if kind == "int":
        return f"IntField.specialize(bitwidth={node['bitwidth']}, signed={node['signed']!r})"
    if kind == "float":
        return f"FloatField.specialize(bitwidth={node['bitwidth']})"
    if kind == "enum":
        return _schema_symbol_name(node)
    if kind in {"struct", "array"}:
        return node["type_name"]
    raise ValueError(f"Unsupported schema expression kind: {kind}")


def _schema_symbol_name(node: dict[str, Any]) -> str:
    if node["kind"] == "enum":
        return f"{node['type_name']}Field"
    return node["type_name"]


def _default_include_filename(type_name: str) -> str:
    stem = re.sub(r"(?<!^)(?=[A-Z])", "_", type_name).lower()
    return f"{stem}.h"


def _spec_uses_kind(node: dict[str, Any], kind: str) -> bool:
    if node["kind"] == kind:
        return True
    if node["kind"] == "struct":
        return any(_spec_uses_kind(field, kind) for field in node["fields"])
    if node["kind"] == "array":
        return _spec_uses_kind(node["element"], kind)
    return False


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
