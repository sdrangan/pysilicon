"""MCP server exposing deterministic PySilicon dataschema tooling."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from pysilicon.ai.interface_bundle import (
    generate_interface_bundle as generate_interface_bundle_artifacts,
    interface_bundle_from_callable_symbol,
    interface_bundle_from_python_symbols,
    read_interface_manifest,
    validate_generated_schema_with_vitis,
    validate_interface_bundle_with_vitis,
)
from pysilicon.ai.schema_codegen import generate_vitis_headers, render_dataschema_module, write_dataschema_module
from pysilicon.ai.schema_spec import normalize_module_spec
from pysilicon.ai.type_inference import infer_schema_spec_from_symbol, load_python_symbol
from pysilicon.hw.dataschema import DataArray

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - optional runtime dependency
    FastMCP = None


def validate_schema_spec(spec: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_module_spec(spec)
    return {
        "ok": True,
        "module_name": normalized["module_name"],
        "root_type_name": normalized["root"]["type_name"],
        "spec": normalized,
    }


def spec_from_python_symbol(
    module_path: str,
    symbol_name: str,
    *,
    module_name: str | None = None,
) -> dict[str, Any]:
    symbol = load_python_symbol(module_path, symbol_name)
    spec = infer_schema_spec_from_symbol(symbol, module_name=module_name, root_type_name=symbol_name)
    return {
        "ok": True,
        "spec": spec,
        "root_type_name": spec["root"]["type_name"],
        "module_name": spec["module_name"],
    }


def bundle_from_python_symbols(
    module_path: str,
    members: list[dict[str, Any]],
    *,
    interface_name: str,
    description: str | None = None,
    assumptions: list[str] | None = None,
    word_bw_supported: list[int] | None = None,
    manifest_formats: list[str] | None = None,
) -> dict[str, Any]:
    bundle = interface_bundle_from_python_symbols(
        module_path,
        members,
        interface_name=interface_name,
        description=description,
        assumptions=assumptions,
        word_bw_supported=word_bw_supported,
        manifest_formats=manifest_formats,
    )
    return {
        "ok": True,
        "bundle": bundle,
        "interface_name": bundle["interface_name"],
        "members": [
            {
                "name": member["name"],
                "direction": member["direction"],
                "role": member["role"],
                "root_type_name": member["spec"]["root"]["type_name"],
            }
            for member in bundle["members"]
        ],
    }


def bundle_from_callable_symbol(
    module_path: str,
    symbol_name: str,
    *,
    interface_name: str | None = None,
    description: str | None = None,
    assumptions: list[str] | None = None,
    word_bw_supported: list[int] | None = None,
    sample_inputs: dict[str, Any] | None = None,
    evaluate_outputs: bool = False,
    manifest_formats: list[str] | None = None,
) -> dict[str, Any]:
    bundle = interface_bundle_from_callable_symbol(
        module_path,
        symbol_name,
        interface_name=interface_name,
        description=description,
        assumptions=assumptions,
        word_bw_supported=word_bw_supported,
        sample_inputs=sample_inputs,
        evaluate_outputs=evaluate_outputs,
        manifest_formats=manifest_formats,
    )
    return {
        "ok": True,
        "bundle": bundle,
        "interface_name": bundle["interface_name"],
        "members": [
            {
                "name": member["name"],
                "direction": member["direction"],
                "role": member["role"],
                "root_type_name": member["spec"]["root"]["type_name"],
            }
            for member in bundle["members"]
        ],
    }


def generate_dataschema_module(
    spec: dict[str, Any],
    *,
    output_path: str | None = None,
) -> dict[str, Any]:
    normalized = normalize_module_spec(spec)
    source = render_dataschema_module(normalized)
    result = {
        "ok": True,
        "module_name": normalized["module_name"],
        "root_type_name": normalized["root"]["type_name"],
        "source": source,
    }
    if output_path is not None:
        written_path = write_dataschema_module(normalized, output_path)
        result["output_path"] = str(written_path)
    return result


def generate_schema_headers(
    spec: dict[str, Any],
    *,
    include_dir: str,
    module_output_path: str | None = None,
    word_bw_supported: list[int] | None = None,
) -> dict[str, Any]:
    result = generate_vitis_headers(
        spec,
        include_dir=include_dir,
        module_path=module_output_path,
        word_bw_supported=word_bw_supported,
    )
    result["ok"] = True
    return result


def generate_interface_bundle(
    bundle_spec: dict[str, Any],
    *,
    output_dir: str,
    vector_word_bw: int | None = None,
    validate_python_roundtrip: bool = True,
    validate_vitis_roundtrip: bool = False,
) -> dict[str, Any]:
    result = generate_interface_bundle_artifacts(
        bundle_spec,
        output_dir=output_dir,
        vector_word_bw=vector_word_bw,
        validate_python_roundtrip=validate_python_roundtrip,
        validate_vitis_roundtrip=validate_vitis_roundtrip,
    )
    result["ok"] = True
    return result


def validate_generated_schema(
    spec: dict[str, Any],
    *,
    payload: Any,
    word_bw: int = 32,
) -> dict[str, Any]:
    normalized = normalize_module_spec(spec)
    source = render_dataschema_module(normalized)

    with TemporaryDirectory(prefix="pysilicon_ai_validate_") as tmpdir:
        module_path = Path(tmpdir) / f"{normalized['module_name']}.py"
        module_path.write_text(source, encoding="utf-8")

        from pysilicon.ai.schema_codegen import _load_generated_module  # local import to avoid public coupling

        module = _load_generated_module(module_path, normalized["module_name"])
        root_cls = getattr(module, normalized["root"]["type_name"])

        schema = root_cls()
        if isinstance(schema, DataArray):
            schema.val = payload
        else:
            schema.from_dict(payload)

        packed = schema.serialize(word_bw=word_bw)
        restored = _deserialize_target_instance(root_cls)
        restored.deserialize(packed, word_bw=word_bw)

        return {
            "ok": bool(restored.is_close(schema)),
            "word_bw": word_bw,
            "packed_words": _jsonable(packed),
            "root_type_name": normalized["root"]["type_name"],
        }


def validate_schema_with_vitis(
    spec: dict[str, Any],
    *,
    payload: Any,
    work_dir: str,
    word_bw: int = 32,
) -> dict[str, Any]:
    result = validate_generated_schema_with_vitis(
        spec,
        payload=payload,
        work_dir=work_dir,
        word_bw=word_bw,
    )
    result["ok"] = bool(result.get("ok", False))
    return result


def validate_bundle_with_vitis(
    bundle_spec: dict[str, Any],
    *,
    output_dir: str,
    word_bw: int = 32,
) -> dict[str, Any]:
    result = validate_interface_bundle_with_vitis(
        bundle_spec,
        output_dir=output_dir,
        word_bw=word_bw,
    )
    result["ok"] = bool(result.get("ok", False))
    return result


def load_interface_manifest(
    manifest_path: str,
) -> dict[str, Any]:
    manifest = read_interface_manifest(manifest_path)
    return {
        "ok": True,
        "manifest": manifest,
        "interface_name": manifest["interface_name"],
    }


def build_server() -> Any:
    if FastMCP is None:
        raise RuntimeError(
            "The optional 'mcp' package is not installed. Install pysilicon with the 'ai' extra."
        )

    server = FastMCP(
        name="pysilicon-schema",
        instructions=(
            "Deterministic tools for turning constrained schema specs or Python symbols "
            "into PySilicon dataschema modules and Vitis headers."
        ),
    )

    @server.tool()
    def validate_schema_spec_tool(spec: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize a constrained dataschema spec."""
        return validate_schema_spec(spec)

    @server.tool()
    def spec_from_python_symbol_tool(
        module_path: str,
        symbol_name: str,
        module_name: str | None = None,
    ) -> dict[str, Any]:
        """Infer a constrained dataschema spec from a Python symbol or notebook symbol."""
        return spec_from_python_symbol(module_path, symbol_name, module_name=module_name)

    @server.tool()
    def bundle_from_python_symbols_tool(
        module_path: str,
        members: list[dict[str, Any]],
        interface_name: str,
        description: str | None = None,
        assumptions: list[str] | None = None,
        word_bw_supported: list[int] | None = None,
        manifest_formats: list[str] | None = None,
    ) -> dict[str, Any]:
        """Infer a normalized interface bundle from Python or notebook symbols."""
        return bundle_from_python_symbols(
            module_path,
            members,
            interface_name=interface_name,
            description=description,
            assumptions=assumptions,
            word_bw_supported=word_bw_supported,
            manifest_formats=manifest_formats,
        )

    @server.tool()
    def bundle_from_callable_symbol_tool(
        module_path: str,
        symbol_name: str,
        interface_name: str | None = None,
        description: str | None = None,
        assumptions: list[str] | None = None,
        word_bw_supported: list[int] | None = None,
        sample_inputs: dict[str, Any] | None = None,
        evaluate_outputs: bool = False,
        manifest_formats: list[str] | None = None,
    ) -> dict[str, Any]:
        """Infer a normalized interface bundle from a callable signature, optionally evaluating outputs."""
        return bundle_from_callable_symbol(
            module_path,
            symbol_name,
            interface_name=interface_name,
            description=description,
            assumptions=assumptions,
            word_bw_supported=word_bw_supported,
            sample_inputs=sample_inputs,
            evaluate_outputs=evaluate_outputs,
            manifest_formats=manifest_formats,
        )

    @server.tool()
    def generate_dataschema_module_tool(
        spec: dict[str, Any],
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Render a PySilicon dataschema module from a validated schema spec."""
        return generate_dataschema_module(spec, output_path=output_path)

    @server.tool()
    def generate_schema_headers_tool(
        spec: dict[str, Any],
        include_dir: str,
        module_output_path: str | None = None,
        word_bw_supported: list[int] | None = None,
    ) -> dict[str, Any]:
        """Generate one Vitis header per named struct/array type in the schema."""
        return generate_schema_headers(
            spec,
            include_dir=include_dir,
            module_output_path=module_output_path,
            word_bw_supported=word_bw_supported,
        )

    @server.tool()
    def generate_interface_bundle_tool(
        bundle_spec: dict[str, Any],
        output_dir: str,
        vector_word_bw: int | None = None,
        validate_python_roundtrip: bool = True,
        validate_vitis_roundtrip: bool = False,
    ) -> dict[str, Any]:
        """Generate specs, dataschema modules, headers, vectors, manifests, and a report for an interface bundle."""
        return generate_interface_bundle(
            bundle_spec,
            output_dir=output_dir,
            vector_word_bw=vector_word_bw,
            validate_python_roundtrip=validate_python_roundtrip,
            validate_vitis_roundtrip=validate_vitis_roundtrip,
        )

    @server.tool()
    def validate_generated_schema_tool(
        spec: dict[str, Any],
        payload_json: str,
        word_bw: int = 32,
    ) -> dict[str, Any]:
        """Roundtrip a payload through the generated schema using Python serialization."""
        return validate_generated_schema(spec, payload=json.loads(payload_json), word_bw=word_bw)

    @server.tool()
    def validate_schema_with_vitis_tool(
        spec: dict[str, Any],
        payload_json: str,
        work_dir: str,
        word_bw: int = 32,
    ) -> dict[str, Any]:
        """Prepare and optionally run Vitis roundtrip validation for one schema."""
        return validate_schema_with_vitis(
            spec,
            payload=json.loads(payload_json),
            work_dir=work_dir,
            word_bw=word_bw,
        )

    @server.tool()
    def validate_bundle_with_vitis_tool(
        bundle_spec: dict[str, Any],
        output_dir: str,
        word_bw: int = 32,
    ) -> dict[str, Any]:
        """Prepare and optionally run Vitis roundtrip validation for an interface bundle."""
        return validate_bundle_with_vitis(
            bundle_spec,
            output_dir=output_dir,
            word_bw=word_bw,
        )

    @server.tool()
    def load_interface_manifest_tool(manifest_path: str) -> dict[str, Any]:
        """Load a previously generated JSON or YAML interface manifest."""
        return load_interface_manifest(manifest_path)

    return server


def main() -> None:
    server = build_server()
    server.run(transport="stdio")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _deserialize_target_instance(root_cls: type[Any]) -> Any:
    schema = root_cls()
    if isinstance(schema, DataArray) and not schema.static:
        schema.val = _full_array_value(root_cls)
    return schema


def _full_array_value(array_cls: type[Any]) -> Any:
    shape = tuple(int(dim) for dim in array_cls.max_shape)
    elem_init = array_cls._element_type().init_value()

    if isinstance(elem_init, np.generic):
        return np.zeros(shape, dtype=np.asarray(elem_init).dtype)

    if isinstance(elem_init, np.ndarray):
        return np.zeros(shape + tuple(elem_init.shape), dtype=elem_init.dtype)

    def build(shape_tail: tuple[int, ...]) -> Any:
        if not shape_tail:
            return array_cls._element_type().init_value()
        return [build(shape_tail[1:]) for _ in range(shape_tail[0])]

    return build(shape)


if __name__ == "__main__":  # pragma: no cover
    main()
