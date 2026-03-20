"""MCP server exposing deterministic PySilicon dataschema tooling."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from pysilicon.ai.schema_codegen import generate_vitis_headers, render_dataschema_module, write_dataschema_module
from pysilicon.ai.schema_spec import normalize_module_spec, snake_case
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
    spec = infer_schema_spec_from_symbol(symbol, module_name=module_name)
    return {
        "ok": True,
        "spec": spec,
        "root_type_name": spec["root"]["type_name"],
        "module_name": spec["module_name"],
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

        schema_name = snake_case(normalized["root"]["type_name"])
        schema = root_cls(name=schema_name)
        if isinstance(schema, DataArray):
            schema.val = payload
        else:
            schema.from_dict(payload)

        packed = schema.serialize(word_bw=word_bw)
        restored = root_cls(name=f"{schema_name}_restored")
        restored.deserialize(packed, word_bw=word_bw)

        return {
            "ok": bool(restored.is_close(schema)),
            "word_bw": word_bw,
            "packed_words": _jsonable(packed),
            "root_type_name": normalized["root"]["type_name"],
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
        """Infer a constrained dataschema spec from a dataclass or TypedDict symbol."""
        return spec_from_python_symbol(module_path, symbol_name, module_name=module_name)

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
    def validate_generated_schema_tool(
        spec: dict[str, Any],
        payload_json: str,
        word_bw: int = 32,
    ) -> dict[str, Any]:
        """Roundtrip a payload through the generated schema using Python serialization."""
        return validate_generated_schema(spec, payload=json.loads(payload_json), word_bw=word_bw)

    return server


def main() -> None:
    server = build_server()
    server.run(transport="stdio")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


if __name__ == "__main__":  # pragma: no cover
    main()
