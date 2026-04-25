"""
Tests for pysilicon MCP tools: schema_examples, registry, and server.
"""
from __future__ import annotations

import pytest

from pysilicon.mcp.schema_examples import get_schema_example, list_schema_examples
from pysilicon.mcp.registry import REGISTRY, ToolRegistry


# ---------------------------------------------------------------------------
# list_schema_examples
# ---------------------------------------------------------------------------


def test_list_schema_examples_returns_dict_with_examples_key():
    result = list_schema_examples()
    assert "examples" in result
    assert "summary" in result


def test_list_schema_examples_returns_all_expected_ids():
    result = list_schema_examples()
    ids = {ex["id"] for ex in result["examples"]}
    expected = {
        "poly_cmd_hdr",
        "poly_resp_hdr",
        "poly_resp_ftr",
        "hist_cmd",
        "hist_resp",
        "conv2d_cmd",
        "conv2d_resp",
        "conv2d_debug",
    }
    assert ids == expected


def test_list_schema_examples_each_entry_has_required_fields():
    result = list_schema_examples()
    for entry in result["examples"]:
        assert "id" in entry
        assert "title" in entry
        assert "description" in entry
        assert "features" in entry
        assert isinstance(entry["features"], list), (
            f"features for {entry['id']!r} should be a list"
        )


# ---------------------------------------------------------------------------
# get_schema_example
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "example_id",
    [
        "poly_cmd_hdr",
        "poly_resp_hdr",
        "poly_resp_ftr",
        "hist_cmd",
        "hist_resp",
        "conv2d_cmd",
        "conv2d_resp",
        "conv2d_debug",
    ],
)
def test_get_schema_example_returns_expected_fields(example_id: str):
    result = get_schema_example(example_id)
    assert result["id"] == example_id
    assert "title" in result
    assert "description" in result
    assert "features" in result
    assert isinstance(result["features"], list)
    assert "primary_symbol" in result
    assert "supporting_symbols" in result
    assert isinstance(result["supporting_symbols"], list)
    assert "source_code" in result
    assert len(result["source_code"]) > 0


def test_get_schema_example_source_code_contains_primary_symbol():
    for example_id in ("poly_cmd_hdr", "hist_resp", "conv2d_debug"):
        result = get_schema_example(example_id)
        assert result["primary_symbol"] in result["source_code"], (
            f"primary_symbol {result['primary_symbol']!r} not found in source_code "
            f"for example {example_id!r}"
        )


def test_get_schema_example_unknown_id_raises_value_error():
    with pytest.raises(ValueError, match="Unknown schema example id"):
        get_schema_example("definitely_not_a_real_id")


def test_get_schema_example_poly_resp_ftr_includes_enum_support():
    result = get_schema_example("poly_resp_ftr")
    assert "PolyError" in result["supporting_symbols"]
    assert "PolyErrorField" in result["supporting_symbols"]
    # Source code should define the enum and the DataList
    assert "PolyError" in result["source_code"]
    assert "PolyRespFtr" in result["source_code"]


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


def test_registry_has_expected_tools():
    names = {s["name"] for s in REGISTRY.tool_schemas()}
    assert "pysilicon_list_schema_examples" in names
    assert "pysilicon_get_schema_example" in names


def test_registry_tool_schemas_are_openai_style():
    for schema in REGISTRY.tool_schemas():
        assert schema["type"] == "function"
        assert "name" in schema
        assert "description" in schema
        assert "parameters" in schema
        assert schema["parameters"]["type"] == "object"


def test_registry_dispatch_list_schema_examples():
    result = REGISTRY.dispatch("pysilicon_list_schema_examples", {})
    assert "examples" in result
    assert len(result["examples"]) > 0


def test_registry_dispatch_get_schema_example():
    result = REGISTRY.dispatch(
        "pysilicon_get_schema_example", {"example_id": "hist_cmd"}
    )
    assert result["id"] == "hist_cmd"


def test_registry_dispatch_unknown_tool_raises_value_error():
    with pytest.raises(ValueError, match="Unknown tool name"):
        REGISTRY.dispatch("not_a_real_tool", {})


def test_registry_register_all_adds_tools_to_mcp():
    from mcp.server.fastmcp import FastMCP

    fresh_mcp = FastMCP("test_server")
    fresh_registry = ToolRegistry()
    fresh_registry.add(
        name="test_tool",
        description="A test tool",
        parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        fn=lambda: {"ok": True},
    )
    fresh_registry.register_all(fresh_mcp)
    # FastMCP doesn't expose a public tool list, but registration should not raise
    # and the tool should be dispatchable via our registry
    result = fresh_registry.dispatch("test_tool", {})
    assert result == {"ok": True}


# ---------------------------------------------------------------------------
# server smoke test
# ---------------------------------------------------------------------------


def test_server_imports_and_has_mcp_instance():
    from pysilicon.mcp.server import mcp
    assert mcp.name == "pysilicon"
