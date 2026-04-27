"""
Tests for pysilicon MCP tools: schema_examples, registry, and server.
"""
from __future__ import annotations

import os

import pytest

from pysilicon.mcp.components import get_components
from pysilicon.mcp.example_rag import get_example_file, search_schema_examples
from pysilicon.mcp.schema_examples import get_schema_example, list_schema_examples
from pysilicon.mcp.registry import REGISTRY, ToolRegistry
from pysilicon.mcp.schema_tools import get_schema_draft_plan, validate_schema


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
    names = {s["function"]["name"] for s in REGISTRY.tool_schemas()}
    assert "pysilicon_list_schema_examples" not in names
    assert "pysilicon_get_schema_example" not in names
    assert "pysilicon_get_schema_draft_plan" in names
    assert "pysilicon_validate_schema" in names
    assert "pysilicon_get_components" in names
    assert "pysilicon_search_schema_examples" in names
    assert "pysilicon_get_example_file" in names


def test_registry_tool_schemas_are_openai_style():
    for schema in REGISTRY.tool_schemas():
        assert schema["type"] == "function"
        assert "function" in schema
        assert "name" in schema["function"]
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]
        assert schema["function"]["parameters"]["type"] == "object"


def test_registry_dispatch_disabled_list_schema_examples_raises_value_error():
    with pytest.raises(ValueError, match="Unknown tool name"):
        REGISTRY.dispatch("pysilicon_list_schema_examples", {})


def test_registry_dispatch_disabled_get_schema_example_raises_value_error():
    with pytest.raises(ValueError, match="Unknown tool name"):
        REGISTRY.dispatch(
            "pysilicon_get_schema_example", {"example_id": "hist_cmd"}
        )


def test_get_schema_draft_plan_final_step_recommends_validation():
    result = get_schema_draft_plan(task="Need a DMA command schema")

    assert result["steps"][-1]["goal"] == "Validate the schema and record assumptions"
    assert result["steps"][-1]["recommended_tools"] == ["pysilicon_validate_schema"]
    assert "structural or typing issues" in result["steps"][-1]["instructions"]


def test_get_schema_draft_plan_first_step_uses_registered_tool_names():
    result = get_schema_draft_plan()

    assert result["steps"][0]["recommended_tools"] == [
        "pysilicon_get_components",
        "pysilicon_search_schema_examples",
        "pysilicon_get_example_file",
    ]


def test_get_schema_draft_plan_first_step_uses_workspace_root_when_provided():
    result = get_schema_draft_plan(workspace_root="c:/demo/workspace")

    assert "Check the workspace at c:/demo/workspace" in result["steps"][0]["instructions"]
    assert "pysilicon_get_components" in result["steps"][0]["instructions"]
    assert "pysilicon_search_schema_examples" in result["steps"][0]["instructions"]


def test_get_schema_draft_plan_first_step_uses_only_example_tools_without_workspace_root():
    result = get_schema_draft_plan()

    instructions = result["steps"][0]["instructions"]
    assert "pysilicon_get_components" in instructions
    assert "pysilicon_search_schema_examples" in instructions
    assert "pysilicon_get_example_file" in instructions


def test_validate_schema_accepts_valid_datalist_source():
    result = validate_schema(
        "\n".join(
            [
                "from pysilicon.hw import DataList, IntField",
                "U16 = IntField.specialize(bitwidth=16, signed=False)",
                "class DemoPacket(DataList):",
                "    elements = {",
                "        'count': U16,",
                "    }",
            ]
        )
    )

    assert result["valid"] is True
    assert result["errors"] == []
    assert result["schema_info"] == {
        "name": "DemoPacket",
        "kind": "DataList",
        "field_count": 1,
        "field_names": ["count"],
    }


def test_validate_schema_reports_syntax_error_location():
    result = validate_schema(
        "class Broken(DataList)\n    pass",
        workspace_root="demo_workspace",
    )

    assert result["valid"] is False
    assert result["errors"][0]["code"] == "syntax_error"
    assert result["errors"][0]["location"]["line"] == 1
    assert result["errors"][0]["location"]["path"] == "demo_workspace"
    assert result["schema_info"] is None


def test_validate_schema_reports_invalid_elements_definition():
    result = validate_schema(
        "\n".join(
            [
                "from pysilicon.hw import DataList, IntField",
                "U16 = IntField.specialize(bitwidth=16, signed=False)",
                "class Broken(DataList):",
                "    elements = {",
                "        'count': {'description': 'missing schema'},",
                "    }",
            ]
        )
    )

    assert result["valid"] is False
    assert result["errors"][0]["code"] == "schema_validation_error"
    assert "must define a 'schema' entry" in result["errors"][0]["message"]
    assert result["schema_info"]["name"] == "Broken"


def test_registry_dispatch_validate_schema():
    result = REGISTRY.dispatch(
        "pysilicon_validate_schema",
        {
            "schema": "\n".join(
                [
                    "from pysilicon.hw import DataArray, IntField",
                    "U8 = IntField.specialize(bitwidth=8, signed=False)",
                    "class Payload(DataArray):",
                    "    element_type = U8",
                    "    max_shape = (4,)",
                    "    static = True",
                ]
            )
        },
    )

    assert result["valid"] is True
    assert result["schema_info"]["kind"] == "DataArray"


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


# ---------------------------------------------------------------------------
# pysilicon_get_components
# ---------------------------------------------------------------------------


def test_get_components_returns_expected_shape():
    result = get_components()
    assert "summary" in result
    assert "components" in result
    assert isinstance(result["summary"], str)
    assert isinstance(result["components"], list)
    assert len(result["components"]) > 0


def test_get_components_each_entry_has_required_fields():
    result = get_components()
    for entry in result["components"]:
        assert "name" in entry, f"entry missing 'name': {entry}"
        assert "kind" in entry, f"entry missing 'kind': {entry}"
        assert "description" in entry, f"entry missing 'description': {entry}"
        assert "keywords" in entry, f"entry missing 'keywords': {entry}"
        assert isinstance(entry["keywords"], list), (
            f"keywords for {entry['name']!r} should be a list"
        )


def test_get_components_contains_core_classes():
    result = get_components()
    names = {c["name"] for c in result["components"]}
    for expected in (
        "DataSchema",
        "DataList",
        "DataArray",
        "DataField",
        "IntField",
        "FloatField",
        "EnumField",
        "MemAddr",
        "IntEnum",
    ):
        assert expected in names, f"Expected component {expected!r} not found"


def test_get_components_is_deterministic():
    result1 = get_components()
    result2 = get_components()
    assert result1 == result2


def test_registry_dispatch_get_components():
    result = REGISTRY.dispatch("pysilicon_get_components", {})
    assert "components" in result
    assert "summary" in result


# ---------------------------------------------------------------------------
# pysilicon_search_schema_examples
# ---------------------------------------------------------------------------


def test_search_schema_examples_missing_env_var_returns_error_dict(monkeypatch):
    """When PYSILICON_EXAMPLES_VECTOR_STORE_ID is unset, a structured error
    dict is returned (no exception raised)."""
    monkeypatch.delenv("PYSILICON_EXAMPLES_VECTOR_STORE_ID", raising=False)

    result = search_schema_examples(task="histogram command schema", keywords=["DataList"])

    assert "summary" in result
    assert "normalized_query" in result
    assert "matches" in result
    assert isinstance(result["matches"], list)
    assert len(result["matches"]) == 0
    assert "error" in result
    assert "PYSILICON_EXAMPLES_VECTOR_STORE_ID" in result["error"]


def test_search_schema_examples_normalized_query_includes_keywords(monkeypatch):
    monkeypatch.delenv("PYSILICON_EXAMPLES_VECTOR_STORE_ID", raising=False)

    result = search_schema_examples(
        task="DMA command", keywords=["MemAddr", "DataList"]
    )

    assert "MemAddr" in result["normalized_query"]
    assert "DataList" in result["normalized_query"]


def test_search_schema_examples_normalized_query_no_keywords(monkeypatch):
    monkeypatch.delenv("PYSILICON_EXAMPLES_VECTOR_STORE_ID", raising=False)

    result = search_schema_examples(task="simple integer field")

    assert result["normalized_query"] == "simple integer field"


def test_registry_dispatch_search_schema_examples_missing_env(monkeypatch):
    monkeypatch.delenv("PYSILICON_EXAMPLES_VECTOR_STORE_ID", raising=False)

    result = REGISTRY.dispatch(
        "pysilicon_search_schema_examples",
        {"task": "conv2d accelerator command", "keywords": ["DataList"], "k": 3},
    )
    assert "error" in result


# ---------------------------------------------------------------------------
# pysilicon_get_example_file
# ---------------------------------------------------------------------------


def test_get_example_file_returns_poly():
    result = get_example_file("poly.py")
    assert result["path"] == "poly.py"
    assert "PolyCmdHdr" in result["content"]
    assert "DataList" in result["content"]


def test_get_example_file_returns_hist():
    result = get_example_file("hist.py")
    assert result["path"] == "hist.py"
    assert "HistCmd" in result["content"]


def test_get_example_file_returns_conv2d():
    result = get_example_file("conv2d.py")
    assert result["path"] == "conv2d.py"
    assert "Conv2DCmd" in result["content"]


def test_get_example_file_unknown_raises_value_error():
    with pytest.raises(ValueError, match="not found"):
        get_example_file("nonexistent_file.py")


def test_get_example_file_rejects_traversal():
    with pytest.raises(ValueError, match="Invalid example path"):
        get_example_file("../mcp/registry.py")


def test_registry_dispatch_get_example_file():
    result = REGISTRY.dispatch("pysilicon_get_example_file", {"path": "hist.py"})
    assert "content" in result
    assert "HistCmd" in result["content"]
