"""
Shared MCP tool registry for pysilicon.

``REGISTRY`` is the single source of truth for all tool definitions.  Both
the MCP server (``server.py``) and the blind-user harness
(``blind_user_llm.py``) import from here so that tool metadata is never
duplicated.

Usage
-----
MCP server::

    from pysilicon.mcp.registry import REGISTRY
    REGISTRY.register_all(mcp)

Blind-user harness::

    from pysilicon.mcp.registry import REGISTRY
    schemas = REGISTRY.tool_schemas()
    result  = REGISTRY.dispatch("pysilicon_list_schema_examples", {})
"""
from __future__ import annotations

from typing import Any, Callable

from mcp.server.fastmcp import FastMCP

from pysilicon.mcp.components import get_components
from pysilicon.mcp.example_rag import search_schema_examples
from pysilicon.mcp.schema_tools import get_schema_draft_plan, validate_schema


# ---------------------------------------------------------------------------
# ToolDef dataclass
# ---------------------------------------------------------------------------


class _ToolDef:
    """Internal holder for a single tool definition."""

    __slots__ = ("name", "description", "parameters", "fn")

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        fn: Callable[..., Any],
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.fn = fn


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Registry of pysilicon MCP tools.

    Responsibilities
    ----------------
    * Store tool definitions (name, description, JSON-Schema parameters, callable).
    * Register all tools with a :class:`~mcp.server.fastmcp.FastMCP` instance.
    * Return OpenAI ``function``-call style schemas for use in the blind-user harness.
    * Dispatch tool calls by name.
    """

    def __init__(self) -> None:
        self._tools: dict[str, _ToolDef] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        fn: Callable[..., Any],
    ) -> None:
        """Add a tool definition to the registry."""
        self._tools[name] = _ToolDef(
            name=name,
            description=description,
            parameters=parameters,
            fn=fn,
        )

    def register_all(self, mcp: FastMCP) -> None:
        """Register every tool in the registry with *mcp*."""
        for tool in self._tools.values():
            # FastMCP.tool() can be used as a decorator factory; we apply it
            # manually so the function stays importable as a plain callable.
            mcp.tool(name=tool.name, description=tool.description)(tool.fn)

    # ------------------------------------------------------------------
    # OpenAI / function-call schema export
    # ------------------------------------------------------------------

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Return a list of OpenAI-style function-call tool schemas.

        Each entry has the shape::

            {
                "type": "function",
                "function": {
                    "name": "<tool-name>",
                    "description": "...",
                    "strict": True,
                    "parameters": { <JSON Schema object> },
                },
            }
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "strict": True,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call the tool registered under *name* with *arguments*.

        Parameters
        ----------
        name:
            Tool name as registered (e.g. ``"pysilicon_list_schema_examples"``).
        arguments:
            Keyword arguments forwarded to the tool function.

        Raises
        ------
        ValueError
            If *name* is not registered.
        """
        tool = self._tools.get(name)
        if tool is None:
            known = sorted(self._tools.keys())
            raise ValueError(
                f"Unknown tool name: {name!r}. Registered tools: {known}"
            )
        return tool.fn(**arguments)


# ---------------------------------------------------------------------------
# Global registry instance and tool registrations
# ---------------------------------------------------------------------------

REGISTRY = ToolRegistry()

ENABLE_CURATED_SCHEMA_EXAMPLE_TOOLS = False

if ENABLE_CURATED_SCHEMA_EXAMPLE_TOOLS:
    REGISTRY.add(
        name="pysilicon_list_schema_examples",
        description=(
            "Return a curated catalog of available pysilicon schema examples. "
            "Each entry includes an id, title, description, and feature list. "
            "Call this first to discover which examples are available before "
            "calling pysilicon_get_schema_example."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        fn=list_schema_examples,
    )

    REGISTRY.add(
        name="pysilicon_get_schema_example",
        description=(
            "Return the full content for a pysilicon schema example by ID. "
            "The response includes metadata, the primary symbol name, supporting "
            "definitions, and the complete curated source code ready to adapt. "
            "Use pysilicon_list_schema_examples first to find valid IDs."
        ),
        parameters={
            "type": "object",
            "properties": {
                "example_id": {
                    "type": "string",
                    "description": "Schema example ID as returned by pysilicon_list_schema_examples.",
                }
            },
            "required": ["example_id"],
            "additionalProperties": False,
        },
        fn=get_schema_example,
    )

REGISTRY.add(
    name="pysilicon_get_schema_draft_plan",
    description=(
        "Return a deterministic step-by-step workflow for drafting a new "
        "pysilicon schema from a natural-language request. This helper does "
        "not search, rank, or recommend specific example IDs."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task": {
                "type": ["string", "null"],
                "description": "Optional natural-language description of the schema the user wants to draft.",
            },
            "workspace_root": {
                "type": ["string", "null"],
                "description": "Optional workspace root path accepted for consistency with other helper tools.",
            },
        },
        "required": ["task", "workspace_root"],
        "additionalProperties": False,
    },
    fn=get_schema_draft_plan,
)

REGISTRY.add(
    name="pysilicon_validate_schema",
    description=(
        "Validate drafted pysilicon schema source text deterministically and "
        "return structured errors, warnings, and extracted schema metadata."
    ),
    parameters={
        "type": "object",
        "properties": {
            "schema": {
                "type": "string",
                "description": "Drafted pysilicon schema source text to validate.",
            },
            "workspace_root": {
                "type": ["string", "null"],
                "description": "Optional workspace root path for contextual location reporting.",
            },
        },
        "required": ["schema", "workspace_root"],
        "additionalProperties": False,
    },
    fn=validate_schema,
)

REGISTRY.add(
    name="pysilicon_get_components",
    description=(
        "Return the canonical pysilicon schema vocabulary glossary. "
        "Includes all core schema classes (DataSchema, DataList, DataArray, "
        "DataField, IntField, FloatField, EnumField, MemAddr, IntEnum) and "
        "common design patterns with descriptions and keywords. "
        "Call this first to select relevant keywords for "
        "pysilicon_search_examples. Deterministic; no network access."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    fn=get_components,
)

REGISTRY.add(
    name="pysilicon_search_examples",
    description=(
        "Search the OpenAI-hosted vector store of pysilicon example corpus files. "
        "Returns the top-k most relevant example snippets for the given task. "
        "Requires PYSILICON_EXAMPLES_VECTOR_STORE_ID env var to be set. "
        "Use pysilicon_get_components first to obtain good keywords."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Natural-language description of the schema you want to build.",
            },
            "keywords": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": (
                    "Optional pysilicon vocabulary keywords (from pysilicon_get_components) "
                    "to augment the search query."
                ),
            },
            "k": {
                "type": ["integer", "null"],
                "description": "Maximum number of matches to return (default 5, max 20).",
            },
        },
        "required": ["task", "keywords", "k"],
        "additionalProperties": False,
    },
    fn=search_schema_examples,
)

REGISTRY.add(
    name="pysilicon_search_schema_examples",
    description=(
        "Compatibility alias for pysilicon_search_examples. "
        "Search the OpenAI-hosted vector store of pysilicon example corpus files. "
        "Requires PYSILICON_EXAMPLES_VECTOR_STORE_ID env var to be set."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Natural-language description of the example you want to find.",
            },
            "keywords": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": (
                    "Optional pysilicon vocabulary keywords (from pysilicon_get_components) "
                    "to augment the search query."
                ),
            },
            "k": {
                "type": ["integer", "null"],
                "description": "Maximum number of matches to return (default 5, max 20).",
            },
        },
        "required": ["task", "keywords", "k"],
        "additionalProperties": False,
    },
    fn=search_schema_examples,
)

