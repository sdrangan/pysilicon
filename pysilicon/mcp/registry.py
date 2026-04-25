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

from pysilicon.mcp.schema_examples import get_schema_example, list_schema_examples


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
