from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from pysilicon.mcp.registry import REGISTRY

mcp = FastMCP("pysilicon")
REGISTRY.register_all(mcp)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
