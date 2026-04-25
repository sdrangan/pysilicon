from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("pysilicon")

def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
