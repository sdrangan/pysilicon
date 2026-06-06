---
title: AI Tooling
parent: Guide
nav_order: 10
has_children: true
---

# AI Tooling

PySilicon includes **optional** AI-assistant integrations: a VS Code extension, an MCP
server that exposes PySilicon's hardware-design tools to agents (Claude Code, VS Code),
OpenAI-backed semantic example search (RAG), and the keys/setup they need. **None of
these are required** to use or develop PySilicon — the [core package](../installation/)
runs standalone. Reach for them only when you want AI-assisted design.

> **Heads-up:** this is an evolving area and these pages can lag the code. If a step
> doesn't match the current tooling, treat the page as intent rather than exact steps.

- [VS Code extension](./vscode.md) — the PySilicon IDE extension.
- [MCP server](./mcp_setup.md) — expose PySilicon's tools to agentic assistants.
- [OpenAI setup](./openai.md) — API key + environment variables for semantic search.
- [Semantic example search (RAG)](./rag.md) — build and manage the example-search stores.
