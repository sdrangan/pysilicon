---
title: OpenAI Setup
parent: Installation
nav_order: 3
has_children: false
---

# OpenAI Setup

## Overview

PySilicon can use OpenAI to power **semantic search** over the curated schema examples that ship with the package. This is an **optional** feature: all core MCP tools work without an OpenAI key. Only `pysilicon_search_schema_examples` requires it.

Because PySilicon uses a **bring-your-own-key** (BYO-key) model, the setup is per-user:

- You create and pay for your own OpenAI account.
- Your API key is never stored in the repository or shared with other users.
- API calls are billed directly to your OpenAI account.

---

## Obtaining an OpenAI API Key

1. Go to <https://platform.openai.com/> and sign in (or create a free account).
2. Navigate to **API keys** → **Create new secret key**.
3. Give the key a recognisable name (e.g. `pysilicon-dev`) and copy it.

> **Keep your key secret.** Treat it like a password — do not commit it to version control or share it in chat.

---

## Required Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Authenticates every call to the OpenAI API. Required when building the vector store. |
| `PYSILICON_EXAMPLES_VECTOR_STORE_ID` | The ID of the vector store created for your account. Written automatically by `pysilicon_mcp_setup --build-rag`. |

Set your API key in the terminal before running any OpenAI-enabled commands:

```bash
export OPENAI_API_KEY=sk-...
```

In Windows PowerShell, use:

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

To persist it for future PowerShell sessions, use:

```powershell
setx OPENAI_API_KEY "sk-..."
```

To make the key available in every new shell session, add the line above to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.).

On Windows, `setx` persists the variable for future shells but does not update the current PowerShell session. After `setx`, either open a new terminal or also run `$env:OPENAI_API_KEY = "sk-..."` in the current session.

> **Never** hardcode the key in `.vscode/mcp.json` or any other file that might be committed to git.

---

## How PySilicon Uses OpenAI

### 1. Vector Store Building

When you run:

```bash
pysilicon_mcp_setup --workspace . --build-rag
```

PySilicon:

1. Reads the example `.py` files packaged with `pysilicon` (poly, hist, conv2d, plus a catalog summary).
2. Uploads them to OpenAI's Files API under your account.
3. Creates a **vector store** named `pysilicon-examples` that embeds and indexes those files.
4. Waits for processing to complete, then saves the returned vector store ID into `.vscode/mcp.json`.

The vector store is created once and reused across sessions. You only need to rebuild it when you update `pysilicon` to a version with new or changed examples, or when your old store has expired.

### 2. Semantic Search at Query Time

Each time the AI assistant invokes `pysilicon_search_schema_examples`, PySilicon sends a natural-language query to OpenAI's vector-store search endpoint. OpenAI returns the most relevant example snippets, which the assistant uses to draft your schema.

No large language model (LLM) is called during search — only the vector-store retrieval endpoint is used. This keeps per-query costs very low (typically fractions of a cent).

---

## Billing and Usage Notes

- **Vector store storage**: OpenAI charges a small monthly fee per GB of vector store data. The pysilicon example corpus is small (a few KB), so this cost is negligible.
- **Retrieval calls**: Each `pysilicon_search_schema_examples` call makes one retrieval request. These are priced per request; see [OpenAI pricing](https://openai.com/pricing) for current rates.
- **Unused stores expire**: OpenAI may expire vector stores that have not been accessed recently. If searches stop working, re-run `pysilicon_mcp_setup --workspace . --build-rag --force` to rebuild.

---

## Next Step

Once your key is exported, wire everything together in one command:

```bash
pysilicon_mcp_setup --workspace . --build-rag
```

See [MCP Server Setup](./mcp_setup.md) for full details on what this command does and how to verify the result.
