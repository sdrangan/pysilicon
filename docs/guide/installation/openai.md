---
title: OpenAI Setup
parent: Installation
nav_order: 2
has_children: false
---

# OpenAI Setup

## Overview

PySilicon can use OpenAI to power **semantic search** over the packaged example corpus that ships with the package. This is an **optional** feature: all core MCP tools work without an OpenAI key. The canonical search tool is `pysilicon_search_examples`.

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
