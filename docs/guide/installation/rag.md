---
title: Semantic Example Search
parent: Installation
nav_order: 4
has_children: false
---

# OpenAI RAG for Semantic Example Search

PySilicon includes a tool called `pysilicon_search_schema_examples` that lets an AI assistant such as GitHub Copilot in VS Code semantically search through the curated schema examples that ship with the package. For example, you can ask for an example with a memory address field and an enum and get back the most relevant example code.

This search is powered by an OpenAI-hosted vector store. Because each user supplies their own OpenAI API key, the vector store lives in your OpenAI account and usage is billed to you. See [OpenAI Setup](./openai.md) for details on obtaining a key.

The RAG setup is optional. Without it:

- All other MCP tools continue to work normally.
- `pysilicon_search_schema_examples` returns a structured message explaining that the vector store is not configured.
- You can still browse and retrieve individual example files with `pysilicon_get_example_file`.

## Enabling Semantic Search with `--build-rag`

To build your personal vector store and wire it into VS Code automatically, run:

```bash
export OPENAI_API_KEY=sk-...       # set your key first
pysilicon_mcp_setup --workspace . --build-rag
```

On Windows PowerShell, use:

```powershell
$env:OPENAI_API_KEY = "sk-..."
pysilicon_mcp_setup --workspace . --build-rag
```

To persist the key for future PowerShell sessions, use:

```powershell
setx OPENAI_API_KEY "sk-..."
```

When `--build-rag` finishes, it prints matching Unix and PowerShell commands for `PYSILICON_EXAMPLES_VECTOR_STORE_ID`. In the normal `pysilicon_mcp_setup --build-rag` flow you usually do not need to run those manually, because the setup command writes the vector store ID into `.vscode/mcp.json` for the `pysilicon` MCP server.

What this does:

1. Uploads the packaged example files (`poly.py`, `hist.py`, `conv2d.py`, and a generated catalog) to your OpenAI account.
2. Creates a vector store named `pysilicon-examples` in your OpenAI project.
3. Waits for OpenAI to finish processing the files.
4. If `PYSILICON_EXAMPLES_VECTOR_STORE_ID` is already set in the environment, PySilicon attempts to delete that previous vector store after the replacement store is ready.
5. Writes the resulting vector store ID into `.vscode/mcp.json` under `servers.pysilicon.env`:

```json
{
  "servers": {
    "pysilicon": {
      "type": "stdio",
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "pysilicon.mcp.server"],
      "env": {
        "PYSILICON_EXAMPLES_VECTOR_STORE_ID": "vs_abc123"
      }
    }
  }
}
```

After VS Code reloads the MCP server, `pysilicon_search_schema_examples` will be fully operational.

## Combining `--build-rag` with Other Flags

`--build-rag` respects the same flags as the base command:

| Flag | Effect |
|------|--------|
| `--force` | Overwrite an existing `.vscode/mcp.json`. |
| `--dry-run` | Print the config (with the env var set) without writing any files. |

Example: preview the config before writing.

```bash
pysilicon_mcp_setup --workspace . --build-rag --dry-run
```

## When to Rebuild

Rebuild the vector store by re-running `--build-rag` whenever:

- You update `pysilicon` to a version that adds or changes example files.
- The old vector store has expired.
- You rotate your OpenAI API key and want a fresh store in that project.

## Deleting Old Stores

During `pysilicon_mcp_setup --build-rag`, PySilicon checks `PYSILICON_EXAMPLES_VECTOR_STORE_ID`. If that variable already points to an older PySilicon vector store, PySilicon tries to delete that old store after the new one has been created successfully.

That cleanup is best-effort only. You should still periodically review your OpenAI account and remove stores that are no longer used.

To find stores on the OpenAI platform website:

1. Sign in at <https://platform.openai.com/>.
2. Open the API dashboard for the same project that owns your `OPENAI_API_KEY`.
3. Look for a storage or vector-store management page in the dashboard UI.
4. Search for stores named `pysilicon-examples` or for the exact store ID shown in `.vscode/mcp.json` under `PYSILICON_EXAMPLES_VECTOR_STORE_ID`.

If your account UI does not expose vector stores directly, use the OpenAI API reference and dashboard as a fallback:

1. Open <https://developers.openai.com/api/reference/resources/vector_stores>.
2. Confirm the vector-store resource and IDs you want to inspect.
3. Use your project credentials to list stores through the API, then delete only the specific store IDs you no longer need.
