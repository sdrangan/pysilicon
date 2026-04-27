---
title: Rebuilding the OpenAI Vector Store
parent: Developer Guide
nav_order: 1
has_children: false
---

# Rebuilding the OpenAI Vector Store

## What Is the Vector Store?

PySilicon's MCP server exposes a tool called `pysilicon_search_schema_examples` that lets an AI assistant (e.g. GitHub Copilot in VS Code) search for relevant schema examples while helping a developer draft a new hardware data schema.

Behind this tool is an **OpenAI-hosted vector store**: a persistent, server-side search index that holds chunked and embedded representations of the curated example files shipped with the package (`pysilicon/examples/poly.py`, `hist.py`, `conv2d.py`, plus a generated catalog summary). When the tool is invoked, it sends a natural-language query to OpenAI, which retrieves the most relevant snippets from the index and returns them to the assistant. The assistant can then adapt those patterns when drafting new schema code.

Key properties of this vector store:

- **Hosted on OpenAI's servers** — there is no local index file to commit or distribute.
- **Referenced by ID** — the vector store is identified by a short string like `vs_abc123`. This ID is stored in the environment variable `PYSILICON_EXAMPLES_VECTOR_STORE_ID`.
- **Rebuilt from scratch** — whenever the examples change (new files, edits), you rebuild the store by running a single command. The old store is replaced.
- **Per-developer** — each developer uses their own OpenAI project and therefore has their own store. There is no shared central store in the repository.

---

## Prerequisites

1. **Python environment** — activate the virtual environment where `pysilicon` is installed:
   ```bash
   source .venv/bin/activate   # or the equivalent for your shell
   ```

2. **OpenAI API key** — set the standard OpenAI environment variable:
   ```bash
   export OPENAI_API_KEY=sk-...
   ```

   If you don't have a key, create one at <https://platform.openai.com/api-keys>.

---

## Rebuilding the Vector Store

Run the built-in CLI command:

```bash
pysilicon-build-example-rag
```

Or equivalently, without a console-script entry point:

```bash
python -m pysilicon.mcp.cli_build_example_rag
```

### What the command does

1. Enumerates every `.py` file in the `pysilicon.examples` package (`poly.py`, `hist.py`, `conv2d.py`).
2. Generates an in-memory `_CATALOG.md` that summarises the examples directory — this helps the retriever match natural-language queries to the right file even when the code is terse.
3. Uploads all files (examples + catalog) to OpenAI's Files API.
4. Creates a new vector store named `pysilicon-examples` and attaches the uploaded files.
5. Waits (polls) until OpenAI has finished chunking and embedding all files.
6. Prints the resulting **vector store ID** and a brief completion summary.

### Example output

```
Scanning packaged examples from 'pysilicon.examples' ...
  conv2d.py  (4211 chars)
  hist.py    (2893 chars)
  poly.py    (4080 chars)

Preparing 4 files for upload (including catalog) ...
  Uploading _CATALOG.md ... -> file-abc111
  Uploading conv2d.py  ... -> file-abc222
  Uploading hist.py    ... -> file-abc333
  Uploading poly.py    ... -> file-abc444

Creating vector store 'pysilicon-examples' ...
  Vector store ID: vs_abc123
  Waiting for processing to complete ...

✓ Vector store ready.
  ID:              vs_abc123
  Files completed: 4
  Files failed:    0

To use this vector store, set the environment variable:
  export PYSILICON_EXAMPLES_VECTOR_STORE_ID=vs_abc123
vs_abc123
```

---

## Persisting the Vector Store ID

Once the build completes, export the printed ID in your shell:

```bash
export PYSILICON_EXAMPLES_VECTOR_STORE_ID=vs_abc123
```

To make this permanent across sessions, add it to your shell profile (e.g. `~/.bashrc`, `~/.zshrc`) or to a `.env` file at the repository root:

```
PYSILICON_EXAMPLES_VECTOR_STORE_ID=vs_abc123
```

If you use VS Code with the MCP server, you may also configure the variable in `.vscode/mcp.json` under the `env` key — see the [MCP setup guide](../installation/mcp_setup.md) for details.

---

## Confirming the Vector Store Is Operational

### 1. Check the environment variable

```bash
echo $PYSILICON_EXAMPLES_VECTOR_STORE_ID
```

You should see a non-empty string like `vs_abc123`.

### 2. Run the search tool from Python

Open a Python REPL (with `pysilicon` installed and `PYSILICON_EXAMPLES_VECTOR_STORE_ID` set) and call the tool directly:

```python
from pysilicon.mcp.example_rag import search_schema_examples

result = search_schema_examples(
    task="command schema with memory address fields and transaction ID",
    keywords=["DataList", "MemAddr"],
    k=3,
)
print(result["summary"])
for m in result["matches"]:
    print("---")
    print(m.get("path", "<no path>"))
    print(m["snippet"][:200])
```

A successful response looks like:

```
Found 3 match(es) for query: 'command schema with memory address fields ...'
---
hist.py
class HistCmd(DataList):
    elements = {
        "tx_id": {"schema": TxIdField, ...},
        "data_addr": {"schema": AddrField, ...},
...
```

### 3. Check for the error key

If `PYSILICON_EXAMPLES_VECTOR_STORE_ID` is **not** set, `search_schema_examples` returns a structured error dict instead of raising an exception:

```python
result = search_schema_examples(task="test")
print(result)
# {'summary': 'Vector store ID not configured. Set the ...', 'matches': [], 'error': '...'}
```

This means the tool always returns a dict you can inspect. The `"error"` key is present only when something went wrong.

### 4. Dispatch through the registry

You can also test through the full MCP tool dispatch path:

```python
from pysilicon.mcp.registry import REGISTRY

result = REGISTRY.dispatch(
    "pysilicon_search_schema_examples",
    {"task": "histogram command schema", "keywords": ["DataList", "MemAddr"], "k": 2},
)
print(result["summary"])
```

---

## When to Rebuild

Rebuild the vector store whenever:

- You add a new example file to `pysilicon/examples/`.
- You make significant edits to an existing example file.
- You rotate your OpenAI API key (old uploads are tied to the old key's project).
- The old vector store has expired (OpenAI may expire unused stores; check the OpenAI dashboard).

The rebuild command is safe to run at any time — it always creates a brand-new store from scratch. Old uploaded files are not automatically deleted from your OpenAI project; you can clean them up manually via the OpenAI dashboard if needed.
