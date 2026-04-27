---
title: Installing the MCP Server
parent: Installation
nav_order: 2
has_children: false
---

# Setting up The MCP Server

## Overview

PySilicon is being designed with agentic assistance to help build and simulate modules with the package. To use this facility, you will have to set up the **model context protocol** or MCP server. For now the setup is designed for users in VS Code. The process can be adapted for other IDEs such as Claude Code.

## Set-Up

To set up the MCP server on VS Code, first follow the [instructions](./python.md) to create and activate a virtual environment, then install `pysilicon` into that environment. You can install from a cloned `pysilicon` repository or from a published package source, as long as the environment you activate contains `pysilicon`.

Independent of where the virtual environment is installed, navigate to the root folder of the repository where you wish to work:

- If you are an user working in a repository with own project, say `my_hw_project`, navigate to that repository;
- If you are a PySilicon developer working on the `pysilicon` repository itself, navigate to the root of the `pysilicon` repository.

From there run:

```bash
pysilicon_mcp_setup --workspace .
```

The function will copy a file, `.vscode/mcp.json` to the repostitory root that VS Code uses a configuration file for the MCP server.  If `.vscode/mcp.json` already exists, rerun the command with `--force` to replace it:

```bash
pysilicon_mcp_setup --workspace . --force
```

The setup command discovers its own interpreter path, verifies that `pysilicon.mcp.server` can be imported from that interpreter, and then writes `.vscode/mcp.json` for the workspace.

In most cases this command only needs to be run once per workspace. If you later recreate or move the virtual environment, rerun `pysilicon_mcp_setup --workspace .` so `.vscode/mcp.json` points to the new interpreter path.

After the command is run, you can launch VS Code from the command line:

```bash
code .
```

## Testing the MCP Server is Running

The simplest way to confirm that the MCP server is running is:

1. In VS Code, Open the Command Palette.
2. Run `MCP: List Servers`.
3. Select `pysilicon`.
4. Choose `Restart`, or `Stop` and then `Start`.

After changing `mcp.json` or reinstalling packages, you can also use the above procedure to restart the server in VS Code:

You usually do not need to close and reopen VS Code unless you changed the environment after VS Code was already open.

---

## OpenAI RAG for Semantic Example Search (Optional)

PySilicon includes a tool called `pysilicon_search_schema_examples` that lets an AI assistant (e.g. GitHub Copilot in VS Code) **semantically search** through the curated schema examples that ship with the package. For example, you can ask "find me an example with a memory address field and an enum" and get back the most relevant example code.

This search is powered by an **OpenAI-hosted vector store** — a search index that holds embedded representations of the example files. Because each user supplies their own OpenAI API key, the vector store lives in **your** OpenAI account and your usage is billed to you. See [OpenAI Setup](./openai.md) for details on obtaining a key.

The RAG setup is **optional**. Without it:

- All other MCP tools continue to work normally.
- `pysilicon_search_schema_examples` returns a structured message explaining that the vector store is not configured.
- You can still browse and retrieve individual example files with `pysilicon_get_example_file`.

### Enabling Semantic Search with `--build-rag`

To build your personal vector store and wire it into VS Code automatically, run:

```bash
export OPENAI_API_KEY=sk-...       # set your key first
pysilicon_mcp_setup --workspace . --build-rag
```

What this does:

1. Uploads the packaged example files (`poly.py`, `hist.py`, `conv2d.py`, and a generated catalog) to your OpenAI account.
2. Creates a vector store named `pysilicon-examples` in your OpenAI project.
3. Waits for OpenAI to finish processing the files.
4. Writes the resulting vector store ID into `.vscode/mcp.json` under `servers.pysilicon.env`:

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

### Combining `--build-rag` with Other Flags

`--build-rag` respects the same flags as the base command:

| Flag | Effect |
|------|--------|
| `--force` | Overwrite an existing `.vscode/mcp.json`. |
| `--dry-run` | Print the config (with the env var set) without writing any files. |

Example — preview the config before writing:

```bash
pysilicon_mcp_setup --workspace . --build-rag --dry-run
```

### When to Rebuild

Rebuild the vector store (re-run `--build-rag`) whenever:

- You update `pysilicon` to a version that adds or changes example files.
- The old vector store has expired (OpenAI may expire unused stores; check your OpenAI dashboard).
- You rotate your OpenAI API key (old uploads are tied to the previous key's project).
