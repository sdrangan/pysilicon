# Gain Shift Agent Demo

This example is meant to be driven by an AI agent, not by running a helper script.

The source file [gain_shift_demo.py](/Users/asheshkaji/projects/pysilicon/examples/gain_shift/gain_shift_demo.py) contains:

- a `pydantic` command model
- a callable `gain_shift_kernel(...)`
- bounded stream annotations
- sample inputs the agent can use for deterministic output generation

Your agent should read that file, infer the interface bundle from the callable, and then use the repo-local skill plus the MCP server to generate the artifacts.

## 1. Setup

Install the repo and AI extras:

```bash
cd /Users/asheshkaji/projects/pysilicon
python -m pip install -e '.[ai,dev]'
```

Add this MCP entry to your agent config.

For Codex-style config:

```toml
[projects."/Users/asheshkaji/projects/pysilicon"]
trust_level = "trusted"

[mcp_servers.pysilicon_schema]
command = "python"
args = ["-m", "pysilicon.ai.mcp_server"]
```

The repo-local skill is already present:

- `.codex/skills/pysilicon-dataschema-authoring/SKILL.md`

## 2. What To Tell The Agent

Paste this prompt into the agent:

```text
Use the repo-local `pysilicon-dataschema-authoring` skill and the `pysilicon_schema` MCP server.

Task:
1. Read `/Users/asheshkaji/projects/pysilicon/examples/gain_shift/gain_shift_demo.py`.
2. Infer an interface bundle from the callable symbol `gain_shift_kernel`.
3. Use `DEMO_SAMPLE_INPUTS` from that file as the sample inputs.
4. Evaluate the callable outputs so the generated bundle includes deterministic sample payloads for the outputs too.
5. Generate the full interface bundle into `/Users/asheshkaji/projects/pysilicon/examples/gain_shift/generated`.
6. Return:
   - the normalized bundle spec
   - generated artifact paths
   - Python validation status for each member
   - any failed headers

Important constraints:
- Do not hand-write dataschema classes first.
- Go through the deterministic bundle path:
  - `bundle_from_callable_symbol`
  - `generate_interface_bundle`
- Prefer JSON manifests unless I ask for YAML too.
```

## 3. Optional Vitis Validation Prompt

If Vitis is installed and `PYSILICON_VITIS_PATH` is configured, follow up with:

```text
Using the generated gain-shift interface bundle in `/Users/asheshkaji/projects/pysilicon/examples/gain_shift/generated`, run bundle-level Vitis validation at word width 32 and return the per-member results.
Use the deterministic MCP tool for bundle Vitis validation rather than inventing your own flow.
```

## 4. What Success Looks Like

The agent should generate:

- `examples/gain_shift/generated/specs/*.json`
- `examples/gain_shift/generated/schemas/*.py`
- `examples/gain_shift/generated/include/*.h`
- `examples/gain_shift/generated/vectors/*`
- `examples/gain_shift/generated/interface_manifest.json`
- `examples/gain_shift/generated/INTERFACE_REPORT.md`

For this example, the expected top-level member names are:

- `cmd`
- `samples`
- `gain_shift_response`
- `shifted_samples`
- `gain_shift_state`

## 5. How To Check The Result

Ask the agent to show or summarize:

- the normalized bundle spec
- `failed_headers`
- each member’s Python validation result

You can also verify directly on disk:

```bash
find /Users/asheshkaji/projects/pysilicon/examples/gain_shift/generated -maxdepth 2 -type f | sort
```

The most useful files to inspect first are:

- `examples/gain_shift/generated/INTERFACE_REPORT.md`
- `examples/gain_shift/generated/interface_manifest.json`
- `examples/gain_shift/generated/include/gain_shift_command.h`
- `examples/gain_shift/generated/include/shifted_samples.h`

## 6. Why This Demo Exists

This is the intended user path for the repository:

1. the user gives an agent a Python callable, symbols, or a prompt
2. the agent uses a constrained MCP workflow instead of inventing PySilicon syntax
3. the repo produces deterministic dataschema and Vitis-facing artifacts
4. the manifest and report make the result reviewable by humans and repeatable by other agents
