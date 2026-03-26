# PySilicon AI Agent Workflow

`pysilicon/ai/` is the deterministic layer that an LLM agent should use when turning prompts, Python symbols, or callable signatures into PySilicon dataschemas and Vitis-facing artifacts.

The primary audience for this directory is not a Python developer calling helper APIs directly. The intended user journey is:

1. give an AI agent a prompt, callable, model, or notebook symbol
2. have the agent use the repo-local skill plus the repo-local MCP server
3. force the agent through constrained schema or bundle generation steps
4. get generated code, headers, vectors, manifests, and reports back in a repeatable form

## What The Agent Can Do

The current workflow supports:

- inferring single schemas from:
  - dataclasses
  - `TypedDict`
  - `pydantic.BaseModel`
  - NumPy structured dtypes
  - symbols loaded from `.py` files or `.ipynb` notebooks
- inferring whole accelerator interfaces from callable signatures
- generating:
  - normalized specs
  - PySilicon dataschema modules
  - Vitis HLS headers
  - sample vectors
  - JSON or YAML manifests
  - markdown reports
- validating:
  - Python roundtrip serialization
  - optional Vitis roundtrip behavior

## Agent Setup

Install the repository with the AI extras:

```bash
cd /Users/asheshkaji/projects/pysilicon
python -m pip install -e '.[ai,dev]'
```

Add the local MCP server to your agent config.

For Codex-style config:

```toml
[projects."/Users/asheshkaji/projects/pysilicon"]
trust_level = "trusted"

[mcp_servers.pysilicon_schema]
command = "python"
args = ["-m", "pysilicon.ai.mcp_server"]
```

The local skill is already part of the repo:

- `.codex/skills/pysilicon-dataschema-authoring/SKILL.md`

## Required Agent Behavior

The agent should not invent PySilicon syntax from scratch if the deterministic workflow can handle the request.

Use this order:

1. infer or constrain into a normalized spec or bundle
2. validate
3. generate dataschema modules
4. generate headers
5. validate generated output
6. emit manifests and reports

For interface-level tasks, prefer the bundle workflow over isolated per-type generation.

## MCP Tools The Agent Should Use

Single-schema path:

- `validate_schema_spec`
- `spec_from_python_symbol`
- `generate_dataschema_module`
- `generate_schema_headers`
- `validate_generated_schema`
- `validate_schema_with_vitis`

Interface-bundle path:

- `bundle_from_python_symbols`
- `bundle_from_callable_symbol`
- `generate_interface_bundle`
- `validate_bundle_with_vitis`
- `load_interface_manifest`

## Copy-Paste Prompt Template

Use this when you already have a callable or source file in the repo:

```text
Use the repo-local `pysilicon-dataschema-authoring` skill and the `pysilicon_schema` MCP server.

Task:
1. Read the source file I specify.
2. Infer a deterministic PySilicon interface bundle from the callable or symbols in that file.
3. Use the provided sample inputs to evaluate outputs when possible.
4. Generate the full bundle:
   - normalized specs
   - dataschema Python modules
   - headers
   - vectors
   - manifest
   - markdown report
5. Validate the generated results with Python roundtrip validation.
6. Return:
   - the normalized bundle spec
   - artifact paths
   - per-member validation status
   - any `failed_headers`

Important constraints:
- Do not hand-write dataschema classes first.
- Prefer the deterministic bundle path:
  - `bundle_from_callable_symbol` or `bundle_from_python_symbols`
  - `generate_interface_bundle`
```

## Gain-Shift Demo

The agent-facing example is in:

- [gain_shift_demo.py](/Users/asheshkaji/projects/pysilicon/examples/gain_shift/gain_shift_demo.py)
- [README.md](/Users/asheshkaji/projects/pysilicon/examples/gain_shift/README.md)

That example is intentionally structured for an agent:

- the file contains a callable source of truth
- it includes deterministic sample inputs
- the README gives the exact prompt to paste into the agent
- the output tree is easy to inspect after generation

## Expected Generated Tree

For a bundle run, the agent should typically produce:

```text
generated/
├── specs/
├── schemas/
├── include/
├── vectors/
├── interface_manifest.json
└── INTERFACE_REPORT.md
```

## Validation

A good agent run should end with:

- `failed_headers == []`
- Python validation success for each generated member
- a manifest that records generated artifacts and hashes

If Vitis is available, the agent can then run compile-level validation as a second phase rather than mixing that into the initial generation step.
