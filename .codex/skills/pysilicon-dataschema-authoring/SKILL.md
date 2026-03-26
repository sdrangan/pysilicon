---
name: pysilicon-dataschema-authoring
description: Use when generating or updating PySilicon dataschema classes from freeform hardware-schema prompts or from Python dataclasses/TypedDicts, and when you need to drive the local PySilicon schema MCP server deterministically.
---

# PySilicon Dataschema Authoring

Use this skill when the user wants a PySilicon dataschema generated from:
- a freeform prompt that describes packet/layout fields
- an existing Python dataclass or `TypedDict`
- a repeatable AI-agent workflow around dataschema codegen

## Goal

Do not hand-write arbitrary `DataList` or `DataArray` code unless the deterministic helpers fail.

The stable path is:
1. `prompt or Python symbol`
2. constrained schema spec
3. validation
4. generated PySilicon dataschema module
5. optional header generation or Python roundtrip validation
6. optional interface-bundle generation with manifests, reports, vectors, and Vitis validation

Generated modules should target the merged class-specialized dataschema API:
- scalar fields via `IntField.specialize(...)`, `FloatField.specialize(...)`, or `EnumField.specialize(...)`
- arrays via `DataArray.specialize(...)`
- structs via `class TypeName(DataList): elements = {...}`

## Workflow

### 1. Choose the source path

- If the user already has a Python dataclass or `TypedDict`, use the MCP helper `spec_from_python_symbol`.
- If the user has a `pydantic` model, a NumPy structured dtype, a notebook symbol, or a callable signature, use the deterministic inference helpers instead of hand-translating them.
- If the user gives a prompt, first translate it into the constrained schema spec described below.

### 2. Use the constrained schema spec

Supported node kinds:
- `struct`
- `array`
- `int`
- `float`
- `enum`

The spec lives in the deterministic layer under `pysilicon/ai/schema_spec.py`.

Read [references/spec-format.md](references/spec-format.md) only if you need the exact JSON shape or Python-type hint patterns.

### 3. Validate before generating code

Call the MCP helper `validate_schema_spec` first.

Do not skip this step for prompt-derived specs. This is the guardrail that keeps outputs consistent across agents.

### 4. Generate code through the server

Use:
- `generate_dataschema_module` to create the Python dataschema source
- `generate_schema_headers` to emit one header per named struct/array type
- `validate_generated_schema` to roundtrip a payload through Python serialization
- `bundle_from_python_symbols` or `bundle_from_callable_symbol` to infer an accelerator interface bundle
- `generate_interface_bundle` to emit specs, modules, headers, vectors, manifests, and a markdown report
- `validate_schema_with_vitis` or `validate_bundle_with_vitis` for compile-level validation

Check `failed_headers` after header generation. A successful run should usually emit semantic header names derived from schema `type_name` values, rather than generic element-type names.

### 5. State assumptions explicitly

If the prompt omits important hardware details:
- integer bitwidth/sign
- float bitwidth
- enum values
- array bounds

either ask, or use the documented defaults and say so explicitly.

Defaults in the initial pipeline:
- `int`: 32-bit signed
- `float`: 32-bit
- `enum` bitwidth: derived from the largest enum value
- `word_bw_supported`: `[32, 64]`

## Constraints

- Arrays must be bounded with explicit `max_shape`.
- The initial pipeline does not support nested arrays.
- The initial Python-type inference supports dataclasses and `TypedDict` classes.
- Reuse the same `type_name` only when the shape is actually identical.

## MCP Server

The repo-local server entrypoint is:

```bash
python -m pysilicon.ai.mcp_server
```

For Codex-style local config, the command block is:

```toml
[mcp_servers.pysilicon_schema]
command = "python"
args = ["-m", "pysilicon.ai.mcp_server"]
```
