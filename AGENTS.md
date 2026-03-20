If the task is to generate or update PySilicon dataschemas from prompts or Python types, use the local skill at `.codex/skills/pysilicon-dataschema-authoring/SKILL.md`.

Prefer the deterministic pipeline in `pysilicon/ai/` over hand-writing dataschema classes:
1. Constrain the request into the schema spec format.
2. Validate or infer the spec with the MCP/server helpers in `pysilicon.ai.mcp_server`.
3. Generate the Python dataschema module.
4. Generate headers or run the roundtrip validator.

The initial AI pipeline supports:
- `struct`, `array`, `int`, `float`, and `enum` nodes
- dataclass and `TypedDict` symbol inference
- bounded arrays with explicit `max_shape`

It does not currently support nested arrays or arbitrary unions.
