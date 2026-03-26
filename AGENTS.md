If the task is to generate or update PySilicon dataschemas from prompts or Python types, use the local skill at `.codex/skills/pysilicon-dataschema-authoring/SKILL.md`.

Prefer the deterministic pipeline in `pysilicon/ai/` over hand-writing dataschema classes:
1. Constrain the request into the schema spec format.
2. Validate or infer the spec with the MCP/server helpers in `pysilicon.ai.mcp_server`.
3. Generate the Python dataschema module.
4. Generate headers or run the roundtrip validator.
5. For multi-port accelerators, prefer the interface bundle helpers over generating isolated schemas.

Generated dataschema modules should follow the merged class-specialized API from `pysilicon.hw`:
- `IntField.specialize(...)`, `FloatField.specialize(...)`, `EnumField.specialize(...)`
- `DataArray.specialize(...)`
- `class TypeName(DataList): elements = {...}`

The initial AI pipeline supports:
- `struct`, `array`, `int`, `float`, and `enum` nodes
- dataclass, `TypedDict`, `pydantic`, NumPy structured dtype, notebook, and callable-signature inference
- bounded arrays with explicit `max_shape`
- interface bundle generation with manifests, reports, vectors, and optional Vitis validation

It does not currently support nested arrays or arbitrary unions.
