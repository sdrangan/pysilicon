"""Deterministic schema-planning and validation helpers for pysilicon MCP tools."""
from __future__ import annotations

import ast
from enum import IntEnum
from typing import Any

from pysilicon.hw import DataArray, DataField, DataList, DataSchema, EnumField, FloatField, IntField, MemAddr


def get_schema_draft_plan(
	task: str | None = None,
	workspace_root: str | None = None,
) -> dict:
	"""Return a deterministic workflow for drafting a new schema.

	The returned plan is intentionally generic and tool-friendly. It does not
	search, rank, or recommend specific example IDs. The optional
	``workspace_root`` argument is used only to tailor the first step's
	instructions.
	"""
	if workspace_root:
		first_step_instructions = (
			f"Check the workspace at {workspace_root} for related schemas when "
			"available, or use pysilicon_list_schema_examples and "
			"pysilicon_get_schema_example to review comparable patterns before "
			"drafting new code."
		)
	else:
		first_step_instructions = (
			"Use pysilicon_list_schema_examples and "
			"pysilicon_get_schema_example to review comparable patterns before "
			"drafting new code."
		)

	summary = (
		"Use this plan to draft a new pysilicon schema from a natural-language "
		"request by reviewing similar patterns, identifying required elements, "
		"drafting the main schema and supporting types, and validating the result."
	)
	if task:
		summary = f"Schema drafting plan for request: {task}"

	return {
		"summary": summary,
		"steps": [
			{
				"goal": "Retrieve similar schema patterns",
				"instructions": first_step_instructions,
				"recommended_tools": [
					"pysilicon_list_schema_examples",
					"pysilicon_get_schema_example",
				],
			},
			{
				"goal": "Determine required schema elements",
				"instructions": (
					"List the fields and structures implied by the request, including "
					"command fields, nested records, arrays, memory-address fields, "
					"scalar parameters, enums, and repeated structures."
				),
				"recommended_tools": [],
			},
			{
				"goal": "Draft the main schema",
				"instructions": (
					"Write the primary schema using the closest structural pattern and "
					"reuse naming and layout conventions from similar examples when "
					"practical."
				),
				"recommended_tools": [],
			},
			{
				"goal": "Draft missing supporting schemas",
				"instructions": (
					"If the schema depends on nested or reusable element types that do "
					"not already exist, draft those supporting schemas as part of the "
					"same design pass."
				),
				"recommended_tools": [],
			},
			{
				"goal": "Validate the schema and record assumptions",
				"instructions": (
					"Run pysilicon_validate_schema on the completed draft to catch "
					"structural or typing issues before use, and document any assumed "
					"scalar types, array lengths, optional fields, or address "
					"representations for later review."
				),
				"recommended_tools": ["pysilicon_validate_schema"],
			},
		],
	}


def validate_schema(schema: str, workspace_root: str | None = None) -> dict:
	"""Validate a drafted pysilicon schema provided as Python source text.

	This helper is deterministic. It parses the input as Python, executes it in
	a constrained namespace populated with the public pysilicon schema classes,
	and then validates any discovered schema classes using existing class-level
	metadata helpers.
	"""
	path = workspace_root
	errors: list[dict[str, Any]] = []
	warnings: list[dict[str, Any]] = []

	if not isinstance(schema, str) or not schema.strip():
		errors.append(
			_diagnostic(
				message="Schema text must be a non-empty string.",
				code="empty_schema",
				path=path,
			)
		)
		return _validation_result(
			valid=False,
			summary="Schema validation failed with 1 error.",
			errors=errors,
			warnings=warnings,
			schema_info=None,
		)

	try:
		tree = ast.parse(schema, filename="<schema>")
	except SyntaxError as exc:
		errors.append(
			_diagnostic(
				message=exc.msg,
				code="syntax_error",
				line=exc.lineno,
				column=exc.offset,
				path=path,
			)
		)
		return _validation_result(
			valid=False,
			summary="Schema validation failed with 1 error.",
			errors=errors,
			warnings=warnings,
			schema_info=None,
		)

	class_lines = {
		node.name: node.lineno
		for node in ast.walk(tree)
		if isinstance(node, ast.ClassDef)
	}

	namespace = _build_validation_namespace()
	initial_names = set(namespace)

	try:
		exec(compile(tree, "<schema>", "exec"), namespace)
	except Exception as exc:  # noqa: BLE001
		errors.append(_diagnostic_from_exception(exc, code="execution_error", path=path))
		return _validation_result(
			valid=False,
			summary=f"Schema validation failed with {len(errors)} error.",
			errors=errors,
			warnings=warnings,
			schema_info=None,
		)

	schema_classes = _discover_schema_classes(namespace, initial_names, class_lines)
	if not schema_classes:
		errors.append(
			_diagnostic(
				message="No pysilicon schema classes were defined in the provided text.",
				code="no_schema_found",
				path=path,
			)
		)
		return _validation_result(
			valid=False,
			summary="Schema validation failed with 1 error.",
			errors=errors,
			warnings=warnings,
			schema_info=None,
		)

	for entry in schema_classes:
		line = entry["line"]
		schema_cls = entry["class"]
		try:
			_validate_schema_class(schema_cls)
		except Exception as exc:  # noqa: BLE001
			errors.append(
				_diagnostic(
					message=f"{schema_cls.__name__}: {exc}",
					code="schema_validation_error",
					line=line,
					path=path,
				)
			)

	primary_info = _schema_info(schema_classes[0]["class"])

	if len(schema_classes) > 1:
		warnings.append(
			_diagnostic(
				message=(
					"Multiple schema classes were found. schema_info describes the first "
					"discovered schema class."
				),
				code="multiple_schema_classes",
				line=schema_classes[0]["line"],
				path=path,
			)
		)

	if errors:
		summary = f"Schema validation failed with {len(errors)} error{'s' if len(errors) != 1 else ''}."
		if primary_info is None:
			primary_info = None
		return _validation_result(
			valid=False,
			summary=summary,
			errors=errors,
			warnings=warnings,
			schema_info=primary_info,
		)

	summary = "Schema is valid."
	if warnings:
		summary = f"Schema is valid with {len(warnings)} warning{'s' if len(warnings) != 1 else ''}."

	return _validation_result(
		valid=True,
		summary=summary,
		errors=errors,
		warnings=warnings,
		schema_info=primary_info,
	)


def _build_validation_namespace() -> dict[str, Any]:
	return {
		"__builtins__": __builtins__,
		"__name__": "pysilicon.mcp._schema_validation",
		"DataSchema": DataSchema,
		"DataField": DataField,
		"IntField": IntField,
		"MemAddr": MemAddr,
		"FloatField": FloatField,
		"EnumField": EnumField,
		"DataList": DataList,
		"DataArray": DataArray,
		"IntEnum": IntEnum,
	}


def _discover_schema_classes(
	namespace: dict[str, Any],
	initial_names: set[str],
	class_lines: dict[str, int],
) -> list[dict[str, Any]]:
	classes: list[dict[str, Any]] = []
	for name, value in namespace.items():
		if name in initial_names or name.startswith("_"):
			continue
		if not isinstance(value, type) or not issubclass(value, DataSchema):
			continue
		if value in {DataSchema, DataField, IntField, MemAddr, FloatField, EnumField, DataList, DataArray}:
			continue
		classes.append(
			{
				"name": name,
				"class": value,
				"line": class_lines.get(name),
			}
		)

	classes.sort(key=lambda item: (item["line"] is None, item["line"] or 0, item["name"]))
	return classes


def _validate_schema_class(schema_cls: type[DataSchema]) -> None:
	schema_cls.get_bitwidth()
	schema_cls.init_value()

	if issubclass(schema_cls, DataList):
		schema_cls._iter_elements()
		schema_cls()
		return

	if issubclass(schema_cls, DataArray):
		schema_cls._element_type()
		schema_cls._normalized_shape()
		schema_cls()


def _schema_info(schema_cls: type[DataSchema]) -> dict[str, Any]:
	if issubclass(schema_cls, DataList):
		try:
			field_names = [name for name, _ in schema_cls._iter_elements()]
			field_count: int | None = len(field_names)
		except Exception:  # noqa: BLE001
			field_names = []
			field_count = None
		return {
			"name": schema_cls.__name__,
			"kind": "DataList",
			"field_count": field_count,
			"field_names": field_names,
		}

	if issubclass(schema_cls, DataArray):
		try:
			member_name = schema_cls._member_name()
		except Exception:  # noqa: BLE001
			member_name = None
		return {
			"name": schema_cls.__name__,
			"kind": "DataArray",
			"field_count": 1 if member_name is not None else None,
			"field_names": [member_name] if member_name is not None else [],
		}

	if issubclass(schema_cls, DataField):
		return {
			"name": schema_cls.__name__,
			"kind": "DataField",
			"field_count": 0,
			"field_names": [],
		}

	return {
		"name": schema_cls.__name__,
		"kind": "DataSchema",
		"field_count": None,
		"field_names": [],
	}


def _diagnostic(
	*,
	message: str,
	code: str | None,
	line: int | None = None,
	column: int | None = None,
	path: str | None = None,
) -> dict[str, Any]:
	location = None
	if line is not None or column is not None or path is not None:
		location = {
			"line": line,
			"column": column,
			"path": path,
		}
	return {
		"message": message,
		"code": code,
		"location": location,
	}


def _diagnostic_from_exception(exc: Exception, *, code: str | None, path: str | None) -> dict[str, Any]:
	line = None
	column = None
	if isinstance(exc, SyntaxError):
		line = exc.lineno
		column = exc.offset
	traceback_obj = exc.__traceback__
	while traceback_obj is not None:
		if traceback_obj.tb_frame.f_code.co_filename == "<schema>":
			line = traceback_obj.tb_lineno
		traceback_obj = traceback_obj.tb_next
	return _diagnostic(
		message=str(exc),
		code=code,
		line=line,
		column=column,
		path=path,
	)


def _validation_result(
	*,
	valid: bool,
	summary: str,
	errors: list[dict[str, Any]],
	warnings: list[dict[str, Any]],
	schema_info: dict[str, Any] | None,
) -> dict:
	return {
		"valid": valid,
		"summary": summary,
		"errors": errors,
		"warnings": warnings,
		"schema_info": schema_info,
	}
