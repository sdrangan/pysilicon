"""Deterministic multi-schema interface bundle generation for PySilicon."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import shutil
import subprocess
from copy import deepcopy
from enum import IntEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any, Mapping, get_args, get_origin, get_type_hints, is_typeddict

import numpy as np

from pysilicon.ai.schema_codegen import _load_generated_module, generate_vitis_headers, write_dataschema_module
from pysilicon.ai.schema_spec import collect_named_nodes, normalize_module_spec, pascal_case, snake_case
from pysilicon.ai.type_inference import (
    infer_interface_bundle_from_callable,
    infer_schema_spec_from_symbol,
    load_python_symbol,
)
from pysilicon.hw import DataArray
from pysilicon.xilinxutils import toolchain

try:  # pragma: no cover - optional at runtime
    import yaml
except ImportError:  # pragma: no cover - optional at runtime
    yaml = None

try:  # pragma: no cover - import is covered indirectly when pydantic is installed
    from pydantic import BaseModel
except ImportError:  # pragma: no cover - optional at runtime
    BaseModel = None


MANIFEST_VERSION = 1
DEFAULT_MANIFEST_FORMATS = ("json",)
_VALID_MANIFEST_FORMATS = {"json", "yaml"}
_VALID_DIRECTIONS = {"input", "output", "state", "config", "internal"}

_TEST_RESOURCE_DIR = Path(__file__).resolve().parents[2] / "tests" / "hw" / "resources"
_SERIALIZE_CPP_PATH = _TEST_RESOURCE_DIR / "serialize_test.cpp"
_SERIALIZE_TCL_PATH = _TEST_RESOURCE_DIR / "serialize_run.tcl"


def interface_bundle_from_python_symbols(
    module_path: str | Path,
    members: list[dict[str, Any]],
    *,
    interface_name: str,
    description: str | None = None,
    assumptions: list[str] | None = None,
    word_bw_supported: list[int] | None = None,
    manifest_formats: list[str] | None = None,
) -> dict[str, Any]:
    source_path = Path(module_path)
    bundle_members: list[dict[str, Any]] = []

    for raw_member in members:
        symbol_name = raw_member["symbol_name"]
        symbol = load_python_symbol(source_path, symbol_name)
        spec = infer_schema_spec_from_symbol(
            symbol,
            module_name=raw_member.get("module_name"),
            root_type_name=raw_member.get("root_type_name") or symbol_name,
        )
        inferred_name = raw_member.get("name") or snake_case(spec["root"]["type_name"])
        direction = raw_member.get("direction", "input")
        role = raw_member.get("role") or direction
        bundle_members.append(
            {
                "name": inferred_name,
                "direction": direction,
                "role": role,
                "description": raw_member.get("description"),
                "sample_payload": deepcopy(raw_member.get("sample_payload")),
                "source": {
                    "kind": "python_symbol",
                    "module_path": str(source_path),
                    "symbol_name": symbol_name,
                },
                "spec": spec,
            }
        )

    return normalize_interface_bundle_spec(
        {
            "interface_name": interface_name,
            "description": description,
            "assumptions": assumptions or [],
            "word_bw_supported": word_bw_supported or [32, 64],
            "manifest_formats": manifest_formats or list(DEFAULT_MANIFEST_FORMATS),
            "members": bundle_members,
        }
    )


def interface_bundle_from_callable_symbol(
    module_path: str | Path,
    symbol_name: str,
    *,
    interface_name: str | None = None,
    description: str | None = None,
    assumptions: list[str] | None = None,
    word_bw_supported: list[int] | None = None,
    sample_inputs: dict[str, Any] | None = None,
    evaluate_outputs: bool = False,
    manifest_formats: list[str] | None = None,
) -> dict[str, Any]:
    source_path = Path(module_path)
    symbol = load_python_symbol(source_path, symbol_name)
    bundle = infer_interface_bundle_from_callable(
        symbol,
        interface_name=interface_name,
        description=description,
        assumptions=assumptions,
        word_bw_supported=word_bw_supported,
    )

    for member in bundle["members"]:
        member["source"]["module_path"] = str(source_path)
        member["source"]["symbol_name"] = symbol_name
        if sample_inputs is not None and member["direction"] == "input":
            if member["name"] in sample_inputs:
                member["sample_payload"] = deepcopy(sample_inputs[member["name"]])

    if evaluate_outputs:
        if sample_inputs is None:
            raise ValueError("evaluate_outputs=True requires sample_inputs.")
        output_payloads = simulate_callable_outputs(symbol, sample_inputs)
        for member in bundle["members"]:
            if member["direction"] == "output" and member["name"] in output_payloads:
                member["sample_payload"] = output_payloads[member["name"]]

    bundle["manifest_formats"] = manifest_formats or list(DEFAULT_MANIFEST_FORMATS)
    return normalize_interface_bundle_spec(bundle)


def normalize_interface_bundle_spec(bundle_spec: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(bundle_spec, Mapping):
        raise TypeError("Interface bundle spec must be a mapping.")

    interface_name_raw = bundle_spec.get("interface_name") or bundle_spec.get("name")
    if not isinstance(interface_name_raw, str) or not interface_name_raw.strip():
        raise ValueError("Interface bundle spec must include a non-empty 'interface_name'.")
    interface_name = pascal_case(interface_name_raw)

    description = bundle_spec.get("description")
    if description is not None and not isinstance(description, str):
        raise TypeError("'description' must be a string when provided.")

    assumptions_raw = bundle_spec.get("assumptions", [])
    if not isinstance(assumptions_raw, list) or any(not isinstance(item, str) for item in assumptions_raw):
        raise TypeError("'assumptions' must be a list of strings.")

    word_bw_raw = bundle_spec.get("word_bw_supported", [32, 64])
    if not isinstance(word_bw_raw, list):
        raise TypeError("'word_bw_supported' must be a list of integers.")
    word_bw_supported = sorted({int(value) for value in word_bw_raw})
    if not word_bw_supported or any(value <= 0 for value in word_bw_supported):
        raise ValueError("'word_bw_supported' must contain positive integers.")

    manifest_formats_raw = bundle_spec.get("manifest_formats", list(DEFAULT_MANIFEST_FORMATS))
    if not isinstance(manifest_formats_raw, list):
        raise TypeError("'manifest_formats' must be a list of strings.")
    manifest_formats = [str(fmt).lower() for fmt in manifest_formats_raw]
    invalid_formats = sorted(set(manifest_formats) - _VALID_MANIFEST_FORMATS)
    if invalid_formats:
        raise ValueError(f"Unsupported manifest format(s): {', '.join(invalid_formats)}.")

    members_raw = bundle_spec.get("members")
    if not isinstance(members_raw, list) or not members_raw:
        raise ValueError("Interface bundle spec must include a non-empty 'members' list.")

    output_layout = {
        "spec_dir": str(bundle_spec.get("spec_dir", "specs")),
        "schema_dir": str(bundle_spec.get("schema_dir", "schemas")),
        "include_dir": str(bundle_spec.get("include_dir", "include")),
        "vector_dir": str(bundle_spec.get("vector_dir", "vectors")),
        "report_filename": str(bundle_spec.get("report_filename", "INTERFACE_REPORT.md")),
        "manifest_stem": str(bundle_spec.get("manifest_stem", "interface_manifest")),
        "vitis_dir": str(bundle_spec.get("vitis_dir", "_vitis")),
    }

    normalized_members: list[dict[str, Any]] = []
    seen_member_names: set[str] = set()
    type_registry: dict[str, str] = {}

    for index, raw_member in enumerate(members_raw):
        if not isinstance(raw_member, Mapping):
            raise TypeError(f"Interface member at index {index} must be a mapping.")

        name = str(raw_member.get("name") or "").strip()
        if not name:
            raise ValueError(f"Interface member at index {index} must include a non-empty 'name'.")
        member_name = snake_case(name)
        if member_name in seen_member_names:
            raise ValueError(f"Duplicate interface member name '{member_name}'.")
        seen_member_names.add(member_name)

        direction = str(raw_member.get("direction", "input")).lower()
        if direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"Interface member '{member_name}' uses unsupported direction '{direction}'."
            )

        role = str(raw_member.get("role") or direction).lower()
        member_description = raw_member.get("description")
        if member_description is not None and not isinstance(member_description, str):
            raise TypeError(f"Description for interface member '{member_name}' must be a string.")

        raw_spec = raw_member.get("spec")
        if raw_spec is None:
            raise ValueError(f"Interface member '{member_name}' must include a 'spec'.")

        spec = normalize_module_spec(raw_spec)
        spec["module_name"] = snake_case(f"{interface_name}_{member_name}")
        spec["word_bw_supported"] = list(word_bw_supported)
        _register_named_types(type_registry, spec)

        normalized_members.append(
            {
                "name": member_name,
                "direction": direction,
                "role": role,
                "description": member_description,
                "sample_payload": deepcopy(raw_member.get("sample_payload")),
                "source": deepcopy(raw_member.get("source")) if raw_member.get("source") is not None else None,
                "spec": spec,
            }
        )

    return {
        "manifest_version": MANIFEST_VERSION,
        "interface_name": interface_name,
        "description": description,
        "assumptions": list(assumptions_raw),
        "word_bw_supported": word_bw_supported,
        "manifest_formats": manifest_formats,
        "output_layout": output_layout,
        "source": deepcopy(bundle_spec.get("source")) if bundle_spec.get("source") is not None else None,
        "members": normalized_members,
    }


def simulate_callable_outputs(symbol: Any, sample_inputs: Mapping[str, Any]) -> dict[str, Any]:
    if not callable(symbol):
        raise TypeError("simulate_callable_outputs expects a callable symbol.")

    signature = inspect.signature(symbol)
    hints = get_type_hints(symbol, include_extras=True)
    kwargs: dict[str, Any] = {}

    for parameter in signature.parameters.values():
        if parameter.name not in sample_inputs:
            raise ValueError(f"Sample input for callable parameter '{parameter.name}' is required.")
        annotation = hints.get(parameter.name, parameter.annotation)
        if annotation is inspect.Signature.empty:
            raise TypeError(f"Callable parameter '{parameter.name}' must have a type annotation.")
        kwargs[parameter.name] = _materialize_runtime_value(annotation, sample_inputs[parameter.name])

    result = symbol(**kwargs)

    return_annotation = hints.get("return", signature.return_annotation)
    if return_annotation in {inspect.Signature.empty, None}:
        return {}

    outputs = result if isinstance(result, tuple) else (result,)
    output_annotations = _split_return_annotations(return_annotation)
    if len(outputs) != len(output_annotations):
        raise ValueError("Callable runtime outputs do not match the annotated return arity.")

    payloads: dict[str, Any] = {}
    used_names: set[str] = set()
    for index, (annotation, value) in enumerate(zip(output_annotations, outputs, strict=True)):
        suggested_name = _annotation_member_name(annotation, index=index, fallback=f"output_{index}")
        member_name = _dedupe_name(suggested_name, used_names)
        used_names.add(member_name)
        payloads[member_name] = _jsonable_runtime_value(value)

    return payloads


def generate_interface_bundle(
    bundle_spec: Mapping[str, Any],
    output_dir: str | Path,
    *,
    vector_word_bw: int | None = None,
    validate_python_roundtrip: bool = True,
    validate_vitis_roundtrip: bool = False,
) -> dict[str, Any]:
    normalized = normalize_interface_bundle_spec(bundle_spec)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    layout = normalized["output_layout"]
    spec_root = output_root / layout["spec_dir"]
    schema_root = output_root / layout["schema_dir"]
    include_root = output_root / layout["include_dir"]
    vector_root = output_root / layout["vector_dir"]

    for path in (spec_root, schema_root, include_root, vector_root):
        path.mkdir(parents=True, exist_ok=True)

    vector_word_bw = int(vector_word_bw or normalized["word_bw_supported"][0])
    member_results: list[dict[str, Any]] = []

    for member in normalized["members"]:
        member_name = member["name"]
        spec = member["spec"]

        spec_path = spec_root / f"{member_name}.json"
        spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")

        module_path = schema_root / f"{spec['module_name']}.py"
        write_dataschema_module(spec, module_path)

        header_result = generate_vitis_headers(
            spec,
            include_dir=include_root,
            module_path=module_path,
            word_bw_supported=normalized["word_bw_supported"],
        )

        python_validation = None
        vector_artifacts = None
        if member.get("sample_payload") is not None:
            payload = member["sample_payload"]
            if validate_python_roundtrip:
                python_validation = _validate_generated_schema_roundtrip(
                    spec,
                    payload=payload,
                    word_bw=vector_word_bw,
                )
            vector_artifacts = _write_member_vectors(
                spec=spec,
                module_path=module_path,
                payload=payload,
                output_root=vector_root,
                member_name=member_name,
                word_bw=vector_word_bw,
            )

        member_results.append(
            {
                "name": member_name,
                "direction": member["direction"],
                "role": member["role"],
                "description": member["description"],
                "source": member["source"],
                "spec": spec,
                "spec_path": str(spec_path),
                "module_path": str(module_path),
                "header_generation": header_result,
                "python_validation": python_validation,
                "vector_artifacts": vector_artifacts,
            }
        )

    vitis_validation = None
    if validate_vitis_roundtrip:
        vitis_validation = validate_interface_bundle_with_vitis(
            normalized,
            output_dir=output_root / layout["vitis_dir"],
            word_bw=vector_word_bw,
        )

    manifest = build_interface_manifest(
        normalized,
        output_root=output_root,
        member_results=member_results,
        vector_word_bw=vector_word_bw,
        vitis_validation=vitis_validation,
    )

    manifest_paths = []
    for manifest_format in normalized["manifest_formats"]:
        suffix = ".yaml" if manifest_format == "yaml" else ".json"
        manifest_path = output_root / f"{layout['manifest_stem']}{suffix}"
        write_interface_manifest(manifest, manifest_path)
        manifest_paths.append(str(manifest_path))

    report_path = output_root / layout["report_filename"]
    report_path.write_text(render_interface_report(manifest), encoding="utf-8")

    return {
        "ok": True,
        "interface_name": normalized["interface_name"],
        "output_dir": str(output_root),
        "vector_word_bw": vector_word_bw,
        "manifest_paths": manifest_paths,
        "report_path": str(report_path),
        "members": member_results,
        "vitis_validation": vitis_validation,
        "manifest": manifest,
    }


def build_interface_manifest(
    bundle_spec: Mapping[str, Any],
    *,
    output_root: Path,
    member_results: list[dict[str, Any]],
    vector_word_bw: int,
    vitis_validation: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized = normalize_interface_bundle_spec(bundle_spec)
    manifest_members: list[dict[str, Any]] = []

    vitis_results_by_name = {}
    if vitis_validation is not None:
        for member in vitis_validation.get("members", []):
            vitis_results_by_name[member["name"]] = member

    for member in member_results:
        artifacts = {
            "spec": _artifact_entry(Path(member["spec_path"]), output_root),
            "schema_module": _artifact_entry(Path(member["module_path"]), output_root),
            "headers": [
                _artifact_entry(Path(path), output_root)
                for path in member["header_generation"]["header_paths"]
            ],
        }

        if member["vector_artifacts"] is not None:
            vector_artifacts = {
                key: _artifact_entry(Path(path), output_root)
                for key, path in member["vector_artifacts"].items()
            }
            artifacts["vectors"] = vector_artifacts

        manifest_members.append(
            {
                "name": member["name"],
                "direction": member["direction"],
                "role": member["role"],
                "description": member["description"],
                "source": member["source"],
                "root_type_name": member["spec"]["root"]["type_name"],
                "module_name": member["spec"]["module_name"],
                "generated_types": list(member["header_generation"]["generated_types"]),
                "failed_headers": list(member["header_generation"]["failed_headers"]),
                "python_validation": deepcopy(member["python_validation"]),
                "vitis_validation": deepcopy(vitis_results_by_name.get(member["name"])),
                "artifacts": artifacts,
            }
        )

    return {
        "manifest_version": MANIFEST_VERSION,
        "interface_name": normalized["interface_name"],
        "description": normalized["description"],
        "assumptions": list(normalized["assumptions"]),
        "word_bw_supported": list(normalized["word_bw_supported"]),
        "vector_word_bw": vector_word_bw,
        "manifest_formats": list(normalized["manifest_formats"]),
        "source": deepcopy(normalized["source"]),
        "members": manifest_members,
    }


def write_interface_manifest(manifest: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()
    if suffix == ".json":
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return path

    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is not installed. Use JSON manifests or install PyYAML.")
        path.write_text(yaml.safe_dump(dict(manifest), sort_keys=False), encoding="utf-8")
        return path

    raise ValueError(f"Unsupported manifest file extension for {path}.")


def read_interface_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path)
    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8")

    if suffix == ".json":
        return json.loads(raw)

    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is not installed. Cannot read YAML manifests.")
        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            raise TypeError("Manifest root must deserialize to a mapping.")
        return data

    raise ValueError(f"Unsupported manifest file extension for {path}.")


def validate_generated_schema_with_vitis(
    spec: Mapping[str, Any],
    *,
    payload: Any,
    work_dir: str | Path,
    word_bw: int = 32,
) -> dict[str, Any]:
    normalized = normalize_module_spec(spec)
    workspace = Path(work_dir)
    workspace.mkdir(parents=True, exist_ok=True)

    module_path = workspace / f"{normalized['module_name']}.py"
    write_dataschema_module(normalized, module_path)
    header_result = generate_vitis_headers(
        normalized,
        include_dir=workspace,
        module_path=module_path,
        word_bw_supported=[word_bw],
    )

    module = _load_generated_module(module_path, normalized["module_name"])
    root_cls = getattr(module, normalized["root"]["type_name"])
    schema = _schema_instance_from_payload(root_cls, payload)

    input_payload_path = workspace / "packet_input.json"
    schema.to_json(file_path=input_payload_path, indent=2)

    packed = schema.serialize(word_bw=word_bw)
    words_path = workspace / "packet_words.txt"
    save_dtype = np.uint32 if word_bw <= 32 else np.uint64
    np.savetxt(words_path, np.asarray(packed).astype(save_dtype), fmt="%u")

    cpp_src = (
        _SERIALIZE_CPP_PATH.read_text(encoding="utf-8")
        .replace("__HEADER__", root_cls.resolved_include_filename())
        .replace("__EXTRA_INCLUDES__", "")
        .replace("__PACKET_CLASS__", root_cls.__name__)
        .replace("__WORD_BW__", str(word_bw))
        .replace("__RW_ARGS__", _rw_args(root_cls))
    )
    cpp_path = workspace / "serialize_test.cpp"
    cpp_path.write_text(cpp_src, encoding="utf-8")

    tcl_path = workspace / "serialize_run.tcl"
    shutil.copy(_SERIALIZE_TCL_PATH, tcl_path)

    if not toolchain.find_vitis_path():
        return {
            "ok": False,
            "skipped": True,
            "reason": "Vitis installation not found.",
            "word_bw": word_bw,
            "workspace": str(workspace),
            "header_generation": header_result,
        }

    output_json_path = workspace / "packet_out.json"
    try:
        completed = toolchain.run_vitis_hls(tcl_path, work_dir=workspace)
    except subprocess.CalledProcessError as exc:
        return {
            "ok": False,
            "skipped": False,
            "word_bw": word_bw,
            "workspace": str(workspace),
            "header_generation": header_result,
            "error": str(exc),
            "stdout": exc.stdout,
            "stderr": exc.stderr,
        }
    except RuntimeError as exc:
        return {
            "ok": False,
            "skipped": True,
            "reason": str(exc),
            "word_bw": word_bw,
            "workspace": str(workspace),
            "header_generation": header_result,
        }

    restored = root_cls().from_json(output_json_path)
    return {
        "ok": bool(restored.is_close(schema)),
        "skipped": False,
        "word_bw": word_bw,
        "workspace": str(workspace),
        "header_generation": header_result,
        "output_json_path": str(output_json_path),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def validate_interface_bundle_with_vitis(
    bundle_spec: Mapping[str, Any],
    *,
    output_dir: str | Path,
    word_bw: int = 32,
) -> dict[str, Any]:
    normalized = normalize_interface_bundle_spec(bundle_spec)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    member_results: list[dict[str, Any]] = []
    skipped = False
    for member in normalized["members"]:
        if member.get("sample_payload") is None:
            member_results.append(
                {
                    "name": member["name"],
                    "skipped": True,
                    "reason": "No sample_payload provided for this interface member.",
                }
            )
            continue

        result = validate_generated_schema_with_vitis(
            member["spec"],
            payload=member["sample_payload"],
            work_dir=output_root / member["name"],
            word_bw=word_bw,
        )
        result["name"] = member["name"]
        member_results.append(result)
        skipped = skipped or result.get("skipped", False)

    overall_ok = all(result.get("ok", False) or result.get("skipped", False) for result in member_results)
    return {
        "ok": overall_ok,
        "skipped": skipped,
        "word_bw": word_bw,
        "output_dir": str(output_root),
        "members": member_results,
    }


def render_interface_report(manifest: Mapping[str, Any]) -> str:
    lines = [
        f"# {manifest['interface_name']} AI Interface Bundle Report",
        "",
    ]

    if manifest.get("description"):
        lines.extend([str(manifest["description"]), ""])

    lines.extend(
        [
            "## Summary",
            "",
            f"- Supported word widths: {', '.join(str(value) for value in manifest['word_bw_supported'])}",
            f"- Vector serialization width: {manifest['vector_word_bw']}",
            f"- Members: {len(manifest['members'])}",
            "",
            "## Members",
            "",
            "| Name | Direction | Role | Root Type | Python Validation | Vitis Validation |",
            "|------|-----------|------|-----------|-------------------|------------------|",
        ]
    )

    for member in manifest["members"]:
        py_status = _validation_status(member.get("python_validation"))
        vitis_status = _validation_status(member.get("vitis_validation"))
        lines.append(
            f"| {member['name']} | {member['direction']} | {member['role']} | "
            f"{member['root_type_name']} | {py_status} | {vitis_status} |"
        )

    if manifest.get("assumptions"):
        lines.extend(["", "## Assumptions", ""])
        for assumption in manifest["assumptions"]:
            lines.append(f"- {assumption}")

    lines.extend(["", "## Artifacts", ""])
    for member in manifest["members"]:
        lines.extend(
            [
                f"### {member['name']}",
                "",
                f"- Spec: `{member['artifacts']['spec']['path']}`",
                f"- Schema module: `{member['artifacts']['schema_module']['path']}`",
            ]
        )
        for header in member["artifacts"]["headers"]:
            lines.append(f"- Header: `{header['path']}`")
        vectors = member["artifacts"].get("vectors", {})
        for key, artifact in vectors.items():
            lines.append(f"- {key.replace('_', ' ').title()}: `{artifact['path']}`")
        if member.get("failed_headers"):
            for failure in member["failed_headers"]:
                lines.append(
                    f"- Failed header for `{failure['schema_symbol']}`: {failure['error']}"
                )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _write_member_vectors(
    *,
    spec: Mapping[str, Any],
    module_path: Path,
    payload: Any,
    output_root: Path,
    member_name: str,
    word_bw: int,
) -> dict[str, str]:
    module = _load_generated_module(module_path, spec["module_name"])
    root_cls = getattr(module, spec["root"]["type_name"])
    schema = _schema_instance_from_payload(root_cls, payload)

    payload_json_path = output_root / f"{member_name}_payload.json"
    schema.to_json(file_path=payload_json_path, indent=2)

    packed = schema.serialize(word_bw=word_bw)
    save_dtype = np.uint32 if word_bw <= 32 else np.uint64
    words_array = np.asarray(packed).astype(save_dtype)

    words_txt_path = output_root / f"{member_name}_words_{word_bw}.txt"
    np.savetxt(words_txt_path, words_array, fmt="%u")

    words_bin_path = output_root / f"{member_name}_words_{word_bw}.bin"
    words_array.tofile(words_bin_path)

    return {
        "payload_json": str(payload_json_path),
        "words_txt": str(words_txt_path),
        "words_bin": str(words_bin_path),
    }


def _validate_generated_schema_roundtrip(
    spec: Mapping[str, Any],
    *,
    payload: Any,
    word_bw: int,
) -> dict[str, Any]:
    normalized = normalize_module_spec(spec)
    with TemporaryDirectory(prefix="pysilicon_ai_bundle_validate_") as tmpdir:
        module_path = Path(tmpdir) / f"{normalized['module_name']}.py"
        write_dataschema_module(normalized, module_path)
        module = _load_generated_module(module_path, normalized["module_name"])
        root_cls = getattr(module, normalized["root"]["type_name"])
        schema = _schema_instance_from_payload(root_cls, payload)

        packed = schema.serialize(word_bw=word_bw)
        restored = _deserialize_target_instance(root_cls)
        restored.deserialize(packed, word_bw=word_bw)

        return {
            "ok": bool(restored.is_close(schema)),
            "word_bw": word_bw,
            "packed_words": _jsonable_runtime_value(packed),
            "root_type_name": normalized["root"]["type_name"],
        }


def _schema_instance_from_payload(root_cls: type[Any], payload: Any) -> Any:
    schema = root_cls()
    if isinstance(schema, DataArray):
        schema.val = payload
    else:
        schema.from_dict(payload)
    return schema


def _deserialize_target_instance(root_cls: type[Any]) -> Any:
    schema = root_cls()
    if isinstance(schema, DataArray) and not schema.static:
        schema.val = _full_array_value(root_cls)
    return schema


def _full_array_value(array_cls: type[Any]) -> Any:
    shape = tuple(int(dim) for dim in array_cls.max_shape)
    elem_init = array_cls._element_type().init_value()

    if isinstance(elem_init, np.generic):
        return np.zeros(shape, dtype=np.asarray(elem_init).dtype)

    if isinstance(elem_init, np.ndarray):
        return np.zeros(shape + tuple(elem_init.shape), dtype=elem_init.dtype)

    def build(shape_tail: tuple[int, ...]) -> Any:
        if not shape_tail:
            return array_cls._element_type().init_value()
        return [build(shape_tail[1:]) for _ in range(shape_tail[0])]

    return build(shape)


def _artifact_entry(path: Path, output_root: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(output_root)),
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _register_named_types(type_registry: dict[str, str], spec: Mapping[str, Any]) -> None:
    for node in collect_named_nodes(spec):
        node_signature = _node_signature(node)
        existing = type_registry.get(node["type_name"])
        if existing is None:
            type_registry[node["type_name"]] = node_signature
        elif existing != node_signature:
            raise ValueError(
                f"Named schema type '{node['type_name']}' is reused across bundle members with different shapes."
            )


def _node_signature(node: Mapping[str, Any]) -> str:
    data = deepcopy(dict(node))
    data.pop("name", None)
    if data["kind"] == "struct":
        data["fields"] = [_node_signature(field) for field in data["fields"]]
    elif data["kind"] == "array":
        data["element"] = _node_signature(data["element"])
    return json.dumps(data, sort_keys=True)


def _validation_status(result: Mapping[str, Any] | None) -> str:
    if result is None:
        return "not-run"
    if result.get("skipped"):
        return f"skipped ({result.get('reason', 'no-reason')})"
    return "ok" if result.get("ok") else "failed"


def _rw_args(packet_type: type[Any]) -> str:
    if issubclass(packet_type, DataArray) and not packet_type.static:
        shape_args = ", ".join(str(int(dim)) for dim in tuple(packet_type.max_shape))
        return f", {shape_args}" if shape_args else ""
    return ""


def _materialize_runtime_value(annotation: Any, payload: Any) -> Any:
    raw_annotation, _ = _unwrap_annotated(annotation)
    origin = get_origin(raw_annotation)

    if origin in {list, tuple}:
        element_annotation = get_args(raw_annotation)[0]
        return [_materialize_runtime_value(element_annotation, value) for value in payload]

    if raw_annotation is int:
        return int(payload)
    if raw_annotation is float:
        return float(payload)
    if isinstance(raw_annotation, type) and issubclass(raw_annotation, IntEnum):
        return raw_annotation(payload)

    if isinstance(raw_annotation, type) and dataclasses.is_dataclass(raw_annotation):
        values = {}
        hints = get_type_hints(raw_annotation, include_extras=True)
        for field in dataclasses.fields(raw_annotation):
            values[field.name] = _materialize_runtime_value(hints[field.name], payload[field.name])
        return raw_annotation(**values)

    if isinstance(raw_annotation, type) and is_typeddict(raw_annotation):
        return dict(payload)

    if BaseModel is not None and isinstance(raw_annotation, type) and issubclass(raw_annotation, BaseModel):
        return raw_annotation.model_validate(payload)

    return payload


def _jsonable_runtime_value(value: Any) -> Any:
    if BaseModel is not None and isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable_runtime_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, np.ndarray):
        return [_jsonable_runtime_value(item) for item in value.tolist()]
    if isinstance(value, np.void):
        return {name: _jsonable_runtime_value(value[name]) for name in value.dtype.names or ()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, IntEnum):
        return int(value.value)
    if isinstance(value, tuple):
        return [_jsonable_runtime_value(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_runtime_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable_runtime_value(val) for key, val in value.items()}
    return value


def _split_return_annotations(annotation: Any) -> list[Any]:
    raw_annotation, _ = _unwrap_annotated(annotation)
    origin = get_origin(raw_annotation)
    if origin is tuple:
        args = [arg for arg in get_args(raw_annotation) if arg is not Ellipsis]
        if args:
            return list(args)
    return [annotation]


def _annotation_member_name(annotation: Any, *, index: int, fallback: str) -> str:
    raw_annotation, extras = _unwrap_annotated(annotation)
    for extra in extras:
        if dataclasses.is_dataclass(extra):
            metadata = dataclasses.asdict(extra)
        elif isinstance(extra, dict):
            metadata = extra
        else:
            continue
        type_name = metadata.get("type_name")
        if isinstance(type_name, str):
            return snake_case(type_name)

    if isinstance(raw_annotation, type):
        return snake_case(raw_annotation.__name__)
    return fallback


def _unwrap_annotated(annotation: Any) -> tuple[Any, list[Any]]:
    extras: list[Any] = []
    while get_origin(annotation) is Annotated:
        args = get_args(annotation)
        annotation = args[0]
        extras.extend(args[1:])
    return annotation, extras


def _dedupe_name(name: str, seen: set[str]) -> str:
    if name not in seen:
        return name
    index = 2
    while f"{name}_{index}" in seen:
        index += 1
    return f"{name}_{index}"
