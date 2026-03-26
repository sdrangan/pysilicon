"""AI-assisted dataschema utilities for pysilicon."""

from pysilicon.ai.interface_bundle import (
    build_interface_manifest,
    generate_interface_bundle,
    interface_bundle_from_callable_symbol,
    interface_bundle_from_python_symbols,
    normalize_interface_bundle_spec,
    read_interface_manifest,
    render_interface_report,
    simulate_callable_outputs,
    validate_generated_schema_with_vitis,
    validate_interface_bundle_with_vitis,
    write_interface_manifest,
)
from pysilicon.ai.schema_codegen import (
    generate_vitis_headers,
    render_dataschema_module,
    write_dataschema_module,
)
from pysilicon.ai.schema_spec import collect_named_nodes, normalize_module_spec
from pysilicon.ai.type_inference import (
    ArrayHint,
    EnumHint,
    FloatHint,
    IntHint,
    infer_interface_bundle_from_callable,
    infer_module_spec_from_annotation,
    infer_schema_spec_from_symbol,
    load_python_symbol,
)

__all__ = [
    "ArrayHint",
    "EnumHint",
    "FloatHint",
    "IntHint",
    "build_interface_manifest",
    "collect_named_nodes",
    "generate_interface_bundle",
    "generate_vitis_headers",
    "infer_interface_bundle_from_callable",
    "infer_module_spec_from_annotation",
    "infer_schema_spec_from_symbol",
    "interface_bundle_from_callable_symbol",
    "interface_bundle_from_python_symbols",
    "load_python_symbol",
    "normalize_module_spec",
    "normalize_interface_bundle_spec",
    "read_interface_manifest",
    "render_dataschema_module",
    "render_interface_report",
    "simulate_callable_outputs",
    "validate_generated_schema_with_vitis",
    "validate_interface_bundle_with_vitis",
    "write_interface_manifest",
    "write_dataschema_module",
]
