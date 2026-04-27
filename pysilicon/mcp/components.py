"""
Deterministic glossary of pysilicon schema vocabulary.

``get_components`` returns a dict with canonical descriptions and keywords
for all core schema/field classes and common patterns.  This tool is
network-free and always returns the same result.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Component glossary
# ---------------------------------------------------------------------------

_COMPONENTS: list[dict] = [
    # -----------------------------------------------------------------------
    # Core schema base classes
    # -----------------------------------------------------------------------
    {
        "name": "DataSchema",
        "kind": "base_class",
        "description": (
            "Root base class for all pysilicon schemas.  Every DataList, "
            "DataArray, and DataField subclass ultimately inherits from "
            "DataSchema.  Provides get_bitwidth(), init_value(), and "
            "code-generation helpers."
        ),
        "keywords": ["base", "schema", "bitwidth", "init_value"],
    },
    {
        "name": "DataList",
        "kind": "schema_class",
        "description": (
            "A fixed-layout record (struct) whose fields are declared in an "
            "``elements`` dict.  Each entry maps a field name to a sub-schema "
            "and a human-readable description.  Use DataList for command, "
            "response, debug, and header structs."
        ),
        "keywords": ["struct", "record", "command", "response", "header", "elements", "DataList"],
    },
    {
        "name": "DataArray",
        "kind": "schema_class",
        "description": (
            "An array of homogeneous elements.  Subclasses set ``element_type`` "
            "(a DataField or DataArray), ``max_shape`` (tuple of ints), and "
            "optionally ``static = True`` for fixed-size arrays.  Use DataArray "
            "for arrays of coefficients, pixel rows, etc."
        ),
        "keywords": ["array", "vector", "coefficients", "static", "element_type", "max_shape", "DataArray"],
    },
    {
        "name": "DataField",
        "kind": "field_class",
        "description": (
            "Leaf scalar field base class.  Subclasses (IntField, FloatField, "
            "EnumField, MemAddr) add type-specific attributes.  Use "
            "``SomeField.specialize(...)`` to create named specializations with "
            "fixed parameters."
        ),
        "keywords": ["field", "scalar", "leaf", "specialize", "DataField"],
    },
    # -----------------------------------------------------------------------
    # Concrete field classes
    # -----------------------------------------------------------------------
    {
        "name": "IntField",
        "kind": "field_class",
        "description": (
            "Integer scalar field.  Specialize with ``bitwidth`` (int) and "
            "``signed`` (bool).  Example: "
            "``TxIdField = IntField.specialize(bitwidth=16, signed=False)``."
        ),
        "keywords": ["integer", "int", "bitwidth", "signed", "unsigned", "IntField", "specialize"],
    },
    {
        "name": "FloatField",
        "kind": "field_class",
        "description": (
            "Floating-point scalar field.  Specialize with ``bitwidth`` (32 or "
            "64) and optionally ``include_dir``.  Maps to IEEE 754 float/double "
            "in generated C headers."
        ),
        "keywords": ["float", "double", "floating-point", "IEEE754", "FloatField", "bitwidth"],
    },
    {
        "name": "EnumField",
        "kind": "field_class",
        "description": (
            "Enum-backed integer field.  Specialize with ``enum_type`` (an "
            "``IntEnum`` subclass), optional ``bitwidth``, and "
            "``include_filename``.  Generates a C enum in the header.  Use for "
            "status codes, error codes, and event types."
        ),
        "keywords": [
            "enum", "status", "error", "event", "IntEnum", "EnumField",
            "specialize", "include_filename",
        ],
    },
    {
        "name": "MemAddr",
        "kind": "field_class",
        "description": (
            "Memory-address field.  Specialize with ``bitwidth`` (typically 32 "
            "or 64) and ``include_dir``.  Signals that a field carries a "
            "hardware memory address rather than an arbitrary integer."
        ),
        "keywords": ["address", "memory", "pointer", "addr", "MemAddr", "64-bit", "32-bit"],
    },
    {
        "name": "IntEnum",
        "kind": "python_class",
        "description": (
            "Standard Python ``enum.IntEnum``.  Define error codes, status "
            "codes, and event types as IntEnum subclasses, then pass the class "
            "to ``EnumField.specialize(enum_type=...)``."
        ),
        "keywords": ["enum", "IntEnum", "error_code", "status_code", "event_type", "Python"],
    },
    # -----------------------------------------------------------------------
    # Common design patterns
    # -----------------------------------------------------------------------
    {
        "name": "transaction_id_pattern",
        "kind": "pattern",
        "description": (
            "Reusable ``TxIdField = IntField.specialize(bitwidth=16, "
            "signed=False)`` placed as the first field of a command DataList "
            "and echoed as the first field of the matching response DataList.  "
            "Enables request-response correlation when the accelerator processes "
            "multiple commands concurrently."
        ),
        "keywords": ["transaction", "tx_id", "correlation", "command", "response", "echo"],
    },
    {
        "name": "include_filename_pattern",
        "kind": "pattern",
        "description": (
            "Setting ``include_filename = 'my_schema.h'`` on a DataList or on "
            "a specialized EnumField controls the name of the generated C header "
            "file.  Useful when the accelerator expects a specific header name."
        ),
        "keywords": ["include_filename", "header", "C", "codegen", "custom_filename"],
    },
    {
        "name": "static_array_pattern",
        "kind": "pattern",
        "description": (
            "A DataArray subclass with ``static = True`` and a concrete "
            "``max_shape`` tuple generates a C array with a fixed compile-time "
            "size instead of a length-prefixed dynamic array.  Typical use: "
            "coefficient arrays, kernel weights."
        ),
        "keywords": ["static", "array", "fixed-size", "compile-time", "DataArray", "max_shape"],
    },
    {
        "name": "nested_dataarray_pattern",
        "kind": "pattern",
        "description": (
            "Embedding a DataArray as a field inside a DataList (e.g., "
            "``coeffs: CoeffArray``) packs the array inline rather than by "
            "reference.  Combine with ``static = True`` for zero-overhead "
            "fixed-size arrays in the generated struct."
        ),
        "keywords": ["nested", "DataArray", "DataList", "inline", "embedded", "struct"],
    },
    {
        "name": "command_response_debug_pattern",
        "kind": "pattern",
        "description": (
            "Three-schema set for an accelerator interface: a *command* DataList "
            "(what the host sends), a *response* DataList (what the accelerator "
            "returns), and optionally a *debug* DataList (internal events "
            "emitted during co-simulation for testbench visibility).  Echo the "
            "``tx_id`` from command to response."
        ),
        "keywords": [
            "command", "response", "debug", "accelerator", "co-simulation",
            "testbench", "interface", "pattern",
        ],
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_components() -> dict:
    """Return the canonical pysilicon schema vocabulary glossary.

    This function is deterministic and requires no network access.

    Returns
    -------
    dict
        ``summary`` – one-line description string.
        ``components`` – list of component dicts, each with keys
        ``name``, ``kind``, ``description``, and ``keywords``.
    """
    return {
        "summary": (
            "Canonical pysilicon schema vocabulary: core classes "
            "(DataSchema, DataList, DataArray, DataField, IntField, FloatField, "
            "EnumField, MemAddr, IntEnum) and common design patterns. "
            "Use the keywords to form search queries for pysilicon_search_schema_examples."
        ),
        "components": list(_COMPONENTS),
    }
