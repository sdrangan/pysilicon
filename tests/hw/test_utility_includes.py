"""Tests for ``DataSchema.get_utility_includes`` and its overrides."""
from __future__ import annotations

from pysilicon.hw.dataschema import (
    DataArray,
    DataList,
    FloatField,
    IntField,
)


Float32 = FloatField.specialize(bitwidth=32, include_dir="include")
UInt16 = IntField.specialize(bitwidth=16, signed=False, include_dir="include")
Int16 = IntField.specialize(bitwidth=16, signed=True, include_dir="include")


def test_bare_int_field_returns_empty():
    assert UInt16.get_utility_includes() == []


def test_bare_float_field_returns_empty():
    assert Float32.get_utility_includes() == []


def test_data_array_of_float32_returns_float32_array_utils():
    class Float32Array(DataArray):
        element_type = Float32
        static = True
        max_shape = (4,)

    assert Float32Array.get_utility_includes() == ["include/float32_array_utils.h"]


def test_data_list_scalar_fields_only_returns_empty():
    class Header(DataList):
        elements = {
            "tx_id": {"schema": UInt16},
            "nsamp": {"schema": UInt16},
        }

    assert Header.get_utility_includes() == []


def test_data_list_with_data_array_member_propagates():
    class Float32Array(DataList):
        elements = {"v": {"schema": Float32}}

    # Wrap the array in a parent DataList so we exercise the recursion.
    class Float32A(DataArray):
        element_type = Float32
        static = True
        max_shape = (4,)

    class Container(DataList):
        elements = {
            "coeffs": {"schema": Float32A},
            "tx_id": {"schema": UInt16},
        }

    assert Container.get_utility_includes() == ["include/float32_array_utils.h"]


def test_data_list_with_user_defined_composite_array():
    """A user-defined composite (DataList) wrapped in a DataArray works uniformly."""
    class ComplexInt16(DataList):
        include_dir = "include"
        elements = {
            "re": {"schema": Int16},
            "im": {"schema": Int16},
        }

    class ComplexInt16Array(DataArray):
        element_type = ComplexInt16
        static = True
        max_shape = (4,)

    # Array of ComplexInt16: include path uses snake_case of the element name.
    assert ComplexInt16Array.get_utility_includes() == [
        "include/complex_int16_array_utils.h",
    ]


def test_dedup_happens_at_consumer_not_method():
    """``DataList.get_utility_includes`` may return duplicates across members."""
    class Float32A(DataArray):
        element_type = Float32
        static = True
        max_shape = (4,)

    class TwoArrays(DataList):
        elements = {
            "a": {"schema": Float32A},
            "b": {"schema": Float32A},
        }

    # Method-level: duplicates allowed. Caller is responsible for dedup.
    paths = TwoArrays.get_utility_includes()
    assert paths.count("include/float32_array_utils.h") == 2
