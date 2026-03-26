"""Auto-generated PySilicon dataschema module."""

from pysilicon.hw import DataArray, DataList, FloatField, IntField


QArray = DataArray.specialize(
    element_type=FloatField.specialize(bitwidth=32),
    max_shape=(1024,),
    static=True,
    member_name="q_elem",
    cpp_repr="QArray",
    include_filename="q_array.h",
)


class CginvInput(DataList):
    elements = {
        "n": IntField.specialize(bitwidth=8, signed=False),
        "nit": IntField.specialize(bitwidth=16, signed=False),
        "Q": QArray,
    }
