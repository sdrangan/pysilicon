"""Auto-generated PySilicon dataschema module."""

from pysilicon.hw import DataArray, DataList, FloatField, IntField


XArray = DataArray.specialize(
    element_type=FloatField.specialize(bitwidth=32),
    max_shape=(1024,),
    static=True,
    member_name="x_elem",
    cpp_repr="XArray",
    include_filename="x_array.h",
)


class CginvOutput(DataList):
    elements = {
        "n": IntField.specialize(bitwidth=8, signed=False),
        "X": XArray,
    }
