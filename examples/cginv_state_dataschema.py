"""Auto-generated PySilicon dataschema module."""

from pysilicon.hw import DataArray, DataList, FloatField, IntField


RArray = DataArray.specialize(
    element_type=FloatField.specialize(bitwidth=32),
    max_shape=(1024,),
    static=True,
    member_name="r_elem",
    cpp_repr="RArray",
    include_filename="r_array.h",
)


PArray = DataArray.specialize(
    element_type=FloatField.specialize(bitwidth=32),
    max_shape=(1024,),
    static=True,
    member_name="p_elem",
    cpp_repr="PArray",
    include_filename="p_array.h",
)


XArray = DataArray.specialize(
    element_type=FloatField.specialize(bitwidth=32),
    max_shape=(1024,),
    static=True,
    member_name="x_elem",
    cpp_repr="XArray",
    include_filename="x_array.h",
)


RnormArray = DataArray.specialize(
    element_type=FloatField.specialize(bitwidth=32),
    max_shape=(32,),
    static=True,
    member_name="rnorm_elem",
    cpp_repr="RnormArray",
    include_filename="rnorm_array.h",
)


class CginvState(DataList):
    elements = {
        "n": IntField.specialize(bitwidth=8, signed=False),
        "iteration": IntField.specialize(bitwidth=16, signed=False),
        "R": RArray,
        "P": PArray,
        "X": XArray,
        "rnorm": RnormArray,
    }
