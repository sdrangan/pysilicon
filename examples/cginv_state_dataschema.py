"""Auto-generated PySilicon dataschema module."""

from pysilicon.hw.dataschema import DataArray, DataList, EnumField, FloatField, IntField


class RArray(DataArray):
    def __init__(self, name=None):
        super().__init__(
            name=name,
            element_type=FloatField(name='r_elem', bitwidth=32),
            max_shape=(1024,),
            static=True,
        )


class PArray(DataArray):
    def __init__(self, name=None):
        super().__init__(
            name=name,
            element_type=FloatField(name='p_elem', bitwidth=32),
            max_shape=(1024,),
            static=True,
        )


class XArray(DataArray):
    def __init__(self, name=None):
        super().__init__(
            name=name,
            element_type=FloatField(name='x_elem', bitwidth=32),
            max_shape=(1024,),
            static=True,
        )


class RnormArray(DataArray):
    def __init__(self, name=None):
        super().__init__(
            name=name,
            element_type=FloatField(name='rnorm_elem', bitwidth=32),
            max_shape=(32,),
            static=True,
        )


class CginvState(DataList):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.add_elem(IntField(name='n', bitwidth=8, signed=False))
        self.add_elem(IntField(name='iteration', bitwidth=16, signed=False))
        self.add_elem(RArray(name='R'))
        self.add_elem(PArray(name='P'))
        self.add_elem(XArray(name='X'))
        self.add_elem(RnormArray(name='rnorm'))
