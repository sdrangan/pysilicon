"""Auto-generated PySilicon dataschema module."""

from pysilicon.hw.dataschema import DataArray, DataList, EnumField, FloatField, IntField


class QArray(DataArray):
    def __init__(self, name=None):
        super().__init__(
            name=name,
            element_type=FloatField(name='q_elem', bitwidth=32),
            max_shape=(1024,),
            static=True,
        )


class CginvInput(DataList):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.add_elem(IntField(name='n', bitwidth=8, signed=False))
        self.add_elem(IntField(name='nit', bitwidth=16, signed=False))
        self.add_elem(QArray(name='Q'))
