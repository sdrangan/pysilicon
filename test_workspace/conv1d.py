from waveflow.hw.dataschema import DataArray, DataList, FloatField, IntField, MemAddr


INCLUDE_DIR = "include"

TxIdField = IntField.specialize(bitwidth=16, signed=False)
LengthField = IntField.specialize(bitwidth=32, signed=False)
AddrField = MemAddr.specialize(bitwidth=64, include_dir=INCLUDE_DIR)
CoeffField = FloatField.specialize(bitwidth=32, include_dir=INCLUDE_DIR)


class Conv1DCoeffArray(DataArray):
    """Fixed-size array of 1D convolution coefficients stored inline in the command."""

    ncoeff: int = 8
    element_type = CoeffField
    static = True
    max_shape = (ncoeff,)
    include_dir = INCLUDE_DIR


class Conv1DCmd(DataList):
    """Command for a 1D convolution accelerator with inline coefficients and buffer addresses."""

    elements = {
        "tx_id": {
            "schema": TxIdField,
            "description": "Transaction ID",
        },
        "nsamp": {
            "schema": LengthField,
            "description": "Number of input samples to process",
        },
        "coeffs": {
            "schema": Conv1DCoeffArray,
            "description": "1D convolution coefficients stored inline in the command",
        },
        "input_addr": {
            "schema": AddrField,
            "description": "Base memory address of the input sample buffer",
        },
        "output_addr": {
            "schema": AddrField,
            "description": "Base memory address of the output sample buffer",
        },
    }
    include_dir = INCLUDE_DIR


SCHEMA_CLASSES = [
    Conv1DCoeffArray,
    Conv1DCmd,
]
