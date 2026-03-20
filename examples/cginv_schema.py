"""
Dataclass schema for the cginv (conjugate gradient matrix inverse) function.

cginv computes the matrix inverse for an nxn positive semi-definite matrix.
"""
from dataclasses import dataclass
from enum import IntEnum
from typing import Annotated

from pysilicon.ai.type_inference import ArrayHint, FloatHint, IntHint


# Maximum matrix dimension (32x32 = 1024 elements)
MAX_N = 32
MAX_MATRIX_ELEMENTS = MAX_N * MAX_N


@dataclass
class CginvInput:
    """Input parameters for the cginv algorithm."""
    n: Annotated[int, IntHint(8, signed=False)]  # matrix size (unsigned)
    nit: Annotated[int, IntHint(16, signed=False)]  # number of iterations
    Q: Annotated[
        list[Annotated[float, FloatHint(32)]],
        ArrayHint(max_shape=(MAX_MATRIX_ELEMENTS,), element_name="q_elem"),
    ]


@dataclass
class CginvOutput:
    """Output of the cginv algorithm."""
    n: Annotated[int, IntHint(8, signed=False)]  # matrix size
    X: Annotated[
        list[Annotated[float, FloatHint(32)]],
        ArrayHint(max_shape=(MAX_MATRIX_ELEMENTS,), element_name="x_elem"),
    ]


@dataclass
class CginvState:
    """Internal state during cginv computation."""
    n: Annotated[int, IntHint(8, signed=False)]
    iteration: Annotated[int, IntHint(16, signed=False)]
    R: Annotated[
        list[Annotated[float, FloatHint(32)]],
        ArrayHint(max_shape=(MAX_MATRIX_ELEMENTS,), element_name="r_elem"),
    ]
    P: Annotated[
        list[Annotated[float, FloatHint(32)]],
        ArrayHint(max_shape=(MAX_MATRIX_ELEMENTS,), element_name="p_elem"),
    ]
    X: Annotated[
        list[Annotated[float, FloatHint(32)]],
        ArrayHint(max_shape=(MAX_MATRIX_ELEMENTS,), element_name="x_elem"),
    ]
    rnorm: Annotated[
        list[Annotated[float, FloatHint(32)]],
        ArrayHint(max_shape=(MAX_N,), element_name="rnorm_elem"),
    ]
