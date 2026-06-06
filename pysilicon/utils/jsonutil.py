"""Small JSON-serialization helpers shared by build scripts that write status /
burst / timing JSON.

``json_scalar`` coerces a numpy scalar to a native Python scalar so ``json.dumps``
accepts it; ``hex_word`` formats an integer as a fixed-width, masked hex string.
The numpy-scalar coercion idiom recurs across the example build scripts and the
data-model serialization (e.g. ``dataschema._to_jsonable``, which walks nested
structures and can call ``json_scalar`` as its scalar-level primitive).
"""
from __future__ import annotations

import numpy as np


def json_scalar(value):
    """Coerce a numpy integer/floating scalar to a native Python scalar.

    Pass-through for any value that is not a numpy scalar, so it is safe to apply
    blanket-ly before ``json.dumps``.
    """
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def hex_word(value, bitwidth: int) -> str:
    """Format ``value`` as a ``0x``-prefixed, zero-padded hex string of exactly
    ``ceil(bitwidth/4)`` digits, masked to ``bitwidth`` bits.

    Raises ``ValueError`` if ``bitwidth`` is not positive.
    """
    if bitwidth <= 0:
        raise ValueError("bitwidth must be positive.")
    int_value = int(json_scalar(value))
    mask = (1 << bitwidth) - 1
    hex_width = (bitwidth + 3) // 4
    return f"0x{(int_value & mask):0{hex_width}x}"
