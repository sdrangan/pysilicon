"""Agent-facing source example for AI-assisted interface generation.

This file is intentionally the *input* to the AI workflow, not the driver.
An LLM agent should inspect this callable and the sample payloads below, then
use the `pysilicon-dataschema-authoring` skill plus the `pysilicon_schema`
MCP server to generate the interface bundle artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy as np
from pydantic import BaseModel, Field

from pysilicon.ai.type_inference import ArrayHint, FloatHint, IntHint


class GainShiftCommand(BaseModel):
    tx_id: Annotated[int, IntHint(16, signed=False)] = Field(description="Transaction id")
    gain: Annotated[float, FloatHint(32)] = Field(description="Multiplicative gain")
    bias: Annotated[float, FloatHint(32)] = Field(description="Additive offset")
    clip_min: Annotated[float, FloatHint(32)] = Field(description="Lower output clamp")
    clip_max: Annotated[float, FloatHint(32)] = Field(description="Upper output clamp")


@dataclass
class GainShiftResponse:
    tx_id: Annotated[int, IntHint(16, signed=False)]
    clipped: Annotated[int, IntHint(16, signed=False)]


@dataclass
class GainShiftState:
    peak: Annotated[float, FloatHint(32)]
    produced: Annotated[int, IntHint(16, signed=False)]


GainShiftStatsDType = np.dtype([("produced", np.uint16), ("clipped", np.uint16)])


def gain_shift_kernel(
    cmd: GainShiftCommand,
    samples: Annotated[
        list[Annotated[float, FloatHint(32)]],
        ArrayHint(max_shape=(8,), static=False, type_name="GainShiftSamples", element_name="sample"),
    ],
) -> tuple[
    GainShiftResponse,
    Annotated[
        list[Annotated[float, FloatHint(32)]],
        ArrayHint(max_shape=(8,), static=False, type_name="ShiftedSamples", element_name="sample"),
    ],
    GainShiftState,
]:
    shifted: list[float] = []
    clipped = 0
    peak = 0.0

    for value in samples:
        out = float(value) * float(cmd.gain) + float(cmd.bias)
        if out < float(cmd.clip_min):
            out = float(cmd.clip_min)
            clipped += 1
        elif out > float(cmd.clip_max):
            out = float(cmd.clip_max)
            clipped += 1
        shifted.append(out)
        peak = max(peak, abs(out))

    return (
        GainShiftResponse(tx_id=int(cmd.tx_id), clipped=clipped),
        shifted,
        GainShiftState(peak=peak, produced=len(shifted)),
    )


DEMO_SAMPLE_INPUTS = {
    "cmd": {
        "tx_id": 7,
        "gain": 1.5,
        "bias": -0.25,
        "clip_min": -1.0,
        "clip_max": 1.0,
    },
    "samples": [-1.0, -0.5, 0.25, 1.0, 0.0, 0.5, -0.75, 0.9],
}


DEMO_EXPECTED_INTERFACE = {
    "interface_name": "GainShiftDemo",
    "members": [
        {"name": "cmd", "direction": "input", "role": "config"},
        {"name": "samples", "direction": "input", "role": "stream"},
        {"name": "gain_shift_response", "direction": "output", "role": "response"},
        {"name": "shifted_samples", "direction": "output", "role": "stream"},
        {"name": "gain_shift_state", "direction": "output", "role": "state"},
    ],
}
