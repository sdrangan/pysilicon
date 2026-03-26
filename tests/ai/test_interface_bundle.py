from __future__ import annotations

import json
from pathlib import Path

import pytest

from pysilicon.ai.interface_bundle import (
    generate_interface_bundle,
    interface_bundle_from_callable_symbol,
    read_interface_manifest,
    validate_generated_schema_with_vitis,
)
from pysilicon.ai.mcp_server import bundle_from_callable_symbol
from pysilicon.ai.type_inference import infer_schema_spec_from_symbol, load_python_symbol
from pysilicon.xilinxutils import toolchain


def _write_gain_shift_module(module_path: Path) -> None:
    module_path.write_text(
        "\n".join(
            [
                "from dataclasses import dataclass",
                "from typing import Annotated",
                "",
                "import numpy as np",
                "from pydantic import BaseModel, Field",
                "",
                "from pysilicon.ai.type_inference import ArrayHint, FloatHint, IntHint",
                "",
                "class GainShiftCommand(BaseModel):",
                "    tx_id: Annotated[int, IntHint(16, signed=False)] = Field(description='Transaction id')",
                "    gain: Annotated[float, FloatHint(32)] = Field(description='Multiplicative gain')",
                "    bias: Annotated[float, FloatHint(32)] = Field(description='Additive offset')",
                "    clip_min: Annotated[float, FloatHint(32)]",
                "    clip_max: Annotated[float, FloatHint(32)]",
                "",
                "@dataclass",
                "class GainShiftResponse:",
                "    tx_id: Annotated[int, IntHint(16, signed=False)]",
                "    clipped: Annotated[int, IntHint(16, signed=False)]",
                "",
                "@dataclass",
                "class GainShiftState:",
                "    peak: Annotated[float, FloatHint(32)]",
                "    produced: Annotated[int, IntHint(16, signed=False)]",
                "",
                "GainShiftStatsDType = np.dtype([('produced', np.uint16), ('clipped', np.uint16)])",
                "",
                "def gain_shift_kernel(",
                "    cmd: GainShiftCommand,",
                "    samples: Annotated[",
                "        list[Annotated[float, FloatHint(32)]],",
                "        ArrayHint(max_shape=(8,), static=False, type_name='GainShiftSamples', element_name='sample'),",
                "    ],",
                ") -> tuple[",
                "    GainShiftResponse,",
                "    Annotated[",
                "        list[Annotated[float, FloatHint(32)]],",
                "        ArrayHint(max_shape=(8,), static=False, type_name='ShiftedSamples', element_name='sample'),",
                "    ],",
                "    GainShiftState,",
                "]:",
                "    shifted = []",
                "    clipped = 0",
                "    peak = 0.0",
                "    for value in samples:",
                "        out = float(value) * float(cmd.gain) + float(cmd.bias)",
                "        if out < float(cmd.clip_min):",
                "            out = float(cmd.clip_min)",
                "            clipped += 1",
                "        elif out > float(cmd.clip_max):",
                "            out = float(cmd.clip_max)",
                "            clipped += 1",
                "        shifted.append(out)",
                "        peak = max(peak, abs(out))",
                "    return (",
                "        GainShiftResponse(tx_id=int(cmd.tx_id), clipped=clipped),",
                "        shifted,",
                "        GainShiftState(peak=peak, produced=len(shifted)),",
                "    )",
            ]
        ),
        encoding="utf-8",
    )


def test_infer_pydantic_and_numpy_dtype_symbols(tmp_path: Path):
    module_path = tmp_path / "gain_shift_symbols.py"
    _write_gain_shift_module(module_path)

    command_symbol = load_python_symbol(module_path, "GainShiftCommand")
    command_spec = infer_schema_spec_from_symbol(command_symbol)
    stats_symbol = load_python_symbol(module_path, "GainShiftStatsDType")
    stats_spec = infer_schema_spec_from_symbol(stats_symbol, root_type_name="GainShiftStats")

    assert command_spec["root"]["type_name"] == "GainShiftCommand"
    assert command_spec["root"]["fields"][0]["description"] == "Transaction id"
    assert stats_spec["root"]["type_name"] == "GainShiftStats"
    assert [field["name"] for field in stats_spec["root"]["fields"]] == ["produced", "clipped"]
    assert all(field["bitwidth"] == 16 for field in stats_spec["root"]["fields"])


def test_load_notebook_symbol_roundtrip(tmp_path: Path):
    notebook_path = tmp_path / "schema_demo.ipynb"
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "from dataclasses import dataclass\n",
                    "from typing import Annotated\n",
                    "from pysilicon.ai.type_inference import FloatHint, IntHint\n",
                    "\n",
                    "@dataclass\n",
                    "class NotebookPacket:\n",
                    "    count: Annotated[int, IntHint(8, signed=False)]\n",
                    "    gain: Annotated[float, FloatHint(32)]\n",
                ],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")

    packet_symbol = load_python_symbol(notebook_path, "NotebookPacket")
    spec = infer_schema_spec_from_symbol(packet_symbol)

    assert spec["root"]["type_name"] == "NotebookPacket"
    assert [field["name"] for field in spec["root"]["fields"]] == ["count", "gain"]


def test_callable_bundle_generation_and_manifest(tmp_path: Path):
    module_path = tmp_path / "gain_shift_symbols.py"
    _write_gain_shift_module(module_path)

    sample_inputs = {
        "cmd": {
            "tx_id": 7,
            "gain": 1.5,
            "bias": -0.25,
            "clip_min": -1.0,
            "clip_max": 1.0,
        },
        "samples": [-1.0, -0.5, 0.25, 1.0, 0.0, 0.5, -0.75, 0.9],
    }

    bundle = interface_bundle_from_callable_symbol(
        module_path,
        "gain_shift_kernel",
        interface_name="GainShiftDemo",
        description="Gain-and-clip demo interface inferred from a callable.",
        assumptions=["Samples use float32 and the stream is bounded to 8 elements."],
        sample_inputs=sample_inputs,
        evaluate_outputs=True,
    )
    result = generate_interface_bundle(bundle, tmp_path / "generated")
    manifest = read_interface_manifest(result["manifest_paths"][0])

    assert result["ok"] is True
    assert [member["name"] for member in bundle["members"]] == [
        "cmd",
        "samples",
        "gain_shift_response",
        "shifted_samples",
        "gain_shift_state",
    ]
    assert (tmp_path / "generated" / "INTERFACE_REPORT.md").exists()
    assert manifest["interface_name"] == "GainShiftDemo"
    assert len(manifest["members"]) == 5
    assert all(member["python_validation"]["ok"] for member in manifest["members"])
    assert (tmp_path / "generated" / "include" / "gain_shift_command.h").exists()
    assert (tmp_path / "generated" / "include" / "gain_shift_response.h").exists()
    assert (tmp_path / "generated" / "include" / "shifted_samples.h").exists()
    assert (tmp_path / "generated" / "vectors" / "cmd_payload.json").exists()
    assert (tmp_path / "generated" / "vectors" / "shifted_samples_words_32.txt").exists()


def test_mcp_bundle_from_callable_symbol_wrapper(tmp_path: Path):
    module_path = tmp_path / "gain_shift_symbols.py"
    _write_gain_shift_module(module_path)

    tool_result = bundle_from_callable_symbol(
        str(module_path),
        "gain_shift_kernel",
        interface_name="GainShiftDemo",
        sample_inputs={
            "cmd": {
                "tx_id": 3,
                "gain": 1.0,
                "bias": 0.0,
                "clip_min": -1.0,
                "clip_max": 1.0,
            },
            "samples": [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        evaluate_outputs=True,
    )

    assert tool_result["ok"] is True
    assert tool_result["interface_name"] == "GainShiftDemo"
    assert [member["name"] for member in tool_result["members"]] == [
        "cmd",
        "samples",
        "gain_shift_response",
        "shifted_samples",
        "gain_shift_state",
    ]


def test_validate_generated_schema_with_vitis_skips_without_toolchain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module_path = tmp_path / "gain_shift_symbols.py"
    _write_gain_shift_module(module_path)
    bundle = interface_bundle_from_callable_symbol(
        module_path,
        "gain_shift_kernel",
        interface_name="GainShiftDemo",
        sample_inputs={
            "cmd": {
                "tx_id": 5,
                "gain": 1.25,
                "bias": 0.0,
                "clip_min": -1.0,
                "clip_max": 1.0,
            },
            "samples": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        evaluate_outputs=True,
    )
    cmd_member = next(member for member in bundle["members"] if member["name"] == "cmd")

    monkeypatch.setattr(toolchain, "find_vitis_path", lambda *args, **kwargs: None)
    result = validate_generated_schema_with_vitis(
        cmd_member["spec"],
        payload=cmd_member["sample_payload"],
        work_dir=tmp_path / "vitis_cmd",
        word_bw=32,
    )

    assert result["skipped"] is True
    assert result["reason"] == "Vitis installation not found."
    assert (tmp_path / "vitis_cmd" / "serialize_test.cpp").exists()
