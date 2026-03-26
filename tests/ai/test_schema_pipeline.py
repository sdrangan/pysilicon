from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Annotated

from pysilicon.ai.mcp_server import (
    build_server,
    generate_schema_headers,
    spec_from_python_symbol,
    validate_generated_schema,
    validate_schema_spec,
)
from pysilicon.ai.type_inference import ArrayHint, FloatHint, IntHint, infer_schema_spec_from_symbol


def test_validate_struct_schema_roundtrip(tmp_path):
    spec = {
        "root": {
            "kind": "struct",
            "type_name": "DemoPacket",
            "fields": [
                {"kind": "int", "name": "count", "bitwidth": 16, "signed": True},
                {"kind": "float", "name": "gain", "bitwidth": 32},
                {
                    "kind": "enum",
                    "name": "mode",
                    "type_name": "Mode",
                    "values": [
                        {"name": "OFF", "value": 0},
                        {"name": "ON", "value": 1},
                    ],
                },
            ],
        }
    }

    normalized = validate_schema_spec(spec)
    result = validate_generated_schema(
        normalized["spec"],
        payload={"count": -7, "gain": 1.5, "mode": 1},
        word_bw=32,
    )
    headers = generate_schema_headers(normalized["spec"], include_dir=tmp_path)

    assert result["ok"] is True
    assert result["packed_words"] == [65529, 1069547520, 1]
    assert (tmp_path / "demo_packet.h").exists()
    assert headers["failed_headers"] == []


def test_validate_array_schema_roundtrip():
    spec = {
        "root": {
            "kind": "array",
            "type_name": "SampData",
            "element_name": "samp",
            "max_shape": [4],
            "element": {"kind": "int", "bitwidth": 13, "signed": True},
        }
    }

    result = validate_generated_schema(spec, payload=[-3, -2, 0, 7], word_bw=32)

    assert result["ok"] is True
    assert result["root_type_name"] == "SampData"


class Mode(IntEnum):
    OFF = 0
    ON = 1


@dataclass
class Inner:
    i: Annotated[int, IntHint(16, signed=True)]
    q: Annotated[float, FloatHint(32)]


@dataclass
class Packet:
    seq: Annotated[int, IntHint(8, signed=False)]
    sample: Inner
    hist: Annotated[
        list[Annotated[int, IntHint(13, signed=True)]],
        ArrayHint(max_shape=(4,), element_name="samp"),
    ]
    mode: Mode


def test_infer_from_dataclass_and_generate_headers(tmp_path):
    spec = infer_schema_spec_from_symbol(Packet)
    result = validate_generated_schema(
        spec,
        payload={
            "seq": 7,
            "sample": {"i": -2, "q": 1.25},
            "hist": [1, 2, 3, 4],
            "mode": 1,
        },
        word_bw=32,
    )
    headers = generate_schema_headers(spec, include_dir=tmp_path)

    assert result["ok"] is True
    assert headers["generated_types"] == ["Inner", "HistArray", "ModeField", "Packet"]
    assert (tmp_path / "inner.h").exists()
    assert (tmp_path / "hist_array.h").exists()
    assert (tmp_path / "mode.h").exists()
    assert (tmp_path / "packet.h").exists()
    assert headers["failed_headers"] == []


@dataclass
class MultiArrayPacket:
    coeffs: Annotated[
        list[Annotated[float, FloatHint(32)]],
        ArrayHint(max_shape=(4,), type_name="CoeffArray", element_name="coeff"),
    ]
    samples: Annotated[
        list[Annotated[float, FloatHint(32)]],
        ArrayHint(max_shape=(4,), type_name="SampleArray", element_name="sample"),
    ]


def test_generate_headers_preserves_array_type_names(tmp_path):
    spec = infer_schema_spec_from_symbol(MultiArrayPacket)
    headers = generate_schema_headers(spec, include_dir=tmp_path)

    assert headers["generated_types"] == ["CoeffArray", "SampleArray", "MultiArrayPacket"]
    assert (tmp_path / "coeff_array.h").exists()
    assert (tmp_path / "sample_array.h").exists()
    assert (tmp_path / "multi_array_packet.h").exists()
    assert not (tmp_path / "float32_array.h").exists()
    assert headers["failed_headers"] == []


def test_spec_from_python_symbol_file_roundtrip(tmp_path):
    module_path = tmp_path / "schema_input.py"
    module_path.write_text(
        "\n".join(
            [
                "from dataclasses import dataclass",
                "from enum import IntEnum",
                "from typing import Annotated",
                "",
                "from pysilicon.ai.type_inference import ArrayHint, FloatHint, IntHint",
                "",
                "class Mode(IntEnum):",
                "    OFF = 0",
                "    ON = 1",
                "",
                "@dataclass",
                "class Packet:",
                "    count: Annotated[int, IntHint(16, signed=True)]",
                "    gain: Annotated[float, FloatHint(32)]",
                "    hist: Annotated[list[Annotated[int, IntHint(13, signed=True)]], ArrayHint(max_shape=(4,), element_name='samp')]",
                "    mode: Mode",
            ]
        ),
        encoding="utf-8",
    )

    tool_result = spec_from_python_symbol(str(module_path), "Packet")
    roundtrip = validate_generated_schema(
        tool_result["spec"],
        payload={"count": -3, "gain": 0.5, "hist": [1, 2, 3, 4], "mode": 1},
        word_bw=32,
    )

    assert tool_result["root_type_name"] == "Packet"
    assert roundtrip["ok"] is True


def test_build_mcp_server_smoke():
    server = build_server()
    assert server is not None
