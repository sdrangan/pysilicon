# Schema Spec

The constrained schema spec is a JSON-like object with:

```json
{
  "module_name": "demo_packet",
  "word_bw_supported": [32, 64],
  "root": {
    "kind": "struct",
    "type_name": "DemoPacket",
    "fields": [
      {
        "kind": "int",
        "name": "count",
        "bitwidth": 16,
        "signed": true
      },
      {
        "kind": "float",
        "name": "gain",
        "bitwidth": 32
      },
      {
        "kind": "enum",
        "name": "mode",
        "type_name": "Mode",
        "values": [
          {"name": "OFF", "value": 0},
          {"name": "ON", "value": 1}
        ]
      }
    ]
  }
}
```

## Arrays

Arrays are always named wrapper types:

```json
{
  "kind": "array",
  "name": "history",
  "type_name": "HistoryArray",
  "element_name": "sample",
  "max_shape": [16],
  "static": true,
  "element": {
    "kind": "int",
    "bitwidth": 13,
    "signed": true
  }
}
```

## Python-Type Inference Hints

The local type inference layer supports dataclasses and `TypedDict` classes.

Use `typing.Annotated` metadata from `pysilicon.ai.type_inference`:

```python
from dataclasses import dataclass
from enum import IntEnum
from typing import Annotated

from pysilicon.ai.type_inference import ArrayHint, FloatHint, IntHint


class Mode(IntEnum):
    OFF = 0
    ON = 1


@dataclass
class Packet:
    count: Annotated[int, IntHint(16, signed=True)]
    gain: Annotated[float, FloatHint(32)]
    samples: Annotated[
        list[Annotated[int, IntHint(13, signed=True)]],
        ArrayHint(max_shape=(16,), element_name="sample"),
    ]
    mode: Mode
```
