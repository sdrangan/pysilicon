"""Code generation utilities for waveflow."""

from __future__ import annotations

from typing import Any

from waveflow.build.build import (
    Buildable,
    BuildConfig,
    BuildDag,
    BuildResult,
    BuildStep,
)
from waveflow.build.streamutils import MemMgrStep, StreamUtilsStep

__all__ = [
    "BuildConfig",
    "BuildDag",
    "BuildResult",
    "BuildStep",
    "Buildable",
    "MemMgrStep",
    "StreamUtilsStep",
    "gen_array_utils",
]


def __getattr__(name: str) -> Any:
    if name == "gen_array_utils":
        from waveflow.hw.arrayutils import gen_array_utils

        return gen_array_utils
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
