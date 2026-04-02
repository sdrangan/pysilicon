"""Code generation utilities for pysilicon."""

from __future__ import annotations

from typing import Any

__all__ = ["gen_array_utils"]


def __getattr__(name: str) -> Any:
	if name == "gen_array_utils":
		from pysilicon.hw.arrayutils import gen_array_utils

		return gen_array_utils
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
