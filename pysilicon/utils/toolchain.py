"""Compatibility wrapper exposing the Vitis toolchain helpers under pysilicon.utils."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union

from pysilicon.toolchain import toolchain as _impl

__all__ = [
	"find_vitis_path",
	"run_vitis_hls",
	"run_vitis_hls_result",
]

platform = _impl.platform


def find_vitis_path(top_dir: Optional[Union[str, Path]] = None) -> Optional[str]:
	return _impl.find_vitis_path(top_dir=top_dir)


def run_vitis_hls(
	tcl_script: Union[str, Path],
	work_dir: Optional[Union[str, Path]] = None,
	args: Optional[List[str]] = None,
	capture_output: bool = True,
) -> subprocess.CompletedProcess:
	return _impl.run_vitis_hls(
		tcl_script=tcl_script,
		work_dir=work_dir,
		args=args,
		capture_output=capture_output,
	)


def run_vitis_hls_result(
	tcl_script: Union[str, Path],
	work_dir: Optional[Union[str, Path]] = None,
	args: Optional[List[str]] = None,
	capture_output: bool = True,
) -> Dict[str, Optional[str]]:
	try:
		result = run_vitis_hls(
			tcl_script=tcl_script,
			work_dir=work_dir,
			args=args,
			capture_output=capture_output,
		)
		return {
			"status": "passed",
			"stdout": result.stdout,
			"stderr": result.stderr,
			"message": None,
		}
	except subprocess.CalledProcessError as exc:
		return {
			"status": "subprocess_error",
			"stdout": exc.stdout,
			"stderr": exc.stderr,
			"message": str(exc),
		}
	except Exception as exc:
		return {
			"status": "runtime_error",
			"stdout": None,
			"stderr": None,
			"message": str(exc),
		}