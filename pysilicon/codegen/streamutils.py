from __future__ import annotations

from pathlib import Path
import shutil


def copy_streamutils(dst_path: str | Path) -> str:
	"""
	Copy ``streamutils.h`` to the specified destination path.

	Parameters
	----------
	dst_path : str | Path
		Destination directory or full destination file path.
		- If a directory is provided, the output file will be
		  ``<dst_path>/streamutils.h``.
		- If a file path is provided, the header is copied to that file path.

	Returns
	-------
	str
		Absolute path to the copied file.
	"""
	src_path = Path(__file__).resolve().with_name("streamutils.h")
	if not src_path.exists():
		raise FileNotFoundError(f"Could not find source header: {src_path}")

	dst = Path(dst_path).expanduser()
	is_dir_target = dst.exists() and dst.is_dir()
	if not is_dir_target and dst.suffix == "":
		is_dir_target = True

	out_file = dst / "streamutils.h" if is_dir_target else dst
	out_file.parent.mkdir(parents=True, exist_ok=True)
	shutil.copy2(src_path, out_file)
	return str(out_file.resolve())

