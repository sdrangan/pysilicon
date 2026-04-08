from __future__ import annotations

from pathlib import Path
import shutil

from pysilicon.build.build import CodeGenConfig


def copy_streamutils(cfg: CodeGenConfig) -> tuple[str, str, str]:
    """Copy ``streamutils_hls.h``, ``streamutils_tb.h``, and ``streamutils.cpp`` into the configured utility directory.

    Parameters
    ----------
    cfg : CodeGenConfig
        Code-generation configuration describing the output root and utility
        directory. The files are written to
        ``cfg.root_dir / cfg.util_dir / "streamutils_..."``.

    Returns
    -------
    tuple[str, str, str]
        Absolute paths to the copied files.
    """
    src_path_hls = Path(__file__).resolve().with_name("streamutils_hls.h")
    src_path_tb = Path(__file__).resolve().with_name("streamutils_tb.h")
    src_path_cpp = Path(__file__).resolve().with_name("streamutils.cpp")
    if not src_path_hls.exists():
        raise FileNotFoundError(f"Could not find source header: {src_path_hls}")
    if not src_path_tb.exists():
        raise FileNotFoundError(f"Could not find source header: {src_path_tb}")
    if not src_path_cpp.exists():
        raise FileNotFoundError(f"Could not find source file: {src_path_cpp}")

    out_dir = cfg.root_dir / cfg.util_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file_hls = out_dir / "streamutils_hls.h"
    out_file_tb = out_dir / "streamutils_tb.h"
    out_file_cpp = out_dir / "streamutils.cpp"
    shutil.copy2(src_path_hls, out_file_hls)
    shutil.copy2(src_path_tb, out_file_tb)
    shutil.copy2(src_path_cpp, out_file_cpp)
    return (
        str(out_file_hls.resolve()),
        str(out_file_tb.resolve()),
        str(out_file_cpp.resolve()),
    )

