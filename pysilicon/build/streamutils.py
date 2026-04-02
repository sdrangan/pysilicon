from __future__ import annotations

from pathlib import Path
import shutil

from pysilicon.build.build import CodeGenConfig


def copy_streamutils(cfg: CodeGenConfig) -> str:
    """Copy ``streamutils_hls.h`` and ``streamutils_tb.h``into the configured utility directory.

    Parameters
    ----------
    cfg : CodeGenConfig
        Code-generation configuration describing the output root and utility
        directory. The header is written to
        ``cfg.root_dir / cfg.util_dir / "streamutils_...h"``.

    Returns
    -------
    str
        Absolute path to the copied file.
    """
    src_path_hls = Path(__file__).resolve().with_name("streamutils_hls.h")
    src_path_tb = Path(__file__).resolve().with_name("streamutils_tb.h")
    if not src_path_hls.exists():
        raise FileNotFoundError(f"Could not find source header: {src_path_hls}")
    if not src_path_tb.exists():
        raise FileNotFoundError(f"Could not find source header: {src_path_tb}")

    out_dir = cfg.root_dir / cfg.util_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file_hls = out_dir / "streamutils_hls.h"
    out_file_tb = out_dir / "streamutils_tb.h"
    shutil.copy2(src_path_hls, out_file_hls)
    shutil.copy2(src_path_tb, out_file_tb)
    return str(out_file_hls.resolve()), str(out_file_tb.resolve())

