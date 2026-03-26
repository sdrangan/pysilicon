from __future__ import annotations

from pathlib import Path
import shutil

from pysilicon.codegen.build import CodeGenConfig


def copy_streamutils(cfg: CodeGenConfig) -> str:
    """Copy ``streamutils.h`` into the configured utility directory.

    Parameters
    ----------
    cfg : CodeGenConfig
        Code-generation configuration describing the output root and utility
        directory. The header is written to
        ``cfg.root_dir / cfg.util_dir / "streamutils.h"``.

    Returns
    -------
    str
        Absolute path to the copied file.
    """
    src_path = Path(__file__).resolve().with_name("streamutils.h")
    if not src_path.exists():
        raise FileNotFoundError(f"Could not find source header: {src_path}")

    out_dir = cfg.root_dir / cfg.util_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "streamutils.h"
    shutil.copy2(src_path, out_file)
    return str(out_file.resolve())

