from __future__ import annotations

from pathlib import Path
import shutil

from pysilicon.build.build import CodeGenConfig


def copy_streamutils(
    cfg: CodeGenConfig,
) -> tuple[str, str, str | None]:
    """Copy streamutils support files into the configured utility directory.

    ``streamutils_hls.h`` and ``streamutils_tb.h`` are always copied because
    generated Vitis HLS code references them unconditionally.

    ``streamutils.cpp`` is only copied when the configured Vitis version is
    strictly older than ``2025.1`` (or when no version is specified, in which
    case the conservative default is to copy it).  If the version is ``2025.1``
    or newer and a stale ``streamutils.cpp`` already exists in the output
    directory it is removed so the output remains reproducible.

    Parameters
    ----------
    cfg : CodeGenConfig
        Code-generation configuration describing the output root and utility
        directory.  Files are written to ``cfg.root_dir / cfg.util_dir``.

    Returns
    -------
    tuple[str, str, str | None]
        A three-element tuple:

        * Absolute path to the copied ``streamutils_hls.h``.
        * Absolute path to the copied ``streamutils_tb.h``.
        * Absolute path to the copied ``streamutils.cpp``, or ``None`` when
          the file was not copied (i.e. Vitis >= 2025.1).
    """
    src_path_hls = Path(__file__).resolve().with_name("streamutils_hls.h")
    src_path_tb = Path(__file__).resolve().with_name("streamutils_tb.h")
    src_path_cpp = Path(__file__).resolve().with_name("streamutils.cpp")
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

    out_file_cpp = out_dir / "streamutils.cpp"
    if cfg.needs_legacy_streamutils_cpp():
        if not src_path_cpp.exists():
            raise FileNotFoundError(f"Could not find source file: {src_path_cpp}")
        shutil.copy2(src_path_cpp, out_file_cpp)
        out_cpp: str | None = str(out_file_cpp.resolve())
    else:
        if out_file_cpp.exists():
            out_file_cpp.unlink()
        out_cpp = None

    return str(out_file_hls.resolve()), str(out_file_tb.resolve()), out_cpp


