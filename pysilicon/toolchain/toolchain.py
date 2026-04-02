import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


def _is_vitis_binary(path: Path, binary_name: str) -> bool:
    if not path.is_file():
        return False
    if path.name != binary_name:
        return False
    # On POSIX we also require executable permission.
    if platform.system() == "Windows":
        return True
    return os.access(path, os.X_OK)


def _version_key(version_text: str) -> Tuple:
    """Sort key that prefers numerically newer version strings."""
    parts = re.split(r"(\d+)", version_text)
    key: List[Tuple[int, Union[int, str]]] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append((1, int(part)))
        else:
            key.append((0, part.lower()))
    return tuple(key)


def _collect_candidates(root: Path, binary_name: str) -> List[Path]:
    """
    Collect candidate vitis-run executables from a root path.

    Supported layouts:
    - <root>/<binary_name>
    - <root>/bin/<binary_name>
    - <root>/<version>/bin/<binary_name>
    - <root>/<version>/Vitis/bin/<binary_name>
    """
    candidates: List[Path] = []

    direct = root / binary_name
    if _is_vitis_binary(direct, binary_name):
        candidates.append(direct)

    bin_direct = root / "bin" / binary_name
    if _is_vitis_binary(bin_direct, binary_name):
        candidates.append(bin_direct)

    if root.is_dir():
        for child in root.iterdir():
            if not child.is_dir():
                continue
            p = child / "bin" / binary_name
            if _is_vitis_binary(p, binary_name):
                candidates.append(p)

            # Windows default installation style: C:\Xilinx\<version>\Vitis\bin\vitis-run.bat
            p_nested = child / "Vitis" / "bin" / binary_name
            if _is_vitis_binary(p_nested, binary_name):
                candidates.append(p_nested)

    return candidates


def _pick_highest_version(candidates: List[Path]) -> Optional[str]:
    if not candidates:
        return None

    def candidate_key(path: Path) -> Tuple:
        version = ""
        if path.parent.name == "bin":
            # Layout: <root>/<version>/bin/<binary>
            version = path.parent.parent.name
            # Layout: <root>/<version>/Vitis/bin/<binary>
            if path.parent.parent.name == "Vitis":
                version = path.parent.parent.parent.name
        return _version_key(version)

    best = max(candidates, key=candidate_key)
    return str(best.resolve())


def find_vitis_path(top_dir: Optional[Union[str, Path]] = None) -> Optional[str]:
    """
    Locates the Vitis execution entry point for hardware synthesis and simulation.

    On Windows, the function searches for the `vitis-run.bat` batch file, which
    initializes the necessary environment variables. On Linux, it searches for
    the `vitis-run` shell script.

    The search priority is as follows:
    1.  **Environment Variable**: Uses the path specified in `PYSILICON_VITIS_PATH`
         if it exists.
    2.  **Explicit Search**: Searches within the provided `top_dir`.
    3.  **Heuristic Search**: Searches standard OS installation paths:
        - Windows: `<top_dir>\\<version>\\Vitis\\bin\\vitis-run.bat`
        - Linux: `<top_dir>/<version>/bin/vitis-run`

    If multiple versions of Vitis are found in the search path, the version with
    the highest alphanumeric value (e.g., 2025.1 over 2024.2) is returned.

    Parameters
    ----------
    top_dir : Optional[Union[str, Path]]
        The directory to begin the search. On Windows, if None, it defaults to
        `C:\\Xilinx`. On Linux, it defaults to `/tools/Xilinx/Vitis`
        (with a fallback to `/opt/Xilinx/Vitis`).

    Returns
    -------
    Optional[str]
        The absolute path to the `vitis-run.bat` (Windows) or `vitis-run` (Linux)
        binary. Returns `None` if no valid installation is detected.
    """
    system_name = platform.system()
    binary_name = "vitis-run.bat" if system_name == "Windows" else "vitis-run"

    env_value = os.environ.get("PYSILICON_VITIS_PATH", "").strip()
    if env_value:
        env_path = Path(env_value).expanduser()
        if _is_vitis_binary(env_path, binary_name):
            return str(env_path.resolve())

        env_candidates = _collect_candidates(env_path, binary_name)
        env_match = _pick_highest_version(env_candidates)
        if env_match is not None:
            return env_match

    roots: List[Path] = []
    if top_dir is not None:
        roots.append(Path(top_dir).expanduser())

    if system_name == "Windows":
        roots.append(Path(r"C:\Xilinx"))
    else:
        roots.append(Path("/tools/Xilinx/Vitis"))
        roots.append(Path("/opt/Xilinx/Vitis"))

    all_candidates: List[Path] = []
    for root in roots:
        all_candidates.extend(_collect_candidates(root, binary_name))

    return _pick_highest_version(all_candidates)


def run_vitis_hls(
    tcl_script: Union[str, Path],
    work_dir: Optional[Union[str, Path]] = None,
    args: Optional[List[str]] = None,
    capture_output: bool = True,
) -> subprocess.CompletedProcess:
    """
    Execute a Vitis HLS TCL script using the discovered Vitis launcher.

    The function first resolves the Vitis executable via :func:`find_vitis_path`.
    It then builds and executes a command of the form:

    - Windows: ``vitis-run.bat --mode hls <tcl_script> [--tclargs ...]``
      executed via ``cmd.exe /c call ...``
    - Linux: ``vitis-run --mode hls <tcl_script> [--tclargs ...]``

    Parameters
    ----------
    tcl_script : Union[str, Path]
        Path to the TCL script consumed by Vitis HLS.
        This is passed as the positional ``<input_file>`` argument to
        ``vitis-run``.
    work_dir : Optional[Union[str, Path]], optional
        Working directory for the subprocess. If ``None``, defaults to
        ``Path(tcl_script).parent``.
    args : Optional[List[str]], optional
        Optional values passed to the TCL script through ``--tclargs``.
        If provided, they are appended after ``--tclargs`` in the exact order
        given.
    capture_output : bool, optional
        If ``True`` (default), captures stdout/stderr and stores them in the
        returned :class:`subprocess.CompletedProcess`. If ``False``, output is
        inherited by the current process terminal.

    Returns
    -------
    subprocess.CompletedProcess
        The completed process object from :func:`subprocess.run`, including
        ``args``, ``returncode``, and optionally ``stdout``/``stderr``.

    Raises
    ------
    RuntimeError
        If no Vitis executable could be discovered.
    subprocess.CalledProcessError
        If Vitis returns a non-zero exit status (``check=True`` behavior).
    """
    vitis_path = find_vitis_path()
    if not vitis_path:
        raise RuntimeError("Vitis installation not found. Please set PYSILICON_VITIS_PATH.")

    cmd_list = [vitis_path, "--mode", "hls", "--tcl", str(tcl_script)]
    if args:
        cmd_list.append("--tclargs")
        cmd_list.extend(args)

    is_windows = platform.system() == "Windows"
    if is_windows:
        joined_cmd = " ".join(f'"{c}"' for c in cmd_list)
        final_cmd = f'cmd.exe /c "call {joined_cmd}"'
    else:
        final_cmd = cmd_list

    return subprocess.run(
        final_cmd,
        cwd=work_dir or Path(tcl_script).parent,
        shell=is_windows,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def run_vitis_hls_result(
    tcl_script: Union[str, Path],
    work_dir: Optional[Union[str, Path]] = None,
    args: Optional[List[str]] = None,
    capture_output: bool = True,
) -> Dict[str, Optional[str]]:
    """
    Execute a Vitis HLS TCL script and return a structured result dictionary.

    This wrapper preserves the existing behavior of :func:`run_vitis_hls` for
    command construction and subprocess execution, but it normalizes success and
    common failure modes into a plain dictionary for callers that prefer not to
    handle exceptions directly.

    Returns
    -------
    Dict[str, Optional[str]]
        A dictionary with the fields:

        - ``status``: One of ``"passed"``, ``"subprocess_error"``, or
          ``"runtime_error"``.
        - ``stdout``: Captured standard output when available.
        - ``stderr``: Captured standard error when available.
        - ``message``: Error message for non-subprocess failures, otherwise
          ``None``.
    """
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
