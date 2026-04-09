from pathlib import Path

class CodeGenConfig(object):
    """
    Defines the configuration for the Vitis HLS code generation
    """
    def __init__(
            self, 
            root_dir: str | Path | None = None,
            util_dir: str | Path | None = None,
            vitis_version: str | None = None) -> None:
        """
        Parameters
        ----------

        root_dir : str | pathlib.Path | None
            The root directory of the project, where the generated code will be placed.
            If None, defaults to the current working directory.
        util_dir : str | pathlib.Path | None
            The directory where utility code (e.g. common headers, helper functions) will be placed.
            This is relative to the root_dir. If None, defaults to ".", meaning the same directory as root directory 
            of the generated code.
        vitis_version : str | None
            The Vitis HLS version string in ``"YYYY.M"`` format (e.g. ``"2023.1"`` or ``"2025.2"``).
            Used to control which compatibility files are emitted. If ``None``, conservative
            (legacy-compatible) behaviour is assumed.
        """
        self.root_dir: Path = Path.cwd() if root_dir is None else Path(root_dir)
        self.util_dir: Path = Path(".") if util_dir is None else Path(util_dir)
        self.vitis_version: str | None = vitis_version

    def vitis_version_tuple(self) -> tuple[int, int] | None:
        """Parse ``vitis_version`` into a ``(major, minor)`` integer tuple.

        Returns
        -------
        tuple[int, int] | None
            Parsed version tuple, or ``None`` when ``vitis_version`` is ``None``.

        Raises
        ------
        ValueError
            When ``vitis_version`` is set but does not follow the ``"YYYY.M"`` format.
        """
        if self.vitis_version is None:
            return None
        parts = self.vitis_version.split(".")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid vitis_version '{self.vitis_version}'. Expected format 'YYYY.M'."
            )
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            raise ValueError(
                f"Invalid vitis_version '{self.vitis_version}'. Expected format 'YYYY.M'."
            )

    def needs_legacy_streamutils_cpp(self) -> bool:
        """Return ``True`` when ``streamutils.cpp`` must be included in the output.

        The file is required for Vitis versions strictly older than ``2025.1``.
        When no version is specified the conservative default is to include it.

        Returns
        -------
        bool
            ``True`` if ``streamutils.cpp`` should be copied, ``False`` otherwise.
        """
        ver = self.vitis_version_tuple()
        if ver is None:
            return True
        return ver < (2025, 1)