from pathlib import Path

class CodeGenConfig(object):
    """
    Defines the configuration for the Vitis HLS code generation
    """
    def __init__(
            self, 
            root_dir: str | Path | None = None,
            util_dir: str | Path | None = None) -> None:
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
        """
        self.root_dir: Path = Path.cwd() if root_dir is None else Path(root_dir)
        self.util_dir: Path = Path(".") if util_dir is None else Path(util_dir)