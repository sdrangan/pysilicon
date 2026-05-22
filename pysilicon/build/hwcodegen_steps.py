"""BuildStep wrappers for HLS codegen."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from pysilicon.build.build import BuildConfig, BuildStep
from pysilicon.build.hwcodegen import extract_kernel
from pysilicon.build.hwgen import (
    _collect_hooks_with_params,
    cpp_kernel_name,
    kernel_files_to_str,
)
from pysilicon.hw.hw_component import HwComponent
from pysilicon.simulation.simulation import Simulation


@dataclass(kw_only=True)
class HlsCodegenStep(BuildStep):
    """Generate ``<component>.hpp``, ``<component>.cpp``, and one impl stub per hook.

    File-lifecycle rules:
    - ``<kernel>.hpp`` / ``<kernel>.cpp`` are always rewritten on every ``run()``.
    - ``<kernel>_<hook>_impl.{cpp,tpp}`` is written **only if absent**, so user
      (or future AI-completion) edits survive rebuilds.  A hook with no
      ``HwParam``-driven stream argument lands in ``.cpp``; a templated hook
      lands in ``.tpp`` (per the hook-templating contract).

    If a hook switches between templated and non-templated across runs, the
    step refuses to proceed: a stale ``.cpp``/``.tpp`` lingers on disk and
    must be deleted explicitly.  Auto-deletion is intentionally not done.
    """

    description: str = (
        "Generate HLS kernel files (.hpp, .cpp, impl stubs) from an HwComponent."
    )
    params: ClassVar[dict] = {}

    comp_class: type[HwComponent]
    source_artifact: str
    output_dir: str = "."

    def __post_init__(self) -> None:
        super().__post_init__()
        self._kernel_name = cpp_kernel_name(self.comp_class)
        self._hook_info = self._discover_hooks()

    def _discover_hooks(self) -> list[tuple[str, str]]:
        """Return ``[(hook_name, extension)]`` per hook on the component.

        Extension is ``"tpp"`` if the hook is templated, ``"cpp"`` otherwise.
        """
        comp = self.comp_class(name="_codegen", sim=Simulation())
        tree = extract_kernel(comp)
        result: list[tuple[str, str]] = []
        for hook, tparams in _collect_hooks_with_params(tree):
            ext = "tpp" if tparams else "cpp"
            result.append((hook.__name__, ext))  # type: ignore[attr-defined]
        return result

    @property
    def consumes(self) -> list:  # type: ignore[override]
        return [self.source_artifact]

    @property
    def produces(self) -> dict:  # type: ignore[override]
        out_dir = Path(self.output_dir)
        kn = self._kernel_name
        d: dict[str, Path] = {
            f"{kn}_hpp": out_dir / f"{kn}.hpp",
            f"{kn}_cpp": out_dir / f"{kn}.cpp",
        }
        for hook_name, ext in self._hook_info:
            d[f"{kn}_{hook_name}_impl"] = out_dir / f"{kn}_{hook_name}_impl.{ext}"
        return d

    def run(self, config: BuildConfig, **_) -> dict[str, Any]:
        comp = self.comp_class(name="_codegen", sim=Simulation())
        files = kernel_files_to_str(comp)
        # BuildConfig.__post_init__ normalises root_dir to a Path, but the type
        # annotation is broader; narrow it here.
        root_dir = Path(config.root_dir) if config.root_dir is not None else Path.cwd()
        out_root = root_dir / self.output_dir
        out_root.mkdir(parents=True, exist_ok=True)

        artifacts: dict[str, Any] = {}
        kn = self._kernel_name

        # Always overwrite .hpp and .cpp
        for ext in ("hpp", "cpp"):
            filename = f"{kn}.{ext}"
            path = out_root / filename
            path.write_text(files[filename], encoding="utf-8")
            artifacts[f"{kn}_{ext}"] = path

        # Stale-file detection: if a hook's expected extension differs from
        # what already exists on disk, refuse to proceed (do not auto-delete).
        for hook_name, ext in self._hook_info:
            stem = f"{kn}_{hook_name}_impl"
            other_ext = "cpp" if ext == "tpp" else "tpp"
            stale_path = out_root / f"{stem}.{other_ext}"
            if stale_path.exists():
                raise RuntimeError(
                    f"Stale impl file detected: {stale_path}. Hook "
                    f"'{hook_name}' is now "
                    f"{'templated' if ext == 'tpp' else 'non-templated'}; "
                    f"expected file is {stem}.{ext}. Delete the stale file "
                    f"and re-run."
                )

        # Sticky impl stubs: write only if the file doesn't already exist.
        # Once an impl file exists (whether the user filled it in or a future
        # AI-completion step wrote it), codegen never touches it again.
        for hook_name, ext in self._hook_info:
            filename = f"{kn}_{hook_name}_impl.{ext}"
            path = out_root / filename
            if not path.exists():
                path.write_text(files[filename], encoding="utf-8")
            artifacts[f"{kn}_{hook_name}_impl"] = path

        return artifacts
