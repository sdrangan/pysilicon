"""Tests for version-aware streamutils copying."""
from __future__ import annotations

from pathlib import Path

import pytest

from pysilicon.build.build import CodeGenConfig
from pysilicon.build.streamutils import copy_streamutils


# ---------------------------------------------------------------------------
# CodeGenConfig.vitis_version_tuple
# ---------------------------------------------------------------------------


def test_vitis_version_tuple_none_when_not_set():
    cfg = CodeGenConfig()
    assert cfg.vitis_version_tuple() is None


def test_vitis_version_tuple_parses_correctly():
    cfg = CodeGenConfig(vitis_version="2023.1")
    assert cfg.vitis_version_tuple() == (2023, 1)


def test_vitis_version_tuple_parses_2025_1():
    cfg = CodeGenConfig(vitis_version="2025.1")
    assert cfg.vitis_version_tuple() == (2025, 1)


def test_vitis_version_tuple_invalid_format_raises():
    cfg = CodeGenConfig(vitis_version="2025")
    with pytest.raises(ValueError, match="Invalid vitis_version"):
        cfg.vitis_version_tuple()


def test_vitis_version_tuple_non_numeric_raises():
    cfg = CodeGenConfig(vitis_version="foo.bar")
    with pytest.raises(ValueError, match="Invalid vitis_version"):
        cfg.vitis_version_tuple()


# ---------------------------------------------------------------------------
# CodeGenConfig.needs_legacy_streamutils_cpp
# ---------------------------------------------------------------------------


def test_needs_legacy_defaults_true_when_no_version():
    cfg = CodeGenConfig()
    assert cfg.needs_legacy_streamutils_cpp() is True


def test_needs_legacy_true_for_2023_1():
    cfg = CodeGenConfig(vitis_version="2023.1")
    assert cfg.needs_legacy_streamutils_cpp() is True


def test_needs_legacy_true_for_2024_2():
    cfg = CodeGenConfig(vitis_version="2024.2")
    assert cfg.needs_legacy_streamutils_cpp() is True


def test_needs_legacy_false_for_2025_1():
    cfg = CodeGenConfig(vitis_version="2025.1")
    assert cfg.needs_legacy_streamutils_cpp() is False


def test_needs_legacy_false_for_2025_2():
    cfg = CodeGenConfig(vitis_version="2025.2")
    assert cfg.needs_legacy_streamutils_cpp() is False


def test_needs_legacy_false_for_2026_1():
    cfg = CodeGenConfig(vitis_version="2026.1")
    assert cfg.needs_legacy_streamutils_cpp() is False


# ---------------------------------------------------------------------------
# copy_streamutils — default (no version specified)
# ---------------------------------------------------------------------------


def test_copy_streamutils_default_copies_headers(tmp_path: Path):
    hls_path, tb_path, cpp_path = copy_streamutils(CodeGenConfig(root_dir=tmp_path))
    assert Path(hls_path).exists()
    assert Path(tb_path).exists()


def test_copy_streamutils_default_copies_cpp(tmp_path: Path):
    _, _, cpp_path = copy_streamutils(CodeGenConfig(root_dir=tmp_path))
    assert cpp_path is not None
    assert Path(cpp_path).exists()


# ---------------------------------------------------------------------------
# copy_streamutils — legacy version (2023.1)
# ---------------------------------------------------------------------------


def test_copy_streamutils_2023_1_copies_headers(tmp_path: Path):
    cfg = CodeGenConfig(root_dir=tmp_path, vitis_version="2023.1")
    hls_path, tb_path, _ = copy_streamutils(cfg)
    assert Path(hls_path).exists()
    assert Path(tb_path).exists()


def test_copy_streamutils_2023_1_copies_cpp(tmp_path: Path):
    cfg = CodeGenConfig(root_dir=tmp_path, vitis_version="2023.1")
    _, _, cpp_path = copy_streamutils(cfg)
    assert cpp_path is not None
    assert Path(cpp_path).exists()


# ---------------------------------------------------------------------------
# copy_streamutils — new version (2025.1)
# ---------------------------------------------------------------------------


def test_copy_streamutils_2025_1_copies_headers(tmp_path: Path):
    cfg = CodeGenConfig(root_dir=tmp_path, vitis_version="2025.1")
    hls_path, tb_path, _ = copy_streamutils(cfg)
    assert Path(hls_path).exists()
    assert Path(tb_path).exists()


def test_copy_streamutils_2025_1_does_not_copy_cpp(tmp_path: Path):
    cfg = CodeGenConfig(root_dir=tmp_path, vitis_version="2025.1")
    _, _, cpp_path = copy_streamutils(cfg)
    assert cpp_path is None
    assert not (tmp_path / "streamutils.cpp").exists()


def test_copy_streamutils_2025_2_does_not_copy_cpp(tmp_path: Path):
    cfg = CodeGenConfig(root_dir=tmp_path, vitis_version="2025.2")
    _, _, cpp_path = copy_streamutils(cfg)
    assert cpp_path is None
    assert not (tmp_path / "streamutils.cpp").exists()


# ---------------------------------------------------------------------------
# copy_streamutils — stale-file cleanup
# ---------------------------------------------------------------------------


def test_copy_streamutils_removes_stale_cpp_for_2025_1(tmp_path: Path):
    # Simulate a previous run with an older version that copied streamutils.cpp
    stale = tmp_path / "streamutils.cpp"
    stale.write_text("stale content", encoding="utf-8")

    cfg = CodeGenConfig(root_dir=tmp_path, vitis_version="2025.1")
    _, _, cpp_path = copy_streamutils(cfg)

    assert cpp_path is None
    assert not stale.exists()


def test_copy_streamutils_removes_stale_cpp_in_util_subdir(tmp_path: Path):
    util_dir = tmp_path / "common"
    util_dir.mkdir()
    stale = util_dir / "streamutils.cpp"
    stale.write_text("stale content", encoding="utf-8")

    cfg = CodeGenConfig(root_dir=tmp_path, util_dir="common", vitis_version="2025.1")
    _, _, cpp_path = copy_streamutils(cfg)

    assert cpp_path is None
    assert not stale.exists()


# ---------------------------------------------------------------------------
# copy_streamutils — invalid version format
# ---------------------------------------------------------------------------


def test_copy_streamutils_invalid_version_raises(tmp_path: Path):
    cfg = CodeGenConfig(root_dir=tmp_path, vitis_version="2025")
    with pytest.raises(ValueError, match="Invalid vitis_version"):
        copy_streamutils(cfg)
