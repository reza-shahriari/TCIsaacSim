"""Engine-availability probes and, in particular, how Warp is found (ADR 0014 addendum).

Warp ships as the `omni.warp.core` Kit extension. Kit normally puts it on `sys.path`, which is
why ADR 0014 first recorded it as reachable only from inside a running Kit; it is an ordinary
Python package in the build's extension cache, so `ensure_warp_on_path` can reach it from a bare
interpreter. These tests are engine-free: the extension cache is faked in `tmp_path`, so they say
the same thing on a laptop with no Isaac Sim as on the workstation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from irsim_isaac import env


def _fake_extension(root: Path, name: str = "omni.warp.core-1.16.0+lx64") -> Path:
    ext = root / "extscache" / name
    (ext / "warp").mkdir(parents=True)
    (ext / "warp" / "__init__.py").write_text("# not the real Warp\n")
    return ext


def test_isaac_root_prefers_the_env_var_python_sh_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ISAAC_PATH", str(tmp_path))
    assert env.isaac_root() == tmp_path
    monkeypatch.setenv("ISAAC_PATH", str(tmp_path / "does-not-exist"))
    assert env.isaac_root() is None


def test_warp_extension_path_globs_the_cache_and_ignores_entries_without_the_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ext = _fake_extension(tmp_path)
    (tmp_path / "extscache" / "omni.warp.core-0.0.1+lx64").mkdir(parents=True)  # empty: not it
    monkeypatch.delenv("IRSIM_WARP_PATH", raising=False)
    monkeypatch.setenv("ISAAC_PATH", str(tmp_path))
    assert env.warp_extension_path() == ext


def test_irsim_warp_path_overrides_and_is_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ext = _fake_extension(tmp_path)
    monkeypatch.setenv("IRSIM_WARP_PATH", str(ext))
    assert env.warp_extension_path() == ext
    monkeypatch.setenv("IRSIM_WARP_PATH", str(tmp_path))  # no warp/ package inside
    assert env.warp_extension_path() is None


def test_ensure_warp_on_path_appends_once_and_reports_the_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ext = _fake_extension(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))  # restored by monkeypatch
    monkeypatch.setenv("IRSIM_WARP_PATH", str(ext))
    monkeypatch.delenv("IRSIM_FORCE_NO_WARP", raising=False)
    monkeypatch.setattr(env, "_spec_present", lambda name: False)  # Warp not resolving yet
    assert env.ensure_warp_on_path() == ext
    assert str(ext) in sys.path
    env.ensure_warp_on_path()
    assert sys.path.count(str(ext)) == 1, "idempotent: one entry however often it is called"


def test_an_already_resolving_warp_is_never_shadowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pip or Kit copy wins; the cache directory is only a fallback (ADR 0014)."""
    ext = _fake_extension(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("IRSIM_WARP_PATH", str(ext))
    monkeypatch.setattr(env, "_spec_present", lambda name: True)
    assert env.ensure_warp_on_path() is None
    assert str(ext) not in sys.path


def test_force_no_warp_wins_over_a_findable_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ext = _fake_extension(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("IRSIM_WARP_PATH", str(ext))
    monkeypatch.setenv("IRSIM_FORCE_NO_WARP", "1")
    assert env.ensure_warp_on_path() is None
    assert env.has_warp() is False
    assert str(ext) not in sys.path


def test_force_no_isaac_makes_the_probe_answer_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRSIM_FORCE_NO_ISAAC", "1")
    assert env.has_isaac() is False


@pytest.mark.skipif(not env.has_isaac(), reason="no Isaac Sim build to look inside")
def test_this_isaac_build_really_does_carry_an_importable_warp() -> None:
    """The finding itself, on whatever build is installed: the extension is on disk and the
    probe reports Warp available without Kit ever having been started."""
    assert env.warp_extension_path() is not None, env.isaac_root()
    assert env.has_warp()
