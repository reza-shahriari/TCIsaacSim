"""`scripts/stage_own_hunk.sh` stages only the caller's hunks — the shared-tree commit hazard.

Several sessions edit this repo's README, CHANGELOG and roadmap in one working tree at the same
time, so `git add README.md` commits whatever another session had in flight; it has cost real
prose here more than once. These tests drive the script the way two sessions actually collide:
both hold an edit to one file, one commits first, and each commit must carry only its author's
line while the other's stays in the worktree.
"""

from __future__ import annotations

import os
import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "stage_own_hunk.sh"
SHARED = "\n".join(f"line {i}" for i in range(1, 21)) + "\n"


def _env(session: str | None = None) -> dict[str, str]:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }
    if session is not None:
        env["STAGE_OWN_HUNK_ID"] = session
    return env


def git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True, env=_env()
    ).stdout


def script(repo: pathlib.Path, session: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args], cwd=repo, capture_output=True, text=True, env=_env(session)
    )


@pytest.fixture
def repo(tmp_path: pathlib.Path) -> pathlib.Path:
    r = tmp_path / "tree"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    (r / "shared.md").write_text(SHARED)
    git(r, "add", "shared.md")
    git(r, "commit", "-qm", "base")
    return r


def test_two_sessions_editing_one_file_keep_their_commits_separate(repo: pathlib.Path) -> None:
    shared = repo / "shared.md"

    # session A takes its baseline, then edits near the top
    assert script(repo, "a", "snapshot", "shared.md").returncode == 0
    shared.write_text(SHARED.replace("line 3", "line 3 - A"))

    # session B arrives later: its baseline legitimately contains A's in-flight line, so B's
    # diff is B's alone. Then B edits near the bottom.
    assert script(repo, "b", "snapshot", "shared.md").returncode == 0
    shared.write_text(shared.read_text().replace("line 18", "line 18 - B"))

    # B commits first
    result = script(repo, "b", "stage", "shared.md")
    assert result.returncode == 0, result.stderr
    git(repo, "commit", "-qm", "B's step")
    b_commit = git(repo, "show", "HEAD", "--", "shared.md")
    assert "+line 18 - B" in b_commit
    assert "line 3 - A" not in b_commit, "A's in-flight line leaked into B's commit"

    # A commits second, onto B's HEAD, and must still carry only its own line
    result = script(repo, "a", "stage", "shared.md")
    assert result.returncode == 0, result.stderr
    assert "staged your hunks only" in result.stdout
    git(repo, "commit", "-qm", "A's step")
    a_commit = git(repo, "show", "HEAD", "--", "shared.md")
    assert "+line 3 - A" in a_commit
    assert "line 18 - B" not in a_commit, "B's line leaked into A's commit"

    # and the file itself ends up with both
    head = git(repo, "show", "HEAD:shared.md")
    assert "line 3 - A" in head and "line 18 - B" in head


def test_an_edit_made_after_your_snapshot_and_never_committed_is_the_known_blind_spot(
    repo: pathlib.Path,
) -> None:
    """The one case the script cannot tell from your own work, asserted so it stays documented.

    The snapshot is the only record of what the file looked like before you touched it. If
    another session edits after your snapshot and has not committed, its line is inside your
    diff and there is nothing to distinguish it. The remedy is procedural -- snapshot late,
    stage promptly -- and `stage` prints the hunk headers so it is visible when it happens.
    """
    shared = repo / "shared.md"
    script(repo, "a", "snapshot", "shared.md")
    shared.write_text(SHARED.replace("line 3", "line 3 - A").replace("line 18", "line 18 - B"))
    result = script(repo, "a", "stage", "shared.md")
    assert result.returncode == 0
    assert "@@" in result.stdout, "the staged hunks must be printed for exactly this reason"
    git(repo, "commit", "-qm", "A's step")
    assert "line 18 - B" in git(repo, "show", "HEAD", "--", "shared.md")


def test_a_change_the_other_session_committed_after_your_snapshot_is_not_re_applied(
    repo: pathlib.Path,
) -> None:
    """The common case, and why this is a three-way merge rather than `git apply`: B's line is on
    both sides of the merge, so it is a no-op in A's commit instead of a failed patch."""
    shared = repo / "shared.md"
    script(repo, "a", "snapshot", "shared.md")
    shared.write_text(SHARED.replace("line 3", "line 3 - A"))
    # B snapshots after A's edit, edits elsewhere, and commits -- A's snapshot never saw B's line
    script(repo, "b", "snapshot", "shared.md")
    shared.write_text(shared.read_text().replace("line 18", "line 18 - B"))
    assert script(repo, "b", "stage", "shared.md").returncode == 0
    git(repo, "commit", "-qm", "B's step")

    result = script(repo, "a", "stage", "shared.md")
    assert result.returncode == 0, result.stderr
    git(repo, "commit", "-qm", "A's step")
    a_commit = git(repo, "show", "HEAD", "--", "shared.md")
    assert "+line 3 - A" in a_commit
    assert a_commit.count("line 18 - B") == 0, "B's committed line must not be re-applied"


def test_a_file_absent_from_head_is_refused_with_advice(repo: pathlib.Path) -> None:
    (repo / "new.md").write_text("all mine\n")
    assert script(repo, "a", "snapshot", "new.md").returncode == 0
    result = script(repo, "a", "stage", "new.md")
    assert result.returncode != 0 and "use: git add new.md" in result.stderr


def test_staging_without_a_snapshot_is_refused(repo: pathlib.Path) -> None:
    result = script(repo, "a", "stage", "shared.md")
    assert result.returncode != 0 and "no snapshot" in result.stderr


def test_an_unchanged_file_stages_nothing(repo: pathlib.Path) -> None:
    script(repo, "a", "snapshot", "shared.md")
    result = script(repo, "a", "stage", "shared.md")
    assert result.returncode == 0 and "nothing staged" in result.stdout
    assert git(repo, "diff", "--cached", "--stat") == ""


def test_a_collision_on_the_same_lines_is_reported_not_forced(repo: pathlib.Path) -> None:
    """When the other session changed the same lines the merge conflicts. The script must say so
    and stage nothing, rather than silently picking a side."""
    shared = repo / "shared.md"
    script(repo, "a", "snapshot", "shared.md")
    shared.write_text(SHARED.replace("line 10", "line 10 - B, committed"))
    git(repo, "commit", "-qam", "B, same line")
    shared.write_text(SHARED.replace("line 10", "line 10 - A"))

    result = script(repo, "a", "stage", "shared.md")
    assert result.returncode == 1
    assert "conflict" in result.stderr and "same" in result.stderr
    assert git(repo, "diff", "--cached", "--stat") == "", "nothing half-staged"


def test_sessions_do_not_share_a_snapshot_slot(repo: pathlib.Path) -> None:
    script(repo, "a", "snapshot", "shared.md")
    assert "shared.md" in script(repo, "a", "status").stdout
    assert "no snapshots" in script(repo, "b", "status").stdout
