import subprocess
from types import SimpleNamespace

from src.tasks.codebase_adaptation import generic_runtime as generic_runtime
from src.tasks.codebase_adaptation import sanitized_runtime
from src.tasks.codebase_adaptation import task as task_module


def _run_git(repo, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _init_repo(repo) -> None:
    repo.mkdir()
    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.name", "Test User")
    _run_git(repo, "config", "user.email", "test@example.invalid")


def _local_exec(_container_id, command, cwd, **_kwargs):
    return subprocess.run(
        ["bash", "-lc", command],
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _patch_local_execution(monkeypatch) -> None:
    monkeypatch.setattr(generic_runtime, "_docker_exec", _local_exec)
    monkeypatch.setattr(sanitized_runtime, "_docker_exec", _local_exec)


def test_sanitized_snapshot_removes_future_history_and_ignored_files(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    _init_repo(repo)

    (repo / ".gitignore").write_text("build-metadata/\ntracked-generated.py\n")
    (repo / "module.py").write_text("value = 'base'\n")
    tracked_ignored = repo / "tracked-generated.py"
    tracked_ignored.write_text("generated at base\n")
    _run_git(repo, "add", ".")
    _run_git(repo, "add", "-f", tracked_ignored.name)
    _run_git(repo, "commit", "-q", "-m", "base")
    base_commit = _run_git(repo, "rev-parse", "HEAD").stdout.strip()

    (repo / "module.py").write_text("value = 'future solution'\n")
    _run_git(repo, "commit", "-q", "-am", "future solution")
    future_commit = _run_git(repo, "rev-parse", "HEAD").stdout.strip()
    _run_git(repo, "checkout", "-q", base_commit)

    ignored = repo / "build-metadata"
    ignored.mkdir()
    (ignored / "SOURCES.txt").write_text("future_only.py\n")

    _patch_local_execution(monkeypatch)

    sanitized_runtime.initialize_sanitized_actor_container(
        "container",
        {"base_commit": base_commit, "workdir": str(repo)},
    )

    assert _run_git(repo, "rev-list", "--all", "--count").stdout.strip() == "1"
    assert (
        subprocess.run(
            ["git", "cat-file", "-e", f"{future_commit}^{{commit}}"],
            cwd=repo,
            check=False,
        ).returncode
        != 0
    )
    assert not ignored.exists()
    assert _run_git(repo, "remote").stdout == ""
    assert _run_git(repo, "status", "--short", "--ignored").stdout == ""
    assert (repo / "module.py").read_text() == "value = 'base'\n"
    assert tracked_ignored.read_text() == "generated at base\n"
    _run_git(repo, "ls-files", "--error-unmatch", tracked_ignored.name)

    (repo / "module.py").write_text("value = 'actor edit'\n")
    assert "actor edit" in _run_git(repo, "diff").stdout


def test_task_defaults_to_sanitized_initializer(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        task_module,
        "initialize_sanitized_actor_container",
        lambda *_args, **_kwargs: calls.append("sanitized"),
    )
    monkeypatch.setattr(
        task_module,
        "initialize_generic_pr_base_tree",
        lambda *_args, **_kwargs: calls.append("base"),
    )

    task = task_module.CodebaseAdaptationTask()
    task._env = SimpleNamespace(container_id="container")
    instance = SimpleNamespace(
        instance_id="example__repo-1",
        raw_data={"base_commit": "abc123"},
    )

    task._initialize_generic_pr_workspace(instance)

    assert calls == ["sanitized"]
