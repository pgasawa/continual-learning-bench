import subprocess
from types import SimpleNamespace

from src.tasks.codebase_adaptation import generic_runtime as gr


def test_generic_runtime_detects_runtime_mode():
    assert gr.is_generic_pr_instance({"runtime_mode": gr.GENERIC_PR_RUNTIME})
    assert not gr.is_generic_pr_instance({})


def test_build_test_command_prefers_explicit_test_targets():
    instance = {
        "test_command": "pytest --asyncio-mode=auto",
        "test_changed_file_paths": [
            "responses/tests/test_matchers.py",
            "responses/tests/test_responses.py",
        ],
    }

    command = gr.build_test_command(instance)

    assert command == (
        "pytest --asyncio-mode=auto "
        "responses/tests/test_matchers.py responses/tests/test_responses.py"
    )


def test_test_targets_fallback_to_test_patch():
    instance = {
        "test_patch": (
            "diff --git a/tests/test_example.py b/tests/test_example.py\n"
            "--- a/tests/test_example.py\n"
            "+++ b/tests/test_example.py\n"
            "@@ -1,1 +1,2 @@\n"
            "-old\n+new\n"
        )
    }

    assert gr.test_targets(instance) == ["tests/test_example.py"]


def test_test_targets_skip_non_test_files():
    instance = {
        "test_changed_file_paths": [
            ".github/workflows/test.yml",
            "docs/guides/testing.md",
            "src/pkg/runtime.py",
            "tests/test_runtime.py",
        ]
    }

    assert gr.test_targets(instance) == ["tests/test_runtime.py"]


def test_test_targets_skip_test_fixtures_and_non_python_files():
    instance = {
        "test_changed_file_paths": [
            "tests/files/issue_524.yaml",
            "tests/test_tablib.py",
        ]
    }

    assert gr.test_file_targets(instance) == ["tests/test_tablib.py"]


def test_test_patch_paths_include_non_test_harness_files():
    instance = {
        "test_patch": (
            "diff --git a/.github/workflows/test.yml b/.github/workflows/test.yml\n"
            "--- a/.github/workflows/test.yml\n"
            "+++ b/.github/workflows/test.yml\n"
            "@@ -1,1 +1,1 @@\n"
            "-old\n+new\n"
            "diff --git a/tests/test_runtime.py b/tests/test_runtime.py\n"
            "--- a/tests/test_runtime.py\n"
            "+++ b/tests/test_runtime.py\n"
            "@@ -1,1 +1,2 @@\n"
            "-old\n+new\n"
        )
    }

    assert gr.test_patch_paths(instance) == [
        ".github/workflows/test.yml",
        "tests/test_runtime.py",
    ]


def test_strip_patch_paths_removes_test_owned_file_sections():
    patch = (
        "diff --git a/src/runtime.py b/src/runtime.py\n"
        "--- a/src/runtime.py\n"
        "+++ b/src/runtime.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n+new\n"
        "diff --git a/tests/test_runtime.py b/tests/test_runtime.py\n"
        "--- a/tests/test_runtime.py\n"
        "+++ b/tests/test_runtime.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-oldtest\n+newtest\n"
        "diff --git a/.github/workflows/test.yml b/.github/workflows/test.yml\n"
        "--- a/.github/workflows/test.yml\n"
        "+++ b/.github/workflows/test.yml\n"
        "@@ -1,1 +1,1 @@\n"
        "-oldwf\n+newwf\n"
    )

    stripped = gr.strip_patch_paths(
        patch,
        ["tests/test_runtime.py", ".github/workflows/test.yml"],
    )

    assert "diff --git a/src/runtime.py b/src/runtime.py" in stripped
    assert "diff --git a/tests/test_runtime.py b/tests/test_runtime.py" not in stripped
    assert (
        "diff --git a/.github/workflows/test.yml b/.github/workflows/test.yml"
        not in stripped
    )


def test_test_targets_prefer_exact_node_ids():
    instance = {
        "FAIL_TO_PASS": ["tests/test_runtime.py::test_bugfix"],
        "PASS_TO_PASS": [
            "tests/test_runtime.py::test_regression",
            "tests/test_runtime.py",
        ],
        "test_changed_file_paths": ["tests/test_runtime.py"],
    }

    assert gr.test_targets(instance) == [
        "tests/test_runtime.py::test_bugfix",
        "tests/test_runtime.py::test_regression",
    ]
    assert gr.test_file_targets(instance) == ["tests/test_runtime.py"]


def test_ordered_test_targets_follow_pytest_collection_order(monkeypatch):
    instance = {
        "test_command": "pytest --asyncio-mode=auto",
        "FAIL_TO_PASS": ["tests/test_runtime.py::test_bugfix"],
        "PASS_TO_PASS": [
            "tests/test_runtime.py::test_second",
            "tests/test_runtime.py::test_first",
        ],
    }

    monkeypatch.setattr(
        gr,
        "_docker_exec",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=["docker", "exec"],
            returncode=0,
            stdout="\n".join(
                [
                    "tests/test_runtime.py::test_first",
                    "tests/test_runtime.py::test_bugfix",
                    "tests/test_runtime.py::test_second",
                ]
            ),
        ),
    )

    ordered = gr.ordered_test_targets("cid", instance, cwd="/testbed")

    assert ordered == [
        "tests/test_runtime.py::test_first",
        "tests/test_runtime.py::test_bugfix",
        "tests/test_runtime.py::test_second",
    ]


def test_restore_patch_files_to_base_handles_new_and_existing_files(monkeypatch):
    commands: list[str] = []

    def fake_docker_exec(_container_id, command, _cwd, **kwargs):
        commands.append(command)
        if command == "git cat-file -e HEAD:tests/test_existing.py":
            return subprocess.CompletedProcess(
                args=["docker", "exec"], returncode=0, stdout=""
            )
        if command == "git cat-file -e HEAD:tests/test_new.py":
            return subprocess.CompletedProcess(
                args=["docker", "exec"], returncode=1, stdout=""
            )
        return subprocess.CompletedProcess(
            args=["docker", "exec"], returncode=0, stdout=""
        )

    monkeypatch.setattr(gr, "_docker_exec", fake_docker_exec)

    gr._restore_patch_files_to_base(
        "cid",
        ["tests/test_existing.py", "tests/test_new.py"],
        cwd="/testbed",
    )

    assert commands == [
        "git cat-file -e HEAD:tests/test_existing.py",
        "git checkout -- tests/test_existing.py",
        "git cat-file -e HEAD:tests/test_new.py",
        "rm -f tests/test_new.py",
    ]


def test_instance_cwd_defaults_to_testbed():
    assert gr.instance_cwd({}) == "/testbed"
    assert gr.instance_cwd({"workdir": "/repo"}) == "/repo"


def test_base_tree_initializer_does_not_apply_test_patch(monkeypatch):
    commands: list[str] = []
    applied_patches: list[str] = []

    def fake_docker_exec(_container_id, command, _cwd, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            args=["docker", "exec"], returncode=0, stdout=""
        )

    monkeypatch.setattr(gr, "_docker_exec", fake_docker_exec)
    monkeypatch.setattr(
        gr,
        "_apply_patch_text",
        lambda _container_id, _cwd, patch, **kwargs: applied_patches.append(patch),
    )

    instance = {
        "base_commit": "abc123",
        "test_patch": "OFFICIAL TEST PATCH",
        "workdir": "/testbed",
    }
    gr.initialize_generic_pr_base_tree("cid", instance)

    assert commands == [
        "git reset --hard && git clean -fd",
        "git checkout abc123",
    ]
    assert applied_patches == []


def test_evaluator_initializer_applies_official_test_patch(monkeypatch):
    calls: list[tuple[str, dict]] = []
    applied_patches: list[str] = []
    monkeypatch.setattr(
        gr,
        "initialize_generic_pr_base_tree",
        lambda container_id, instance, **kwargs: calls.append((container_id, instance)),
    )
    monkeypatch.setattr(
        gr,
        "_apply_patch_text",
        lambda _container_id, _cwd, patch, **kwargs: applied_patches.append(patch),
    )

    instance = {
        "base_commit": "abc123",
        "test_patch": "OFFICIAL TEST PATCH",
        "workdir": "/testbed",
    }
    gr.initialize_generic_pr_container("cid", instance)

    assert calls == [("cid", instance)]
    assert applied_patches == ["OFFICIAL TEST PATCH"]


def test_actor_workspace_uses_base_tree_initializer(monkeypatch):
    from src.tasks.codebase_adaptation import task as task_module

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        task_module,
        "initialize_generic_pr_base_tree",
        lambda container_id, instance: calls.append((container_id, instance)),
    )

    task = task_module.CodebaseAdaptationTask()
    task._env = SimpleNamespace(container_id="cid")
    instance = SimpleNamespace(
        instance_id="example__repo-1",
        raw_data={"base_commit": "abc123", "test_patch": "OFFICIAL TEST PATCH"},
    )

    task._initialize_generic_pr_workspace(instance)

    assert calls == [("cid", instance.raw_data)]


def test_evaluate_generic_pr_submission_restores_test_patch_before_running_tests(
    monkeypatch,
):
    applied_patches: list[str] = []
    commands: list[str] = []

    monkeypatch.setattr(gr, "_start_generic_container", lambda *args, **kwargs: "cid")
    monkeypatch.setattr(
        gr, "initialize_generic_pr_container", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(gr, "_cleanup_container", lambda *args, **kwargs: None)

    def fake_apply_patch_text(*args, **kwargs):
        applied_patches.append(args[2])

    def fake_docker_exec(_container_id, command, _cwd, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            args=["docker", "exec"],
            returncode=0,
            stdout="ok",
        )

    monkeypatch.setattr(gr, "_apply_patch_text", fake_apply_patch_text)
    monkeypatch.setattr(gr, "_docker_exec", fake_docker_exec)

    instance = {
        "image_name": "example:latest",
        "base_commit": "abc123",
        "test_patch": "diff --git a/tests/test_runtime.py b/tests/test_runtime.py\n",
        "test_changed_file_paths": ["tests/test_runtime.py"],
        "test_command": "pytest",
        "FAIL_TO_PASS": ["tests/test_runtime.py::test_bugfix"],
    }

    result = gr.evaluate_generic_pr_submission(
        (
            "diff --git a/src/runtime.py b/src/runtime.py\n"
            "--- a/src/runtime.py\n"
            "+++ b/src/runtime.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-old\n+new\n"
            "diff --git a/tests/test_runtime.py b/tests/test_runtime.py\n"
            "--- a/tests/test_runtime.py\n"
            "+++ b/tests/test_runtime.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-oldtest\n+newtest\n"
        ),
        instance,
    )

    assert result.success is True
    assert applied_patches == [
        (
            "diff --git a/src/runtime.py b/src/runtime.py\n"
            "--- a/src/runtime.py\n"
            "+++ b/src/runtime.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-old\n+new\n"
        ),
        "diff --git a/tests/test_runtime.py b/tests/test_runtime.py\n",
    ]
    assert "git checkout -- tests/test_runtime.py" in commands
    assert commands[-1] == "pytest tests/test_runtime.py::test_bugfix"
