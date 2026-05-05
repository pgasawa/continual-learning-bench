import builtins
import sys
import types

import pytest

from src.cli import resolve_parameters
from src.interface import EvalMetrics, TaskResult
from src.registry import get_class_params
from src.runtime.runner import _finalize_task_result, run_task
from src.runs.common import filter_init_params
from src.tasks import environments as envs
from src.tasks.blind_spectrum_monitoring.task import BlindSpectrumMonitoringTask
from src.tasks.codebase_adaptation import generic_runtime as gr
from src.tasks.codebase_adaptation.task import CodebaseAdaptationTask
from src.tasks.sales_prediction.task import SalesPredictionTask


class FakeEnv:
    backend = "daytona"

    def __init__(self):
        self.commands: list[str] = []
        self.writes: list[tuple[str, str]] = []
        self.cleaned = False
        self.cleanup_failed = False

    def execute(self, command: str, *, timeout: int | None = None):
        self.commands.append(command)
        return envs.ShellResult(output="ok", returncode=0)

    def write_text(self, path: str, content: str) -> None:
        self.writes.append((path, content))

    def read_text(self, path: str, *, timeout: int | None = None) -> str:
        return "contents"

    def cleanup(self, *, failed: bool = False) -> None:
        self.cleanup_failed = failed
        self.cleaned = True

    def metadata(self):
        return {"backend": self.backend, "sandbox_id": "fake"}


def test_supported_tasks_default_to_local_docker_backend():
    assert SalesPredictionTask().environment_backend == "local_docker"
    assert CodebaseAdaptationTask().environment_backend == "local_docker"


def test_supported_tasks_accept_daytona_backend():
    assert (
        SalesPredictionTask(environment_backend="daytona").environment_backend
        == "daytona"
    )
    assert (
        CodebaseAdaptationTask(environment_backend="daytona").environment_backend
        == "daytona"
    )


def test_supported_tasks_reject_invalid_backend_value():
    with pytest.raises(ValueError, match="Unsupported environment_backend"):
        SalesPredictionTask(environment_backend="bad")  # type: ignore[arg-type]


def test_cli_rejects_invalid_backend_value_for_supported_task():
    with pytest.raises(ValueError, match="Unsupported environment_backend"):
        resolve_parameters(
            get_class_params(SalesPredictionTask),
            {},
            {"environment_backend": "bad"},
            "task",
        )


def test_unsupported_task_environment_backend_fails_clearly():
    with pytest.raises(ValueError, match="does not support environment_backend"):
        filter_init_params(
            BlindSpectrumMonitoringTask,
            {"environment_backend": "daytona"},
        )


def test_unsupported_task_daytona_param_fails_clearly():
    with pytest.raises(ValueError, match="does not support daytona_target"):
        filter_init_params(BlindSpectrumMonitoringTask, {"daytona_target": "us"})


def test_cli_rejects_unsupported_environment_backend_param():
    with pytest.raises(ValueError, match="does not support environment_backend"):
        resolve_parameters(
            get_class_params(BlindSpectrumMonitoringTask),
            {"environment_backend": "daytona"},
            {},
            "task",
        )


def test_cli_rejects_unsupported_daytona_param():
    with pytest.raises(ValueError, match="does not support daytona_target"):
        resolve_parameters(
            get_class_params(BlindSpectrumMonitoringTask),
            {"daytona_target": "us"},
            {},
            "task",
        )


def test_daytona_missing_credentials_fail_fast(monkeypatch):
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    monkeypatch.delenv("DAYTONA_JWT_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="Daytona credentials are missing"):
        envs.create_shell_environment(
            envs.EnvironmentConfig(backend="daytona", image="python:3.13")
        )


def test_daytona_jwt_requires_organization(monkeypatch):
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    monkeypatch.setenv("DAYTONA_JWT_TOKEN", "test-token")
    monkeypatch.delenv("DAYTONA_ORGANIZATION_ID", raising=False)

    with pytest.raises(RuntimeError, match="DAYTONA_ORGANIZATION_ID"):
        envs.create_shell_environment(
            envs.EnvironmentConfig(backend="daytona", image="python:3.13")
        )


def test_daytona_missing_sdk_fails_only_when_selected(monkeypatch):
    monkeypatch.setenv("DAYTONA_API_KEY", "test-key")

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "daytona":
            raise ModuleNotFoundError("No module named 'daytona'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    envs._DaytonaClientManager._client = None

    with pytest.raises(RuntimeError, match="Daytona SDK is not installed"):
        envs.create_shell_environment(
            envs.EnvironmentConfig(backend="daytona", image="python:3.13")
        )


def test_daytona_snapshot_mapping_fails_when_incomplete(monkeypatch):
    install_fake_daytona(monkeypatch)
    monkeypatch.setenv("DAYTONA_API_KEY", "test-key")

    with pytest.raises(ValueError, match="no entry for image"):
        envs.create_shell_environment(
            envs.EnvironmentConfig(
                backend="daytona",
                image="python:3.13",
                daytona_snapshot_by_image={"other:latest": "snap"},
            )
        )


def test_daytona_environment_uses_persistent_session(monkeypatch):
    fake_daytona = install_fake_daytona(monkeypatch)
    monkeypatch.setenv("DAYTONA_API_KEY", "test-key")

    env = envs.create_shell_environment(
        envs.EnvironmentConfig(backend="daytona", image="python:3.13", cwd="/work")
    )
    result = env.execute("export FOO=BAR")
    env.cleanup()

    sandbox = fake_daytona.last_sandbox
    assert result.returncode == 0
    assert sandbox.process.created_sessions
    assert sandbox.process.commands[0].command == "cd /work"
    assert sandbox.process.commands[1].command == "export FOO=BAR"
    assert fake_daytona.deleted == [sandbox]


def test_daytona_environment_uses_default_command_timeout(monkeypatch):
    fake_daytona = install_fake_daytona(monkeypatch)
    monkeypatch.setenv("DAYTONA_API_KEY", "test-key")

    env = envs.create_shell_environment(
        envs.EnvironmentConfig(
            backend="daytona", image="python:3.13", command_timeout=37
        )
    )
    env.execute("echo hi")
    env.execute("echo explicit", timeout=9)

    assert fake_daytona.last_sandbox.process.timeouts[-2:] == [37, 9]


def test_daytona_submission_sentinel_is_detected_from_output(monkeypatch):
    fake_daytona = install_fake_daytona(monkeypatch)
    fake_daytona.responses.append(FakeSessionResponse(output="ok", stdout="ok"))
    fake_daytona.responses.append(
        FakeSessionResponse(
            output="\x01\x01\x01COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n",
            stdout="\x01\x01\x01COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n",
        )
    )
    monkeypatch.setenv("DAYTONA_API_KEY", "test-key")

    env = envs.create_shell_environment(
        envs.EnvironmentConfig(backend="daytona", image="python:3.13")
    )
    result = env.execute(
        "python generate_output.py && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    )

    assert result.submitted is True
    assert fake_daytona.last_sandbox.process.commands[-1].command.startswith("python")


def test_daytona_submission_sentinel_ignores_command_mentions(monkeypatch):
    fake_daytona = install_fake_daytona(monkeypatch)
    fake_daytona.responses.append(FakeSessionResponse(output="ok", stdout="ok"))
    fake_daytona.responses.append(
        FakeSessionResponse(output="notes\n", stdout="notes\n")
    )
    monkeypatch.setenv("DAYTONA_API_KEY", "test-key")

    env = envs.create_shell_environment(
        envs.EnvironmentConfig(backend="daytona", image="python:3.13")
    )
    result = env.execute("grep COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT notes.md")

    assert result.submitted is False
    assert result.output == "notes\n"


def test_daytona_backend_exception_fails_fast(monkeypatch):
    fake_daytona = install_fake_daytona(monkeypatch)
    fake_daytona.responses.append(FakeSessionResponse(output="ok", stdout="ok"))
    monkeypatch.setenv("DAYTONA_API_KEY", "test-key")

    env = envs.create_shell_environment(
        envs.EnvironmentConfig(backend="daytona", image="python:3.13")
    )
    fake_daytona.command_error = RuntimeError("sandbox is gone")

    with pytest.raises(envs.EnvironmentBackendError, match="sandbox is gone"):
        env.execute("echo hi")


def test_daytona_command_timeout_is_task_result(monkeypatch):
    fake_daytona = install_fake_daytona(monkeypatch)
    fake_daytona.responses.append(FakeSessionResponse(output="ok", stdout="ok"))
    monkeypatch.setenv("DAYTONA_API_KEY", "test-key")

    env = envs.create_shell_environment(
        envs.EnvironmentConfig(backend="daytona", image="python:3.13")
    )
    fake_daytona.command_error = TimeoutError("request timed out")

    result = env.execute("sleep 999")

    assert result.returncode == -1
    assert result.timed_out is True
    assert "timed out" in result.output


def test_daytona_snapshot_mapping_metadata_reports_snapshot(monkeypatch):
    install_fake_daytona(monkeypatch)
    monkeypatch.setenv("DAYTONA_API_KEY", "test-key")

    env = envs.create_shell_environment(
        envs.EnvironmentConfig(
            backend="daytona",
            image="python:3.13",
            daytona_snapshot_by_image={"python:3.13": "snap-1"},
        )
    )

    metadata = env.metadata()
    assert metadata["creation_mode"] == "snapshot"
    assert metadata["image"] is None
    assert metadata["source_image"] == "python:3.13"
    assert metadata["snapshot"] == "snap-1"


def test_sales_command_execution_routes_through_environment():
    task = SalesPredictionTask()
    task._env = FakeEnv()

    result = task._execute("echo hi")

    assert result["returncode"] == 0
    assert task._env.commands == ["echo hi"]


def test_task_finalization_cleans_before_evaluate():
    events: list[str] = []

    class Task:
        def cleanup(self):
            events.append("cleanup")

        def evaluate(self):
            events.append("evaluate")
            return TaskResult(
                metrics={"events": events.copy()},
                summary="done",
                eval_metrics=EvalMetrics(
                    loss_curve=[],
                    optimal_performance=0.0,
                    actual_performance=0.0,
                ),
                instance_outcomes=[],
            )

        def get_instance_outcomes(self):
            return []

    class System:
        def get_run_artifacts(self):
            return {}

    result = _finalize_task_result(Task(), System(), None, [])

    assert events == ["cleanup", "evaluate"]
    assert result.metrics["events"] == ["cleanup", "evaluate"]


def test_runner_cleans_up_when_reset_fails():
    events: list[str] = []

    class Task:
        def reset(self):
            events.append("reset")
            raise RuntimeError("reset failed")

        def cleanup(self):
            events.append("cleanup")

    class System:
        def reset(self):
            events.append("system reset")

        def consume_usage_events(self):
            return []

    with pytest.raises(RuntimeError, match="reset failed"):
        run_task(Task(), System(), show_progress=False)

    assert events == ["reset", "cleanup"]


def test_codebase_daytona_evaluator_uses_environment_factory(monkeypatch):
    fake_envs: list[FakeEnv] = []

    def fake_create(config):
        fake = FakeEnv()
        fake_envs.append(fake)
        assert config.backend == "daytona"
        assert config.name_prefix == "clbench-codebase-eval"
        return fake

    monkeypatch.setattr(gr, "create_shell_environment", fake_create)

    result = gr.evaluate_generic_pr_submission(
        "diff --git a/src/pkg.py b/src/pkg.py\n",
        {
            "image_name": "example:latest",
            "base_commit": "abc123",
            "test_patch": "",
            "test_command": "pytest",
        },
        environment_backend="daytona",
    )

    assert result.success is True
    assert fake_envs
    assert any("git checkout abc123" in command for command in fake_envs[0].commands)
    assert fake_envs[0].cleaned is True
    assert result.environment_metadata == {"backend": "daytona", "sandbox_id": "fake"}


def test_codebase_evaluator_preserves_backend_failures(monkeypatch):
    class BrokenEnv(FakeEnv):
        def execute(self, command: str, *, timeout: int | None = None):
            raise envs.EnvironmentBackendError("daytona unavailable")

    monkeypatch.setattr(gr, "create_shell_environment", lambda config: BrokenEnv())

    with pytest.raises(envs.EnvironmentBackendError, match="daytona unavailable"):
        gr.evaluate_generic_pr_submission(
            "diff --git a/src/pkg.py b/src/pkg.py\n",
            {
                "image_name": "example:latest",
                "base_commit": "abc123",
                "test_patch": "",
                "test_command": "pytest",
            },
            environment_backend="daytona",
        )


def test_codebase_daytona_evaluator_scores_command_timeout(monkeypatch):
    class TimeoutEnv(FakeEnv):
        def execute(self, command: str, *, timeout: int | None = None):
            self.commands.append(command)
            if "pytest" in command:
                return envs.ShellResult(
                    output="ERROR: command timed out after 1s",
                    returncode=-1,
                    exception_info="timeout",
                    timed_out=True,
                )
            return envs.ShellResult(output="ok", returncode=0)

    monkeypatch.setattr(gr, "create_shell_environment", lambda config: TimeoutEnv())

    result = gr.evaluate_generic_pr_submission(
        "diff --git a/src/pkg.py b/src/pkg.py\n",
        {
            "image_name": "example:latest",
            "base_commit": "abc123",
            "test_patch": "",
            "test_command": "pytest",
        },
        environment_backend="daytona",
    )

    assert result.success is False
    assert result.status == "timeout"
    assert result.environment_metadata == {"backend": "daytona", "sandbox_id": "fake"}


def test_codebase_evaluator_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported environment_backend"):
        gr.evaluate_generic_pr_submission(
            "diff --git a/src/pkg.py b/src/pkg.py\n",
            {"image_name": "example:latest", "base_commit": "abc123"},
            environment_backend="bad",  # type: ignore[arg-type]
        )


def test_codebase_preserve_on_failure_passes_failed_cleanup(monkeypatch):
    old_env = FakeEnv()
    new_env = FakeEnv()

    monkeypatch.setattr(
        "src.tasks.codebase_adaptation.task.create_shell_environment",
        lambda config: new_env,
    )
    monkeypatch.setattr(
        "src.tasks.codebase_adaptation.task.initialize_generic_pr_environment",
        lambda *args, **kwargs: None,
    )

    task = CodebaseAdaptationTask(environment_backend="daytona")
    task.instances = [
        type(
            "Instance",
            (),
            {
                "image_name": "example:latest",
                "instance_id": "issue",
                "raw_data": {"runtime_mode": gr.GENERIC_PR_RUNTIME, "workdir": "/repo"},
            },
        )()
    ]
    task._env = old_env
    task._env_failed = True

    task._start_container()

    assert old_env.cleanup_failed is True
    assert task._env is new_env
    assert task._env_failed is False


def test_codebase_setup_failure_is_preserved_on_cleanup(monkeypatch):
    fake_env = FakeEnv()
    monkeypatch.setattr(
        "src.tasks.codebase_adaptation.task.create_shell_environment",
        lambda config: fake_env,
    )

    task = CodebaseAdaptationTask(
        environment_backend="daytona", daytona_preserve_on_failure=True
    )
    task.instances = [
        type(
            "Instance",
            (),
            {
                "image_name": "example:latest",
                "instance_id": "legacy",
                "raw_data": {"runtime_mode": "legacy", "workdir": "/repo"},
            },
        )()
    ]

    with pytest.raises(ValueError, match="generic PR"):
        task.select_run_instances(None)
    task.cleanup()

    assert fake_env.cleanup_failed is True


def test_sales_preserve_on_failure_passes_failed_cleanup(monkeypatch):
    old_env = FakeEnv()
    new_env = FakeEnv()
    monkeypatch.setattr(
        "src.tasks.sales_prediction.task.create_shell_environment",
        lambda config: new_env,
    )

    task = SalesPredictionTask(environment_backend="daytona")
    task._env = old_env
    task._env_failed = True

    task._start_container()

    assert old_env.cleanup_failed is True
    assert task._env is new_env
    assert task._env_failed is False


def test_sales_setup_failure_is_preserved_on_cleanup(monkeypatch):
    fake_env = FakeEnv()
    monkeypatch.setattr(
        "src.tasks.sales_prediction.task.create_shell_environment",
        lambda config: fake_env,
    )

    task = SalesPredictionTask(
        environment_backend="daytona", daytona_preserve_on_failure=True
    )
    task.instances = [object()]
    monkeypatch.setattr(
        task,
        "_push_data_room",
        lambda: (_ for _ in ()).throw(RuntimeError("upload failed")),
    )

    with pytest.raises(RuntimeError, match="upload failed"):
        task.select_run_instances(None)
    task.cleanup()

    assert fake_env.cleanup_failed is True


def test_sales_preserve_on_failure_stays_sticky_across_instances():
    task = SalesPredictionTask(environment_backend="daytona")
    task._env_failed = True

    outcome = task._build_instance_outcome(
        {
            "instance_idx": 0,
            "target_year": 2027,
            "forecast_years": [2027],
            "steps": 1,
            "score": 1.0,
            "format_valid": True,
            "timed_out": False,
        }
    )
    task._env_failed = task._env_failed or not bool(outcome.success)

    assert task._env_failed is True


def install_fake_daytona(monkeypatch):
    envs._DaytonaClientManager._client = None
    module = types.ModuleType("daytona")
    fake_daytona = FakeDaytona()
    module.Daytona = lambda config=None: fake_daytona
    module.DaytonaConfig = FakeDaytonaConfig
    module.CreateSandboxFromImageParams = FakeParams
    module.CreateSandboxFromSnapshotParams = FakeParams
    module.SessionExecuteRequest = FakeSessionExecuteRequest
    monkeypatch.setitem(sys.modules, "daytona", module)
    return fake_daytona


class FakeDaytonaConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeSessionExecuteRequest:
    def __init__(self, command: str):
        self.command = command


class FakeSessionResponse:
    def __init__(
        self,
        output: str = "ok",
        stdout: str = "ok",
        stderr: str = "",
        exit_code: int = 0,
    ):
        self.output = output
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class FakeProcess:
    def __init__(self, daytona):
        self.daytona = daytona
        self.created_sessions: list[str] = []
        self.commands: list[FakeSessionExecuteRequest] = []
        self.timeouts: list[int | None] = []

    def create_session(self, session_id: str):
        self.created_sessions.append(session_id)

    def execute_session_command(self, session_id, request, timeout=None):
        if self.daytona.command_error is not None:
            raise self.daytona.command_error
        self.commands.append(request)
        self.timeouts.append(timeout)
        if self.daytona.responses:
            return self.daytona.responses.pop(0)
        return FakeSessionResponse()

    def delete_session(self, session_id):
        pass


class FakeSandbox:
    id = "sandbox-id"
    name = "sandbox-name"
    target = "us"
    state = "started"
    cpu = 1
    memory = 1
    disk = 3
    network_block_all = True
    network_allow_list = None

    def __init__(self, daytona):
        self.process = FakeProcess(daytona)

    def update_network_settings(self, **kwargs):
        self.network_block_all = kwargs.get("network_block_all")


class FakeDaytona:
    def __init__(self):
        self.last_sandbox: FakeSandbox | None = None
        self.deleted: list[FakeSandbox] = []
        self.responses: list[FakeSessionResponse] = []
        self.command_error: Exception | None = None
        self.created_params: list[FakeParams] = []

    def create(self, params, timeout=60):
        self.created_params.append(params)
        self.last_sandbox = FakeSandbox(self)
        return self.last_sandbox

    def delete(self, sandbox):
        self.deleted.append(sandbox)
