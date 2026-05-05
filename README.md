# Continual Learning Bench

<p align="center">
  <img src="assets/logo.png" alt="Continual Learning Bench" width="200"/>
</p>

[![Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white)](https://discord.gg/7bxjNdfbfH) [![Website](https://img.shields.io/badge/Website-continual--learning--bench.com-2ea44f?logo=googlechrome&logoColor=white)](https://continual-learning-bench.com/) [![Docs](https://img.shields.io/badge/Docs-continual--learning--bench.com%2Fdocs-blue?logo=readthedocs&logoColor=white)](https://continual-learning-bench.com/docs/)

**Continual Learning Bench** measures how well AI agents learn from past environment interactions, the defining challenge for agents that operate over extended horizons and are expected to improve online.

## Quickstart

Continual Learning Bench requires Python 3.13 or later, [uv](https://github.com/astral-sh/uv), and a local Docker installation for tasks and systems that run containerized workspaces. Shell/workspace tasks can also opt into Daytona task environments with the `daytona` extra.

```bash
git clone https://github.com/pgasawa/continual-learning-bench && cd continual-learning-bench
uv sync --all-extras && source .venv/bin/activate
pre-commit install
clbench setup --all
```

Add any model provider keys you need to a `.env` file in the project root, then verify the CLI and run a small benchmark:

```bash
clbench list
clbench run exploitable_poker --schedule quick_test --system icl
```

Local Docker is the default task-environment backend. Sales Prediction and Codebase Adaptation also accept `environment_backend="daytona"` when the Daytona SDK is installed and credentials are exported through `DAYTONA_API_KEY` or `DAYTONA_JWT_TOKEN` plus `DAYTONA_ORGANIZATION_ID`:

```bash
uv sync --extra daytona --extra sales_prediction --extra codebase_adaptation
clbench run --config=configs/sales_prediction/sp_lifecycle_icl_daytona.json
clbench run --config=configs/codebase_adaptation/codebase_adaptation_icl_daytona.json
```

Daytona runs create task-side sandboxes from the task Docker image by default, block runtime network for the offline task environment, and delete sandboxes during cleanup. Optional task params include `daytona_snapshot`, `daytona_snapshot_by_image`, `daytona_target`, `daytona_auto_delete_interval`, and `daytona_preserve_on_failure`.

The [Quickstart Guide](https://continual-learning-bench.com/docs/quickstart/) walks through the same flow in more detail, including how to inspect tasks, systems, schedules, and run outputs.

## Further Documentation

- [Installation](https://continual-learning-bench.com/docs/installation/) — Full setup notes and prerequisites
- [Task Gallery](https://continual-learning-bench.com/tasks.html) — Browse all available tasks
- [Leaderboard](https://continual-learning-bench.com/leaderboard.html) — See how models compare
- [Viewers](https://continual-learning-bench.com/docs/viewers/) — Viewing and comparing results
- [Docs](https://continual-learning-bench.com/docs) — Concepts, metrics, contribution guides, and more

## Core Components

### Dataset of Tasks

Each task spans multiple episodes in a shared environment, so agents that remember feedback and adapt should outperform those that solve each instance from scratch. Tasks include:

- a set of constructed task instances with a learnable structure,
- an evaluation script and reward metric measuring improvement over episodes,
- schedules and variants for repeatable comparisons across systems.

Tasks live in `src/tasks/`. See the [Task Gallery](https://continual-learning-bench.com/tasks.html) for an overview that's easy to browse.

### Systems

Systems are the agents evaluated by the benchmark. Built-in baselines and model-backed systems live in `src/systems/`, and custom systems can be added to test new memory, retrieval, prompting, or tool-use strategies.

### Execution Harness

The harness connects systems to task environments, manages multi-episode rollouts, records traces, and writes viewer artifacts for analysis. After installing, explore available options with:

```bash
clbench run --help
clbench run-all --help
```

## Contribution

- [Roadmap](https://continual-learning-bench.com/docs/roadmap/) — See what we're working on and where you can help
- [Contributing a New Task](https://continual-learning-bench.com/docs/contributing-tasks/) — Add a benchmark environment
- [Contributing a New System](https://continual-learning-bench.com/docs/contributing-systems/) — Add an evaluated agent

Contributions are welcome, especially new tasks that stress-test long-horizon learning. Reach out on [Discord](https://discord.gg/7bxjNdfbfH) to discuss ideas before diving in.

## Citing Us

If you found Continual Learning Bench useful, please cite us as:

```bibtex
@misc{clbench2026,
      title={Continual Learning Bench},
      author={Parth Asawa and Chris Glaze and Gabe Orlanski and Ramya Ramakrishnan and Benji Xu and Asim Biswal and Vincent Sunn Chen and Frederic Sala and Matei Zaharia and Joseph E. Gonzalez},
      year={2026},
}
```
