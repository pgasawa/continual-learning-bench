<!-- Source: https://continual-learning-bench.com/docs/installation/ -->
<!-- Fetched: 2026-05-05 12:49 UTC -->

# Installation

Continual Learning Bench requires [uv](https://github.com/astral-sh/uv) and Python 3.13 or later. Currently, running all tasks also requires a local installation of Docker.

## 1. Install from source

Clone the repository and install the benchmark with all task dependencies:

```
git clone https://github.com/pgasawa/continual-learning-bench && cd continual-learning-bench
uv sync --all-extras && source .venv/bin/activate
pre-commit install && clbench setup --all # Downloads all required task files and images.
```

## 2. Set up API keys

Add any provider API keys to a `.env` file in the project root.

```
# .env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

## 3. Verify that the CLI works

Use `clbench list` to see available tasks and systems:

```
clbench list
```

<div class="callout callout-note">
<span class="callout-label">Next steps</span>
<p>Continue to <a href="../concepts/">Concepts</a> for the vocabulary used throughout the rest of the docs (tasks, sub-tasks, schedules, the interaction loop).</p>
</div>
