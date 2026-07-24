<!-- Source: https://continual-learning-bench.com/docs/submitting/ -->
<!-- Fetched: 2026-05-05 12:49 UTC -->

# Submitting Results

This page describes how to run a full benchmark evaluation and submit results to the public leaderboard.

## Running the full evaluation

Use the `default` schedule for any result you intend to report. Do not use `quick_test` or custom schedule overrides for submitted results.

```
# Run a single task on the default schedule
clbench run exploitable_poker --schedule default --system my_system

# Run all tasks (recommended for full leaderboard submissions)
clbench run-all --name my-system-full --system my_system
```

This will by default run your system for 5 runs on each task, using the default schedules for all tasks.

Single-task results save to `results/<task>/`. Full `run-all` results save to `final_results/runs/<name>/`, including a `manifest.json`, task artifacts, logs, and metric exports.

## Managing full runs

Full benchmark runs can be expensive, so `run-all` supports a few workflow flags that are useful during development and recovery:

```
# Run a subset of tasks into a named final_results/runs/<name>/ folder
clbench run-all --name my-system-dev --system my_system --task exploitable_poker codebase_adaptation

# Resume an existing named run by filling in tasks that are still missing
clbench run-all --name my-system-full --system my_system --missing-only

# Add newly selected tasks to an existing named run
clbench run-all --name my-system-full --system my_system --append --task database_exploration

# Re-run selected tasks while archiving the previous task artifacts/logs
clbench run-all --name my-system-full --system my_system --overwrite-task --task exploitable_poker
```

Use these flags for iterative development, debugging, and recovering interrupted runs. For leaderboard submissions, the final artifact should still represent a complete `default`-schedule evaluation for the submitted system.

### Controlling parallelism

`run-all` exposes two independent parallelism knobs:

- `--task-parallelism N` (alias `-j`, default 4) — how many task subprocesses run concurrently.
- `--per-task-parallelism N` — forwarded to each inner `clbench run` as `--max-workers`, capping concurrent rollout/baseline workers within a single task. Defaults to whatever each task's schedule specifies (typically 12).

This second knob is especially useful for staying under provider rate limits. For example, to run 2 tasks but cap each task to 4 concurrent Anthropic conversations:

```
clbench run-all --name my-system-anthropic --system icl \
  --system.model claude-sonnet-4-6 \
  --task-parallelism 2 --per-task-parallelism 4
```

## What to include in a submission

A leaderboard submission pull request should include:

- System name and a brief description (1-2 sentences)
- Model or models used (including provider and version)
- Mean gain and per-task scores from a full `default`-schedule run (i.e. the artifacts in `final_results` from the result of your run-all command).
- A link to the system implementation (public repository or pull request)
- Date the evaluation was run

<div class="callout callout-note">
<span class="callout-label">Questions</span>
<p>Reach out on <a href="https://discord.gg/7bxjNdfbfH" target="_blank" rel="noopener">Discord</a> before submitting if you have questions about the process or eligibility of your system.</p>
</div>
