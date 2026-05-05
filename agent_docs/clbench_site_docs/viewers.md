<!-- Source: https://continual-learning-bench.com/docs/viewers/ -->
<!-- Fetched: 2026-05-05 12:49 UTC -->

# Viewers

Continual Learning Bench writes lightweight live artifacts while a run is in progress and compressed viewer artifacts after it finishes. By default, the CLI prints a **live URL** that polls the current run. After the run finishes, the live viewer will shut down and you can reopen the saved artifacts with one of the viewers below.

## Single Task Viewer

Single-task runs save their viewer artifact under `results/<task>/`. The most useful file for inspection is usually:

```
results/<task>/viewer_artifact_*.json.gz
```

Open the single-task viewer and load that artifact:

```
viewers/single_task_viewer.html
```

The single-task viewer shows reward and gain curves, baseline comparisons, per-instance prompts and responses, feedback, usage/cost metadata, and task-specific artifacts when available.

## Comparison Trace Viewer

The comparison viewer compares multiple completed task artifacts side by side. It is most useful for comparing different systems or runs on the same task. Open the viewer and load multiple `viewer_artifact_*.json.gz` files (or the appropriate task-named file such as codebase_adaption.json.gz if using the run-all command):

```
viewers/compare_traces.html
```

## Run-all Viewer

For multi-task runs created with `clbench run-all ...`, the CLI writes an aggregate manifest and task artifacts under `final_results/runs/<run-name>/`:

```
final_results/runs/<run-name>/manifest.json
final_results/runs/<run-name>/task_artifacts.json.gz
```

Open the run-all viewer:

```
viewers/run_all_viewer.html
```

Load `manifest.json` to see every task from the run in a single view. Then load `task_artifacts.json.gz`, manually when prompted; this enables embedded drill-down views and lets the top-level **Total Cost** tile compute the true summed cost from traces/artifacts.

In the task table, **Mean Run Cost** is the mean cost across rollout runs for that task. The top-level **Total Cost** tile is different: it is the sum of available baseline and run costs across tasks.

## What to Look For

When reviewing a trace, check:

- Mean Reward: a good system should see a generally increasing trend in mean reward
- Gain: a good system would have positive gain (means that the system's performance on that sub-task is
better having done the previous task compared to the stateless baseline)
- Metadata: expand specific sub-tasks to view per-run metadata. They usually contain system-specific info
such as notepad contents for the notepad system.
