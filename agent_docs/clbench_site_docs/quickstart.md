<!-- Source: https://continual-learning-bench.com/docs/quickstart/ -->
<!-- Fetched: 2026-05-05 12:49 UTC -->

# Quick Start

First, make sure you are set up by following the [installation guide](installation.md).

## Exploring Tasks & Systems

Use `clbench list` to see what's available:

```
clbench list                                  # all tasks and systems
clbench inspect task exploitable_poker        # schedules, variants, accepted params
clbench inspect system icl
```

## Starting Your First Run

Run the `quick_test` schedule on the exploitable poker task with the in-context learning baseline system:

```
clbench run exploitable_poker --schedule quick_test --system icl
```

When the run starts, a live progress URL is printed to the terminal. Open it in a browser to watch the learning curve update in real time. Results are written to `results/<task>/`; after the run finishes, load the newest `viewer_artifact_*.json.gz` in `viewers/single_task_viewer.html`.

After the run, use the viewer to look at the logs:

```
open viewers/single_task_viewer.html
```

## Common Run Options

### Run with a pinned config file for full reproducibility

```
clbench run --config=configs/exploitable_poker/exploitable_poker_icl.json
```

### Override individual parameters inline

```
clbench run exploitable_poker --schedule quick_test --system icl --system.model gpt-4o --task.num-instances 20
```

Use `clbench inspect` to see all accepted parameters for a given task or system.

### Run a system against ALL tasks

```
clbench run-all --name icl-full --system icl
```

<div class="callout callout-note">
<span class="callout-label">Next steps</span>
<p>Continue to <a href="../metrics/">Metrics</a> to see how runs are scored.</p>
</div>
