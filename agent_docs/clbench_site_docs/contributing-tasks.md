<!-- Source: https://continual-learning-bench.com/docs/contributing-tasks/ -->
<!-- Fetched: 2026-05-05 12:49 UTC -->

# Contributing Tasks

This guide covers what makes a good continual learning task and implementation details on how to contribute one.

Task contributions that are accepted will be eligible for authorship on future benchmark papers we plan to submit to a top-ML venue. If you are proposing a new domain or task sequence, we recommend reaching out early on [Discord](https://discord.gg/7bxjNdfbfH) so we can align on scope, evaluation design, and attribution before you invest in a full implementation. The exact authorship criteria is reserved to the discretion of project leads.

## Design Heuristics

A good task should meet these criteria:

- **There is a mechanism for learning.** Task instances must provide observations that carry learning signals such that a sufficiently capable system can use this feedback to improve its performance on future instances. One common way to achieve this is by embedding a hidden latent structure into task instances, such as a poker opponent using a specific strategy, a user preference, etc.
- **Initial performance leaves headroom.** If frontier models now or in the near future can achieve near-perfect rewards on many task instances, there is no room to measure continuous improvement. This applies in both stateless settings (models should not do well when given the task without prior experiences) and learned settings (current systems should not demonstrate significant learning & achieve high rewards on most future instances after experiencing a few instances).
- **Humans would also improve over time.** A useful sanity check is if a human would improve meaningfully across instances. If not, the task may be too noisy.

## Implementing a Task

Use the CLI to generate a task scaffold:

```
clbench init task my_task
```

This creates `src/tasks/my_task/` with a skeleton `task.py` and a `schedules/` directory. A task subclasses `ContinualLearningTask`, registers itself with `@register_task("my_task")`, and implements 4 methods plus a `name` property.

If you are using an AI coding agent, point it at the repository `AGENTS.md`, `src/tasks/AGENTS.md`, and the [create-task skill](https://github.com/pgasawa/continual-learning-bench/blob/main/skills/create-task/SKILL.md). The skill walks agents through task design, prompt/action/observation gates, tests, debug runs, and README documentation. Other reusable agent workflows live in the [skills directory](https://github.com/pgasawa/continual-learning-bench/tree/main/skills).

To implement a task, you will need:

- 4 required methods: `build_canonical_run_state`, `build_current_query`, `step`, and `evaluate`
- Correctly set up constructors and attributes
- A class-level `r_max` for cross-task normalization
- A schedule that defines in what order to run the task instances

## Task Interface - Required Methods

### 1. `build_canonical_run_state() -> None`

This method builds the `self.instances` list, which holds the canonical ordered sequence the run will consume. The element type is up to you -- whatever makes the task easy to write. For example: `drifted_python` stores `Problem` objects (the puzzle to solve), and `codebase_adaptation` stores `TaskInstance` objects loaded from a JSONL dataset (one per GitHub issue). This method should also reset per-run state: clear any counters, history lists, environment handles, and `self._instance_outcomes` you maintain.

Instance generation must be deterministic. Use `random.Random(self.seed)` -- never the global `random` module -- so the canonical order is reproducible.

### 2. `build_current_query() -> Query`

Return the `Query` for whichever instance is currently active. Called **once** at the start of a run to produce the first prompt; subsequent prompts flow out of `step()` instead. A `Query` carries a natural-language prompt, a Pydantic response schema, stable `instance_id`/`instance_index` identifiers, and optional metadata.

### 3. `step(response: Response) -> TaskStepResult`

The core loop. Each call must:

1. **Process the response** -- score it, update internal state (chip counts, problem index, container state, ...), and append an `InstanceOutcome` to `self._instance_outcomes` whenever the *current instance* finishes.
2. **Decide the instance boundary** -- set `observation.instance_complete = True` if this step ends an instance (e.g. a poker hand finished, a problem solved, the budget for an issue exhausted), `False` if you are still mid-instance (e.g. waiting for the next betting action, or the next exploration attempt). Systems read this flag to decide when to consolidate memory.
3. **Build the next prompt** -- return the next `Query` in `next_query`, or `None` together with `done=True` when the run is over.

A single `step()` can advance through multiple instances if needed (e.g. when the system folds and the task auto-resolves the rest of the hand) -- just keep `_instance_outcomes` in sync.

### 4. `evaluate() -> TaskResult`

Called once after the run finishes. Compute task-specific metrics from your accumulated state and return a `TaskResult` with `instance_outcomes`, an `EvalMetrics` (loss curve, optimal/actual performance), and a human-readable summary.

If your task only needs mean per-instance reward, `standard_evaluate(self._instance_outcomes)` does the bookkeeping for you.

## Task Interface - Attributes & Constructors

### Constructor Requirements & Attributes

`__init__` must accept `num_instances` (schedules use this to control run length across all tasks) and `seed` (used for deterministic instance generation). Beyond that, take whatever variant/schedule/config arguments your task needs. Additionally, you need a `name` property.

```python
@register_task("my_task")
class MyTask(ContinualLearningTask):
    r_max = 1.0

    def __init__(self, num_instances: int = 50, seed: int = 0, **kwargs):
        self.num_instances = num_instances
        self.seed = seed
        self.instances: list[MyInstance] = []
        self._current_index = 0
        self._instance_outcomes: list[InstanceOutcome] = []

    @property
    def name(self) -> str:
        return "my_task"

    def build_canonical_run_state(self) -> None:
        rng = random.Random(self.seed)
        self.instances = [self._generate_instance(rng) for _ in range(self.num_instances)]
        self._current_index = 0
        self._instance_outcomes = []
```

### `r_max`

Every registered task must define `r_max` on the task class. This is the mean per-instance maximum reward for the default reported schedule, on the same scale as `InstanceOutcome.reward`:

```python
r_max = cumulative_max_reward_for_default_schedule / num_instances
```

If per-instance ceilings vary, average the per-instance ceilings over the default schedule. If the task uses a natural reward scale, such as poker profit in big blinds or information gain in bits, document how the reference maximum was computed and keep the value with the task implementation.

### General Control Flow

<ol>
<li>The framework calls <code>task.reset()</code>. The default <code>reset()</code> chains three of the methods you implement:
<ul>
<li><code>build_canonical_run_state()</code> -- materialize the canonical instance sequence into <code>self.instances</code> and zero out per-run state.</li>
<li><code>select_run_instances(indices)</code> -- narrow <code>self.instances</code> to the active subset for this run (the default just trims the list; only override if you have side state to keep in sync).</li>
<li><code>build_current_query()</code> -- render the <strong>first</strong> prompt and return it as the seed <code>Query</code>.</li>
</ul>
</li>
<li>The interaction loop runs: <code>system.respond(query)</code> → <code>task.step(response)</code> → <code>system.observe(observation, next_query)</code>. <strong>Every query after the first comes from the <code>next_query</code> field returned by <code>step()</code></strong>, not from <code>build_current_query()</code>. So <code>build_current_query()</code> is only ever invoked once per run, to kick things off.</li>
<li>When <code>step()</code> returns <code>done=True</code> (and <code>next_query=None</code>), the framework calls <code>evaluate()</code> to produce the final <code>TaskResult</code>.</li>
</ol>

## Schedules

Every task needs at least a `schedules/default.json`. This is the schedule used by `clbench run-all` and referenced in reported results. Add a `quick_test` schedule with a small number of instances for fast development iteration.

```json
// schedules/default.json
{
  "num_instances": 100,
  "description": "Standard evaluation schedule"
}

// schedules/quick_test.json
{
  "num_instances": 10,
  "description": "Fast development schedule"
}
```

## Validating & Testing

Run the sanity checks below & full runs against some systems for a full view of your task.

```
clbench validate task my_task          # structural checks
clbench smoke my_task --system icl     # one-interaction wiring check
```

<div class="callout callout-note">
<span class="callout-label">Discussion</span>
<p>Have a task idea but not sure if it's a good fit? Discuss with us and the community on <a href="https://discord.gg/7bxjNdfbfH" target="_blank" rel="noopener">Discord</a> before investing in a full implementation.</p>
</div>
