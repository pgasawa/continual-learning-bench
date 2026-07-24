# Continual Learning Bench Website Docs — Combined Local Markdown

Fetched from <https://continual-learning-bench.com/> on 2026-05-05 12:49 UTC.

See `README.md` in this folder for source URLs and individual files.


---

<!-- Local file: overview.md -->

<!-- Source: https://continual-learning-bench.com/docs/ -->
<!-- Fetched: 2026-05-05 12:49 UTC -->

# Continual Learning Bench

A benchmark that measures how well AI systems learn over time.

<div class="docs-hero-logo">
<img src="https://continual-learning-bench.com/continual_learning_bench.png" alt="Continual Learning Bench" />
</div>

Continual Learning Bench is a methodology and set of task sequences for evaluating how agent systems improve from their interactions with the environment.

<div class="def-grid">
<div class="def-block">
<span class="def-label">Task</span>
<p>Each task comes with a sequence of sub-tasks, designed such that interactions with earlier sub-tasks should lead to improvement in solving later ones.</p>
</div>
<div class="def-block">
<span class="def-label">System</span>
<p>For each task, a system attempts all sub-tasks in a specified order and is evaluated after every sub-task.</p>
</div>
</div>

Most benchmarks treat model behavior as static: each example is independent, and the system being evaluated is expected to stay the same. Continual Learning Bench is built for a different setting, where systems interact with an environment, receive feedback, and should be able to change how they behave over time. The benchmark is built around a simple question: **does the system get better because of what it has experienced?**

That is why tasks are designed as sequences rather than isolated problems. Earlier sub-tasks expose structure, preferences, strategies, or facts that can help on later sub-tasks if the system learns from them.

The benchmark asks whether systems actually improve from experience: whether they learn from limited interactions (sample-efficient learning), whether useful learned structure persists after distraction or noise, and how much cost they incur while adapting. The [metrics](metrics/) page explains how these behaviors are scored.

Want to contribute a task or test out your system? Check out our [contributing tasks](contributing-tasks/) and [contributing systems](contributing-systems/) pages.



---

<!-- Local file: installation.md -->

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



---

<!-- Local file: concepts.md -->

<!-- Source: https://continual-learning-bench.com/docs/concepts/ -->
<!-- Fetched: 2026-05-05 12:49 UTC -->

# Concepts

## System

A **system** is what we aim to evaluate. It is the combination of an **agent** that responds to queries from a task and a **memory system** that carries information between task instances. The memory system might be pure in-context learning (perhaps with caching), lossy context compression (e.g. periodic summarization), parametric (fine-tuning, weight updates), or any combination of these. The benchmark scores neither component in isolation: performance on continual learning is a property of the two together. So we can ask questions like how well an agent does on tasks for a given memory system and how effective memory systems are across agents and tasks.

Systems live under `src/systems/` and implement the `ContinualLearningSystem` interface.

## Task

A **task** is a continual learning problem. It has a theme (poker, codebase, etc) and consists of a sequence of task instances that systems attempt to solve. Every task lives under `src/tasks/` and implements the `ContinualLearningTask` interface. Different tasks are unrelated to each other and evaluated independently.

## Scales of task organization

Within a task, work is organized at three scales:

- **Task instance** — the primary unit of work; reward and gain are calculated per instance.
- **Variant** — a configuration that groups similar instances; variants change *what* the task is about (e.g. a particular poker opponent, a particular codebase).
- **Schedule** — a fixed ordering of one or more variants; schedules change *how long* a run is and *in what order* the system encounters its variants.

These different scales provide the basic scaffold on which we can define continual learning problems (e.g. De Lange et al. 2022; van de Ven et al. 2022) -- accumulating task-relevant knowledge and skills within variants, with continued transfer learning and concept drift across variants (e.g. Lopez-Paz & Ranzato 2017).

Detailed definitions for each scale follow below.

## Task instance

A task instance (or sub-task) is one self-contained problem within the task: for example, a single poker hand or a single coding issue. Task instances are designed to enable learning: interactions with earlier instances should help the system perform better on later ones. Each task instance produces a reward when completed, quantifying how well the system performed on that specific instance.

## Variant

A **variant** is a configuration of a task that changes its content or difficulty (e.g. a different poker opponent, a different codebase). Variants change *what* the task is about and schedules change *how long* and *in what order* things are run. Different variants can be chained together to run sequentially by a schedule.

## Schedule

A **schedule** is a JSON file in the task's `schedules/` directory that defines how many task instances to run and in what order, alongside other task-specific parameters.

## Turns

A task instance is not always a single prompt-and-response. The task and system may go back and forth across multiple **turns** before the task instance is complete, typically in any task that involves terminal or tool-use agents interacting with their environment. For example, solving a single coding issue (single task instance) may require a system to interact with the codebase multiple times (e.g., multiple turns). The `instance_complete` flag on each observation tells the system when the current task instance has ended.

## Interaction Loop
```python
task.build_canonical_run_state() # reset task
query = task.build_current_query() # get first query
system.reset()
while True:
    response = system.respond(query)
    query, observation = task.step(response)
    system.observe(observation)
```

<div class="callout callout-note">
<span class="callout-label">Next steps</span>
<p>With the vocabulary in place, continue to <a href="../metrics/">Metrics</a> to see how runs are scored.</p>
</div>

---

## References

- **Lopez-Paz, D., Ranzato, M. (2017).** "Gradient Episodic Memory for Continual Learning." *NeurIPS 2017*. [arXiv:1706.08840](https://arxiv.org/abs/1706.08840).
- **De Lange, M., Aljundi, R., Masana, M., Parisot, S., Jia, X., Leonardis, A., Slabaugh, G., Tuytelaars, T. (2022).** "A Continual Learning Survey: Defying Forgetting in Classification Tasks." *IEEE Transactions on Pattern Analysis and Machine Intelligence* **44**(7): 3366–3385. [DOI:10.1109/TPAMI.2021.3057446](https://doi.org/10.1109/TPAMI.2021.3057446).
- **van de Ven, G. M., Tuytelaars, T., Tolias, A. S. (2022).** "Three types of incremental learning." *Nature Machine Intelligence* **4**(12): 1185–1197. [DOI:10.1038/s42256-022-00568-3](https://doi.org/10.1038/s42256-022-00568-3).



---

<!-- Local file: quickstart.md -->

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



---

<!-- Local file: metrics.md -->

<!-- Source: https://continual-learning-bench.com/docs/metrics/ -->
<!-- Fetched: 2026-05-05 12:49 UTC -->

# Metrics

Each task is designed to answer one question: **does the system perform better the more experience it has?**

---

## Reward

Each task defines a per-instance reward. Tasks have their own definition (accuracy, normalized error, profit, …), but every reward must satisfy:

- Higher is better.
- Comparable across different instances of a task.
- Reflects the task objective directly, not an unrelated proxy.

If per-instance difficulty is roughly homogeneous across the sequence, reward by itself is enough to read off learning: a system that is genuinely improving from experience will produce an upward-trending reward curve, because the curve isn't being moved by changes in instance difficulty. In that regime, "higher reward at task instance `t`" means "more was learned by instance `t`."

For example, in the "exploitable poker" task, chips are reset after each hand (so winnings are not accumulated). That means that when a system is playing against the same opponent, the basic challenge for that hand is the same as in previous hands. Increases in reward across hands can thus reflect real learning signal — systems are doing progressively better against their opponent.

The per-task performance headline is the running average reward across task instances for each system.

## Gain

<div class="hiw-wrap">
<svg class="hiw-svg" viewBox="0 0 800 270" preserveAspectRatio="xMidYMid meet" role="img" aria-label="A single task forks into two runs: a stateful system that carries state forward across sub-tasks, and a stateless baseline that resets between sub-tasks. Gain is the difference between system reward and baseline reward.">
  <defs>
    <marker id="metrics-doc-gain-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" class="arrow-head"/>
    </marker>
    <marker id="metrics-doc-gain-fork" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" class="fork-arrow-head"/>
    </marker>
  </defs>
  <rect x="10" y="105" width="90" height="60" rx="10" class="task-pill"/>
  <text x="55" y="140" text-anchor="middle" class="task-pill-text">task</text>
  <path d="M 100 135 C 115 135 115 64 130 64" class="fork-arrow" marker-end="url(#metrics-doc-gain-fork)"/>
  <path d="M 100 135 C 115 135 115 202 130 202" class="fork-arrow" marker-end="url(#metrics-doc-gain-fork)"/>
  <text x="130" y="22" class="row-label row-label-stateful">CONTINUAL LEARNING SYSTEM</text>
  <rect x="130" y="36" width="90" height="56" rx="10" class="box-stateful"/>
  <text x="175" y="71" text-anchor="middle" class="box-text">S₁</text>
  <line x1="220" y1="64" x2="240" y2="64" class="arrow" marker-end="url(#metrics-doc-gain-arrow)"/>
  <rect x="240" y="36" width="90" height="56" rx="10" class="box-stateful"/>
  <text x="285" y="71" text-anchor="middle" class="box-text">S₂</text>
  <line x1="330" y1="64" x2="350" y2="64" class="arrow" marker-end="url(#metrics-doc-gain-arrow)"/>
  <rect x="350" y="36" width="90" height="56" rx="10" class="box-stateful"/>
  <text x="395" y="71" text-anchor="middle" class="box-text">S₃</text>
  <line x1="440" y1="64" x2="460" y2="64" class="arrow" marker-end="url(#metrics-doc-gain-arrow)"/>
  <rect x="460" y="36" width="90" height="56" rx="10" class="box-stateful"/>
  <text x="505" y="71" text-anchor="middle" class="box-text">S₄</text>
  <line x1="550" y1="64" x2="595" y2="64" class="arrow" marker-end="url(#metrics-doc-gain-arrow)"/>
  <text x="605" y="68" class="end-label-reward">system reward</text>
  <text x="130" y="159" class="row-label row-label-stateless">STATELESS BASELINE</text>
  <rect x="130" y="174" width="90" height="56" rx="10" class="box-stateless"/>
  <text x="175" y="209" text-anchor="middle" class="box-text-muted">S₁</text>
  <text x="175" y="249" text-anchor="middle" class="reset-text">reset</text>
  <rect x="240" y="174" width="90" height="56" rx="10" class="box-stateless"/>
  <text x="285" y="209" text-anchor="middle" class="box-text-muted">S₂</text>
  <text x="285" y="249" text-anchor="middle" class="reset-text">reset</text>
  <rect x="350" y="174" width="90" height="56" rx="10" class="box-stateless"/>
  <text x="395" y="209" text-anchor="middle" class="box-text-muted">S₃</text>
  <text x="395" y="249" text-anchor="middle" class="reset-text">reset</text>
  <rect x="460" y="174" width="90" height="56" rx="10" class="box-stateless"/>
  <text x="505" y="209" text-anchor="middle" class="box-text-muted">S₄</text>
  <text x="505" y="249" text-anchor="middle" class="reset-text">reset</text>
  <line x1="550" y1="202" x2="595" y2="202" class="arrow-muted" marker-end="url(#metrics-doc-gain-fork)"/>
  <text x="605" y="206" class="end-label-baseline">baseline reward</text>
  <path d="M 730 64 L 740 64 L 740 202 L 730 202" class="gain-bracket"/>
  <text x="750" y="137" class="gain-label">gain</text>
</svg>
</div>

In the real world, task instance difficulty is often heterogeneous, and different systems may have different competencies at a task to begin with. This can complicate interpretation of pure reward curves, especially when comparing different systems. Gain removes these confounds by comparing the system's reward on each task instance to what the same system would have scored on that instance as a stateless system with no prior experience with the task:

```
gain_t = reward_stateful_t − reward_stateless_t
```

Whatever made instance `t` intrinsically easy or hard for that system shows up in both terms, so the pairing cancels per-instance difficulty *as captured by the paired stateless baseline*, leaving the contribution of accumulated state. (More subtle interactions between difficulty and accumulated state — where state changes how difficulty manifests — aren't fully removed.) Gain is what reward measures when you cannot assume the baseline system performance is homogeneous.

The per-task learning headline is the running average gain across task instances for each system.

Reading the sign:

- **Positive** — accumulated state helped, learning is transfering across intstances (and variants).
- **Zero** — the stateful system performed the same as its stateless self, no evidence for learning.
- **Negative** — accumulated state hurt (forgetting, distractor accumulation, hallucinated state).

## Cross-task comparison

To derive aggregate scores across tasks we need to normalize the metrics somehow. How do we average changes in accuracy on classification tasks with changes in the number of steps taken to solve a coding problem?

The benchmark addresses this by (1) requiring a well-defined "max reward achievable" to define available learning headroom (see [contributing tasks](contributing-tasks.md)) and (2) normalizing metrics with that `r_max`:

```
r_norm = (r_stateful − r_external) / (r_max − r_external)
gain_norm = (r_stateful − r_stateless) / (r_max − r_stateless)
```

`r_stateful` and `r_stateless` here are the per-task aggregates from the algorithm at the end of this page (running averages across instances and rollouts), so normalization is applied *after* aggregation rather than per instance.

The two formulas differ in their lower-bound reference: reward normalization uses a fixed external baseline so scores are comparable across the cohort, while gain normalization uses each system's own stateless reward so the score reflects that system's own learning headroom. Specifically:

- **Reward normalization** uses an external, fixed baseline (`r_external`) we derive from a frontier model using in-context learning (GPT-5.4), chosen so the score is submission-independent. A system can be ranked in isolation without depending on the cohort of submitted systems.
- **Gain normalization** uses the system's own `r_stateless` as this reference, so the (`r_max` − `r_stateless`) denominator normalizes for learning headroom. A task whose stateless baseline already sits near `r_max` would contribute an artificially small delta regardless of actual learning, while a task with a low stateless baseline would dominate the average. Dividing by the available headroom puts every task on the same "fraction of room captured" scale.

We can average these metrics for each system across tasks to then compare overall systems with respect to performance and learning. Our `gain_norm` metric has precedent in both the continual learning literature (e.g. Wołczyk et al. 2021) as well as education (e.g. Hake 1998).

`r_max` is task-defined on the same reward scale as the task's per-instance rewards. In code, every task class must define `r_max`: the mean per-instance maximum reward for the default reported schedule. Equivalently, it is the cumulative maximum reward over that schedule divided by the number of instances. Tasks can choose the reward scale that best captures learning, such as poker profit in big blinds or information gain in bits, as long as `r_max` is defined on that same scale and documented with the task.

## How metrics are calculated and averaged

As described more in [concepts](concepts.md), tasks are organized at three scales, with "task instances" as the primary units of work for calculating reward and gain, "variants" grouping similar task instances together, and "schedules" ordering different variants in a fixed sequence.

The primary mechanism for denoising results is to resample within-variant instance sequences via either permutation, resampling, or pure replication (for time-dependent sequences such as running predictions), and then average over those multiple runs (configurable per task). If multiple schedules (different variant orderings) are defined, then results can be further denoised (see [concepts](concepts.md) for more discussion here). Stateless baseline rewards are not resampled but just calculated once since there is no order dependence by default (future work may sample here too in cases where we expect significant stochasticity in system responses even at stateless instances).

**Algorithm — Per-task metric computation**

```python
# Inputs: task with schedule S, system, run_mode, r_max, r_external

# Stateless baseline: one reward per instance, no order dependence
r_stateless = [score(system, task, t, stateful=False) for t in S]

# Stateful rollout: state carries across instances; reorder per run_mode
S_n = apply_run_mode(S, run_mode)   # permute, resample, or replicate
system.reset()
r_stateful, gain = [], []
for t in S_n:
    r = score(system, task, t, stateful=True)
    r_stateful.append(r)
    gain.append(r - r_stateless[t])

# Aggregate over instances (means are further averaged across rollouts; see above)
r_stateful_mean  = mean(r_stateful)
r_stateless_mean = mean(r_stateless)
gain_mean        = mean(gain)

# Cross-task normalization
r_norm    = (r_stateful_mean - r_external)       / (r_max - r_external)
gain_norm = (r_stateful_mean - r_stateless_mean) / (r_max - r_stateless_mean)
```

<div class="callout callout-note">
<span class="callout-label">Related</span>
<p>Use the <a href="../viewers/">Viewers</a> docs to inspect reward curves, per-instance gain, and the running-average gain trace.</p>
</div>

---

## References

- **Hake, R. R. (1998).** "Interactive-engagement versus traditional methods: A six-thousand-student survey of mechanics test data for introductory physics courses." *American Journal of Physics* **66**(1): 64–74. [DOI:10.1119/1.18809](https://doi.org/10.1119/1.18809).
- **Wołczyk, M., Zając, M., Pascanu, R., Kuciński, Ł., Miłoś, P. (2021).** "Continual World: A Robotic Benchmark for Continual Reinforcement Learning." *NeurIPS 2021*. [arXiv:2105.10919](https://arxiv.org/abs/2105.10919).



---

<!-- Local file: viewers.md -->

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



---

<!-- Local file: domain-experts.md -->

<!-- Source: https://continual-learning-bench.com/docs/domain-experts/ -->
<!-- Fetched: 2026-05-05 12:49 UTC -->

# Domain Experts

Continual Learning Bench tasks start from a learnability claim: the sequence should contain structure that a capable system could discover from experience. To make that claim concrete, we engage domain experts to help confirm both the realism and learnability of tasks.

Domain experts help us check whether a task setting reflects the real domain, whether the observations expose meaningful feedback, and whether the scoring captures the behavior we actually want systems to improve at. This is especially important for tasks where realistic difficulty depends on specialized knowledge rather than just prompt formatting.

We are also working with Snorkel AI to study baseline human performance on some tasks where human performance is representative. While we expect truly continual learning systems to eventually surpass human performance in many settings, human baselines are a useful calibration tool: they help us interpret task difficulty, assess whether improvement over time is plausible, and identify when a task may be too noisy or underspecified.

<div class="callout callout-note">
<span class="callout-label">Contributing domains</span>
<p>If you have domain expertise or a task idea that would benefit from expert validation, reach out on <a href="https://discord.gg/7bxjNdfbfH" target="_blank" rel="noopener">Discord</a>.</p>
</div>



---

<!-- Local file: contributing-tasks.md -->

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



---

<!-- Local file: contributing-systems.md -->

<!-- Source: https://continual-learning-bench.com/docs/contributing-systems/ -->
<!-- Fetched: 2026-05-05 12:49 UTC -->

# Contributing Systems

This guide covers how to implement the system interface. The in-context-learning `icl` system (a simple system that keeps the entire interaction history in context) is the most standard system that could be referenced when developing a system.

## What the interface requires

Systems subclass `ContinualLearningSystem` and must implement three things:

- **`respond(query: Query) -> Response`** -- The main interaction loop. Receives a query and returns a structured response.
- **`reset() -> None`** -- Resets the system and clears all internal state and learned knowledge. Called at the start of each run.
- **`name`** -- A string property identifying the system.

## Scaffolding a system

```
clbench init system my_system
```

This creates a `src/systems/my_system/` package, containing a `my_system.py` with a skeleton class.

If you are using an AI coding agent to help build a system, point it at the repository `AGENTS.md` and `src/systems/AGENTS.md` files. They summarize the system contract, state-isolation expectations, provider-call utilities, and usage/artifact reporting conventions that generated system code should follow. We recommend being intentional about using provider-native APIs and caching capabiltiies to minimize cost for runs if using API-based models.

## The interaction lifecycle

Each run, the framework loops the following per turn:

1. `system.respond(query)` -- you produce a `Response` for the current prompt.
2. `task.step(response)` -- the task scores it and emits an `Observation` plus the next `Query`.
3. `system.observe(observation, next_query)` -- the framework hands you the feedback and (when there is one) the upcoming query, *before* the next `respond()` call.

A system sees task feedback through the `observe()` hook, not on the `Query` itself. Use `respond()` to act on the current prompt; use `observe()` to update state from what just happened. Splitting these keeps memory updates off the response-latency path.

Part of an observation is `observation.instance_complete`. The task sets this to `True` on the last step of an instance (a poker hand ending, a coding problem resolved, an issue's budget exhausted) and `False` while the system is still mid-instance.

`next_query` is the prompt that will be passed to your next `respond()` call, or `None` if the run is over. It is provided so you can prefetch, pre-warm caches, or condition memory updates on what is coming next; most systems can ignore it.

## reset()

The framework calls `reset()` at the start of a run to ensure the system starts a task instance with a completely clean slate. You must clear **all** state accumulated and learned for this system. The framework also calls `reset()` between each instance for the stateless baseline run.

## Capability flags

Set these class-level attributes to accurately describe your system:

- `supports_baseline = False` -- Set for systems that cannot be reset between instances (e.g. human evaluators). The benchmark runner skips the stateless baseline phase; gain metrics will be unavailable.
- `parallel_safe = False` -- Set if multiple concurrent instances of this system would conflict over files, ports, or shared state (e.g. a system that writes to a fixed path).

## Validation & Testing

```
clbench validate system my_system      # structural checks
clbench run exploitable_poker --schedule quick_test --system my_system  # end-to-end test
```



---

<!-- Local file: submitting.md -->

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



---

<!-- Local file: roadmap.md -->

<!-- Source: https://continual-learning-bench.com/docs/roadmap/ -->
<!-- Fetched: 2026-05-05 12:49 UTC -->

# Roadmap

Continual Learning Bench 1.0 is a starting point for a continually improving benchmark. This page outlines where we are headed.

## Tech Report

We are finalizing a **technical report** that describes the benchmark design, evaluation methodology, and initial results in detail. It will be released in the coming weeks. The report will cover task design principles, system implementations, metric definitions, and analysis of the launch results.

## Upcoming Tasks

We are actively developing new task sequences. Tasks under development include:

- **Workplace Assistant** — a preference modeling task in which a system acts as a personal assistant and must infer and adapt to the evolving preferences of a specific user over a sequence of interactions. Performance is measured by how well the system anticipates user preferences it has not been explicitly told.
- **Legal Redlining** — a task sequence in which a system reviews and redlines contracts for a specific client. Across instances, the system should learn the client's risk tolerance, preferred clause language, and firm-specific playbook to produce increasingly precise and consistent markup.

These tasks extend the benchmark into higher-stakes professional domains where personalization and memory are especially important.

## Benchmark Improvements

Beyond new tasks, we plan to improve the benchmark infrastructure and coverage:

- Add results from more systems and models (support appreciated!)
- Extend task sequences to longer horizons
- Add results from parametric continual learning methods
- Expand coverage to more specific domains and verticals — share ideas in our [Discord](https://discord.gg/7bxjNdfbfH)
- Improve simulation fidelity for environments outside of engineering domains
- Release a lightweight version of the benchmark to simplify development and iteration (current tasks are intentionally not short-horizon)
- Integrate terminal-based tasks into frameworks like [Harbor](https://www.harborframework.com/)
- Adding cloud-solution support for docker containers for systems like Claude Code and Codex

## Contributing

The fastest way to grow the benchmark is community contributions. Task contributors above a certain threshold will be authors on future papers of the benchmark we plan to submit to a top ML venue.

- To propose a new task, see [Contributing Tasks](contributing-tasks.md)
- To add a new system, see [Contributing Systems](contributing-systems.md)
- To discuss ideas, join [Discord](https://discord.gg/7bxjNdfbfH)



---

<!-- Local file: task_gallery.md -->

<!-- Source: https://continual-learning-bench.com/tasks.html -->
<!-- Fetched: 2026-05-05 12:49 UTC -->

# Task Suite (1.0)

Tasks are authored and validated by domain experts. Each task is a sequence of related instances rather than a single static problem — success requires the agent to adapt as the sequence unfolds.

| Task | Registered id | Sub-tasks | Description | Source |
|---|---:|---:|---|---|
| Database Exploration | `database_exploration` | 40 | The agent answers natural-language questions about an unknown SQLite database by issuing exploratory queries before committing to a final answer. The schema drifts across instances, requiring the agent to relearn structure over time. | [repo](https://github.com/pgasawa/continual-learning-bench/tree/main/src/tasks/database_exploration) |
| Codebase Adaptation | `codebase_adaptation` | 19 | The agent resolves a sequence of GitHub issues on a shared codebase by executing bash commands in a Docker container. Success is measured by how few steps are needed per issue — rewarding agents that accumulate reusable knowledge of the repo over time. | [repo](https://github.com/pgasawa/continual-learning-bench/tree/main/src/tasks/codebase_adaptation) |
| Sales Prediction | `sales_prediction` | 12 | The agent forecasts furniture sales across stores and time periods by writing Python analysis code in Docker. It must learn store-specific growth patterns and schema conventions from historical data, improving its models with each sequential prediction task. | [repo](https://github.com/pgasawa/continual-learning-bench/tree/main/src/tasks/sales_prediction) |
| Cohort Studies | `cohort_studies` | 20 | The agent estimates patient survival across sequential clinical studies with inconsistent variable definitions and coding conventions. It must synthesize epidemiological knowledge across schemas to improve Kaplan-Meier survival estimates for predefined population cohorts. | [repo](https://github.com/pgasawa/continual-learning-bench/tree/main/src/tasks/cohort_studies) |
| Blind Spectrum Monitoring | `blind_spectrum_monitoring` | 90 | The agent monitors RF spectrum signals to detect anomalies and identify emitters, operating with incomplete sensor data and shifting sensor configurations. It must learn persistent emitter patterns while adapting to changing array geometry across monitoring sessions. | [repo](https://github.com/pgasawa/continual-learning-bench/tree/main/src/tasks/blind_spectrum_monitoring) |
| Exploitable Poker | `exploitable_poker` | 120 | The agent plays heads-up poker against a deterministic opponent whose strategy has exploitable patterns. It must infer weaknesses from hand outcomes and adapt its betting decisions to accumulate profit over many hands. | [repo](https://github.com/pgasawa/continual-learning-bench/tree/main/src/tasks/exploitable_poker) |

> Note: the README link labelled “Task Gallery” points to `docs/tasks/`; the current site task gallery is served at `tasks.html`.



---

<!-- Local file: leaderboard.md -->

<!-- Source: https://continual-learning-bench.com/leaderboard.html -->
<!-- Data source: https://continual-learning-bench.com/leaderboard_data.json -->
<!-- Fetched: 2026-05-05 12:49 UTC -->

# Leaderboard

Full benchmark leaderboard with aggregate and per-task rows converted from the public site data.

- Last updated by site: `2026-05-04`
- Ranking note from site: systems with complete task coverage receive a rank; aggregate reward/gain are normalized across tasks for cross-task comparison.

## Metric glossary from the site

- **Reward ↑** — Raw task performance score. Higher is better.
- **Gain ↑** — Reward minus the same system’s stateless baseline; direct measure of how much the system learned from experience. Higher is better.
- **Agg. Reward / Gain ↑** — Each task’s reward or gain normalized against a reference ceiling and fixed/corresponding stateless baseline, then averaged across tasks. Primary ranking metric.
- **Cost ↓** — Aggregate table: sum of each included task’s mean single rollout spend. Task table: mean spend per single task rollout. Lower is better.

## Aggregate ranking

| Rank | System | Run name | System class | Model | Tasks | Norm reward | Norm gain | Total cost | Mean task cost |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | ICL · Claude Sonnet 4.6 | `icl-claude-sonnet-4.6` | `icl` | Claude Sonnet 4.6 | 6 | 0.223 | 0.254 | $30.43 | $5.07 |
| 2 | ICL · GPT-5.4 | `icl-gpt-5.4` | `icl` | GPT-5.4 | 6 | 0.201 | 0.201 | $18.39 | $3.06 |
| 3 | Claude Code · Sonnet 4.6 | `claude-code-sonnet-4.6` | `claude-code` | Sonnet 4.6 | 6 | 0.190 | 0.239 | $38.60 | $6.43 |
| 4 | Mem0 · GPT-5.4 | `mem0-gpt-5.4` | `mem0` | GPT-5.4 | 6 | 0.151 | 0.202 | $18.34 | $3.06 |
| 5 | ICL · Claude Opus 4.7 | `icl-claude-opus-4.7` | `icl` | Claude Opus 4.7 | 6 | 0.102 | 0.195 | $49.62 | $8.27 |
| 6 | ICL Notepad · GPT-5.4 | `icl-notepad-gpt-5.4` | `icl-notepad` | GPT-5.4 | 6 | 0.080 | 0.078 | $14.28 | $2.38 |
| 7 | ICL · Gemini 3 Flash | `icl-gemini-3-flash` | `icl` | Gemini 3 Flash | 6 | 0.080 | 0.164 | $7.60 | $1.27 |
| 8 | Codex · GPT-5.4 | `codex-gpt-5.4` | `codex` | GPT-5.4 | 6 | 0.066 | 0.146 | $27.21 | $4.53 |
| 9 | ACE · GPT-5.4 | `ace-gpt-5.4` | `ace` | GPT-5.4 | 6 | 0.046 | 0.086 | $62.75 | $10.46 |
| 10 | ICL Notepad · Claude Sonnet 4.6 | `icl-notepad-claude-sonnet-4-6` | `icl-notepad` | Claude Sonnet 4.6 | 6 | 0.035 | 0.182 | $31.53 | $5.25 |
| 11 | ICL Notepad · Gemini 3.1 Pro Preview | `icl-notepad-gemini-3.1-pro-preview` | `icl-notepad` | Gemini 3.1 Pro Preview | 6 | -0.002 | 0.094 | $13.32 | $2.22 |
| 12 | ICL · Gemini 3.1 Pro Preview | `icl-gemini-3.1-pro-preview` | `icl` | Gemini 3.1 Pro Preview | 6 | -0.056 | 0.062 | $15.23 | $2.54 |

## Per-task breakdown

| Task | System | Runs | Instances | Reward | Baseline | Reference | Gain | Norm reward | Norm gain | Cost | Latency s | Warnings |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| blind_spectrum_monitoring | `ace-gpt-5.4` | 5 | 90 | 19.778 | 19.761 | 90.000 | 0.017 | 0.000 | 0.000 | $3.96 | 3.99 |  |
| blind_spectrum_monitoring | `claude-code-sonnet-4.6` | 5 | 90 | 44.282 | 19.760 | 90.000 | 24.522 | 0.349 | 0.349 | $10.40 | 23.93 |  |
| blind_spectrum_monitoring | `codex-gpt-5.4` | 1 | 90 | 32.828 | 19.760 | 90.000 | 13.068 | 0.186 | 0.186 | $3.15 | 11.08 |  |
| blind_spectrum_monitoring | `icl-claude-opus-4.7` | 5 | 90 | 33.572 | 19.760 | 90.000 | 13.813 | 0.197 | 0.197 | $7.58 | 13.07 |  |
| blind_spectrum_monitoring | `icl-claude-sonnet-4.6` | 5 | 90 | 36.584 | 19.760 | 90.000 | 16.825 | 0.240 | 0.240 | $3.60 | 9.84 |  |
| blind_spectrum_monitoring | `icl-gemini-3-flash` | 5 | 90 | 33.039 | 19.760 | 90.000 | 13.279 | 0.189 | 0.189 | $0.68 | 3.98 |  |
| blind_spectrum_monitoring | `icl-gemini-3.1-pro-preview` | 5 | 90 | 33.033 | 19.760 | 90.000 | 13.273 | 0.189 | 0.189 | $3.84 | 15.71 |  |
| blind_spectrum_monitoring | `icl-gpt-5.4` | 5 | 90 | 46.198 | 19.761 | 90.000 | 26.437 | 0.376 | 0.376 | $1.93 | 6.14 |  |
| blind_spectrum_monitoring | `icl-notepad-claude-sonnet-4-6` | 5 | 90 | 35.993 | 19.760 | 90.000 | 16.233 | 0.231 | 0.231 | $2.99 | 18.33 |  |
| blind_spectrum_monitoring | `icl-notepad-gemini-3.1-pro-preview` | 5 | 90 | 29.122 | 19.760 | 90.000 | 9.362 | 0.133 | 0.133 | $2.80 | 18.12 |  |
| blind_spectrum_monitoring | `icl-notepad-gpt-5.4` | 5 | 90 | 31.915 | 19.761 | 90.000 | 12.153 | 0.173 | 0.173 | $1.02 | 7.36 |  |
| blind_spectrum_monitoring | `mem0-gpt-5.4` | 5 | 90 | 33.794 | 19.760 | 90.000 | 14.033 | 0.200 | 0.200 | $1.39 | 5.74 |  |
| codebase_adaptation | `ace-gpt-5.4` | 5 | 19 | 11.580 | 10.000 | 19.000 | 1.580 | 0.223 | 0.176 | $15.66 | 3.82 |  |
| codebase_adaptation | `claude-code-sonnet-4.6` | 5 | 19 | 6.630 | 5.900 | 19.000 | 0.730 | -0.295 | 0.056 | $6.78 | 15.55 |  |
| codebase_adaptation | `codex-gpt-5.4` | 1 | 19 | 7.450 | 8.250 | 19.000 | -0.800 | -0.209 | -0.074 | $3.79 | 6.21 |  |
| codebase_adaptation | `icl-claude-opus-4.7` | 5 | 19 | 10.360 | 8.875 | 19.000 | 1.485 | 0.095 | 0.147 | $8.03 | 4.74 |  |
| codebase_adaptation | `icl-claude-sonnet-4.6` | 5 | 19 | 9.755 | 7.050 | 19.000 | 2.705 | 0.032 | 0.226 | $6.92 | 5.20 |  |
| codebase_adaptation | `icl-gemini-3-flash` | 5 | 19 | 7.420 | 7.750 | 19.000 | -0.330 | -0.213 | -0.029 | $2.75 | 6.54 |  |
| codebase_adaptation | `icl-gemini-3.1-pro-preview` | 5 | 19 | 5.125 | 6.675 | 19.000 | -1.550 | -0.453 | -0.126 | $3.40 | 4.62 |  |
| codebase_adaptation | `icl-gpt-5.4` | 5 | 19 | 10.140 | 9.450 | 19.000 | 0.690 | 0.072 | 0.072 | $3.56 | 5.98 |  |
| codebase_adaptation | `icl-notepad-claude-sonnet-4-6` | 5 | 19 | 8.765 | 7.875 | 19.000 | 0.890 | -0.072 | 0.080 | $4.24 | 5.84 |  |
| codebase_adaptation | `icl-notepad-gemini-3.1-pro-preview` | 5 | 19 | 7.110 | 8.425 | 19.000 | -1.315 | -0.245 | -0.124 | $3.51 | 4.29 |  |
| codebase_adaptation | `icl-notepad-gpt-5.4` | 5 | 19 | 8.860 | 8.250 | 19.000 | 0.610 | -0.062 | 0.057 | $3.49 | 9.57 |  |
| codebase_adaptation | `mem0-gpt-5.4` | 5 | 19 | 11.105 | 8.125 | 19.000 | 2.980 | 0.173 | 0.274 | $2.62 | 5.18 |  |
| cohort_studies | `ace-gpt-5.4` | 5 | 20 | 0.759 | 0.376 | 3.244 | 0.383 | -0.105 | 0.134 | $12.80 | 6.33 |  |
| cohort_studies | `claude-code-sonnet-4.6` | 5 | 20 | 0.496 | 0.805 | 3.244 | -0.309 | -0.222 | -0.126 | $7.21 | 25.14 |  |
| cohort_studies | `codex-gpt-5.4` | 1 | 20 | 0.821 | 0.505 | 3.244 | 0.316 | -0.077 | 0.115 | $7.76 | 13.10 |  |
| cohort_studies | `icl-claude-opus-4.7` | 5 | 20 | -0.121 | -0.030 | 3.244 | -0.091 | -0.496 | -0.028 | $6.97 | 8.08 |  |
| cohort_studies | `icl-claude-sonnet-4.6` | 5 | 20 | 0.762 | 0.576 | 3.244 | 0.185 | -0.104 | 0.070 | $5.62 | 15.92 |  |
| cohort_studies | `icl-gemini-3-flash` | 5 | 20 | 0.576 | 0.140 | 3.244 | 0.437 | -0.186 | 0.141 | $1.25 | 4.51 |  |
| cohort_studies | `icl-gemini-3.1-pro-preview` | 5 | 20 | 0.257 | 0.820 | 3.244 | -0.563 | -0.328 | -0.232 | $1.83 | 9.96 |  |
| cohort_studies | `icl-gpt-5.4` | 5 | 20 | 0.957 | 0.994 | 3.244 | -0.037 | -0.017 | -0.017 | $3.73 | 11.22 |  |
| cohort_studies | `icl-notepad-claude-sonnet-4-6` | 5 | 20 | -0.784 | -1.579 | 3.244 | 0.795 | -0.791 | 0.165 | $11.56 | 24.32 |  |
| cohort_studies | `icl-notepad-gemini-3.1-pro-preview` | 5 | 20 | 0.327 | 0.692 | 3.244 | -0.366 | -0.297 | -0.143 | $1.53 | 6.45 |  |
| cohort_studies | `icl-notepad-gpt-5.4` | 5 | 20 | 0.476 | 1.376 | 3.244 | -0.900 | -0.231 | -0.482 | $4.44 | 10.54 |  |
| cohort_studies | `mem0-gpt-5.4` | 5 | 20 | 0.778 | 0.845 | 3.244 | -0.067 | -0.096 | -0.028 | $6.00 | 9.08 |  |
| database_exploration | `ace-gpt-5.4` | 5 | 40 | 7.853 | 5.467 | 40.000 | 2.387 | 0.067 | 0.069 | $8.78 | 3.70 |  |
| database_exploration | `claude-code-sonnet-4.6` | 5 | 40 | 22.053 | 8.200 | 40.000 | 13.853 | 0.479 | 0.436 | $3.34 | 5.79 |  |
| database_exploration | `codex-gpt-5.4` | 1 | 40 | 9.600 | 3.467 | 40.000 | 6.133 | 0.118 | 0.168 | $1.85 | 4.74 |  |
| database_exploration | `icl-claude-opus-4.7` | 5 | 40 | 15.653 | 6.067 | 40.000 | 9.587 | 0.294 | 0.283 | $5.22 | 4.47 |  |
| database_exploration | `icl-claude-sonnet-4.6` | 5 | 40 | 15.013 | 6.533 | 40.000 | 8.480 | 0.275 | 0.253 | $1.95 | 2.03 |  |
| database_exploration | `icl-gemini-3-flash` | 5 | 40 | 15.027 | 3.533 | 40.000 | 11.493 | 0.275 | 0.315 | $0.42 | 1.47 |  |
| database_exploration | `icl-gemini-3.1-pro-preview` | 5 | 40 | 11.560 | 4.733 | 40.000 | 6.827 | 0.175 | 0.194 | $1.32 | 6.08 |  |
| database_exploration | `icl-gpt-5.4` | 5 | 40 | 13.880 | 5.533 | 40.000 | 8.347 | 0.242 | 0.242 | $1.03 | 3.77 |  |
| database_exploration | `icl-notepad-claude-sonnet-4-6` | 5 | 40 | 11.000 | 7.333 | 40.000 | 3.667 | 0.159 | 0.112 | $2.24 | 2.37 |  |
| database_exploration | `icl-notepad-gemini-3.1-pro-preview` | 5 | 40 | 8.520 | 4.200 | 40.000 | 4.320 | 0.087 | 0.121 | $2.63 | 4.50 |  |
| database_exploration | `icl-notepad-gpt-5.4` | 5 | 40 | 12.373 | 6.000 | 40.000 | 6.373 | 0.198 | 0.187 | $1.45 | 3.54 |  |
| database_exploration | `mem0-gpt-5.4` | 5 | 40 | 17.240 | 4.333 | 40.000 | 12.907 | 0.340 | 0.362 | $1.97 | 3.02 |  |
| exploitable_poker | `ace-gpt-5.4` | 5 | 120 | 143.500 | 141.900 | 1138.500 | 1.600 | 0.010 | 0.002 | $13.20 | 4.85 |  |
| exploitable_poker | `claude-code-sonnet-4.6` | 5 | 120 | 343.020 | 284.500 | 1138.500 | 58.520 | 0.208 | 0.069 | $8.65 | 6.61 |  |
| exploitable_poker | `codex-gpt-5.4` | 1 | 120 | 85.000 | 64.500 | 1138.500 | 20.500 | -0.048 | 0.019 | $8.27 | 5.88 |  |
| exploitable_poker | `icl-claude-opus-4.7` | 5 | 120 | 116.680 | 157.700 | 1138.500 | -41.020 | -0.017 | -0.042 | $17.47 | 4.95 |  |
| exploitable_poker | `icl-claude-sonnet-4.6` | 5 | 120 | 339.920 | 316.700 | 1138.500 | 23.220 | 0.205 | 0.028 | $9.39 | 8.43 |  |
| exploitable_poker | `icl-gemini-3-flash` | 5 | 120 | 94.840 | 196.800 | 1138.500 | -101.960 | -0.039 | -0.108 | $1.91 | 2.44 |  |
| exploitable_poker | `icl-gemini-3.1-pro-preview` | 5 | 120 | 76.400 | 43.500 | 1138.500 | 32.900 | -0.057 | 0.030 | $3.99 | 6.49 |  |
| exploitable_poker | `icl-gpt-5.4` | 5 | 120 | 95.760 | 133.600 | 1138.500 | -37.840 | -0.038 | -0.038 | $4.61 | 6.45 |  |
| exploitable_poker | `icl-notepad-claude-sonnet-4-6` | 5 | 120 | 115.420 | 317.000 | 1138.500 | -201.580 | -0.018 | -0.245 | $6.75 | 17.05 |  |
| exploitable_poker | `icl-notepad-gemini-3.1-pro-preview` | 5 | 120 | 53.500 | 36.500 | 1138.500 | 17.000 | -0.080 | 0.015 | $1.33 | 5.35 |  |
| exploitable_poker | `icl-notepad-gpt-5.4` | 5 | 120 | 81.340 | 81.100 | 1138.500 | 0.240 | -0.052 | 0.000 | $1.52 | 4.36 |  |
| exploitable_poker | `mem0-gpt-5.4` | 5 | 120 | 73.420 | 90.400 | 1138.500 | -16.980 | -0.060 | -0.016 | $3.57 | 4.67 |  |
| sales_prediction | `ace-gpt-5.4` | 5 | 12 | 6.116 | 5.203 | 12.000 | 0.913 | 0.081 | 0.134 | $8.36 | 7.27 |  |
| sales_prediction | `claude-code-sonnet-4.6` | 5 | 12 | 9.573 | 5.036 | 12.000 | 4.537 | 0.621 | 0.651 | $2.21 | 12.15 |  |
| sales_prediction | `codex-gpt-5.4` | 5 | 12 | 8.320 | 5.176 | 12.000 | 3.144 | 0.425 | 0.461 | $2.38 | 11.03 |  |
| sales_prediction | `icl-claude-opus-4.7` | 5 | 12 | 9.039 | 4.390 | 12.000 | 4.649 | 0.538 | 0.611 | $4.36 | 9.55 |  |
| sales_prediction | `icl-claude-sonnet-4.6` | 5 | 12 | 10.018 | 5.178 | 12.000 | 4.840 | 0.690 | 0.709 | $2.96 | 15.67 |  |
| sales_prediction | `icl-gemini-3-flash` | 5 | 12 | 8.481 | 5.271 | 12.000 | 3.210 | 0.450 | 0.477 | $0.59 | 3.57 |  |
| sales_prediction | `icl-gemini-3.1-pro-preview` | 5 | 12 | 6.468 | 3.908 | 12.000 | 2.560 | 0.136 | 0.316 | $0.85 | 9.54 |  |
| sales_prediction | `icl-gpt-5.4` | 5 | 12 | 9.230 | 5.597 | 12.000 | 3.633 | 0.567 | 0.567 | $3.52 | 11.34 |  |
| sales_prediction | `icl-notepad-claude-sonnet-4-6` | 5 | 12 | 10.073 | 4.282 | 12.000 | 5.790 | 0.699 | 0.750 | $3.75 | 18.62 |  |
| sales_prediction | `icl-notepad-gemini-3.1-pro-preview` | 5 | 12 | 8.108 | 3.156 | 12.000 | 4.952 | 0.392 | 0.560 | $1.53 | 9.11 |  |
| sales_prediction | `icl-notepad-gpt-5.4` | 5 | 12 | 8.486 | 4.483 | 12.000 | 4.003 | 0.451 | 0.532 | $2.35 | 13.43 |  |
| sales_prediction | `mem0-gpt-5.4` | 5 | 12 | 7.815 | 4.750 | 12.000 | 3.064 | 0.346 | 0.423 | $2.79 | 10.10 |  |

## Normalization metadata

```json
{
  "gain": "gain / (reference_max - stateless baseline of that system)",
  "reference_max": "Task-defined mean per-instance maximum reward from each task class's r_max.",
  "reward": "(reward - stateless baseline of gpt-5.4) / (reference_max - stateless baseline of gpt-5.4)"
}
```

## Detailed run points summary

The public JSON also contains per-run learning curves. To keep this markdown readable, the table below includes run-level scalar fields and omits the curve arrays.

| Run name | Task | Run index | Reward | Baseline | Reference | Gain | Norm reward | Norm gain | Cost | Latency s | Instance count |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ace-gpt-5.4` | blind_spectrum_monitoring | 0 | 19.811 | 19.761 | 90.000 | 0.050 | 0.001 | 0.001 | $3.75 | 4.09 | 90 |
| `ace-gpt-5.4` | blind_spectrum_monitoring | 1 | 19.776 | 19.761 | 90.000 | 0.015 | 0.000 | 0.000 | $4.10 | 3.99 | 90 |
| `ace-gpt-5.4` | blind_spectrum_monitoring | 2 | 19.783 | 19.761 | 90.000 | 0.022 | 0.000 | 0.000 | $4.73 | 4.09 | 90 |
| `ace-gpt-5.4` | blind_spectrum_monitoring | 3 | 19.760 | 19.761 | 90.000 | -0.001 | -0.000 | -0.000 | $2.81 | 3.71 | 90 |
| `ace-gpt-5.4` | blind_spectrum_monitoring | 4 | 19.760 | 19.761 | 90.000 | -0.001 | -0.000 | -0.000 | $4.38 | 4.06 | 90 |
| `claude-code-sonnet-4.6` | blind_spectrum_monitoring | 0 | 39.610 | 19.760 | 90.000 | 19.850 | 0.283 | 0.283 | $4.82 | 13.41 | 90 |
| `claude-code-sonnet-4.6` | blind_spectrum_monitoring | 1 | 44.483 | 19.760 | 90.000 | 24.723 | 0.352 | 0.352 | $14.02 | 34.94 | 90 |
| `claude-code-sonnet-4.6` | blind_spectrum_monitoring | 2 | 44.933 | 19.760 | 90.000 | 25.173 | 0.358 | 0.358 | $5.02 | 13.44 | 90 |
| `claude-code-sonnet-4.6` | blind_spectrum_monitoring | 3 | 43.703 | 19.760 | 90.000 | 23.943 | 0.341 | 0.341 | $12.87 | 27.49 | 90 |
| `claude-code-sonnet-4.6` | blind_spectrum_monitoring | 4 | 48.682 | 19.760 | 90.000 | 28.922 | 0.412 | 0.412 | $15.25 | 30.36 | 90 |
| `codex-gpt-5.4` | blind_spectrum_monitoring | 0 | 32.828 | 19.760 | 90.000 | 13.068 | 0.186 | 0.186 | $3.15 | 11.08 | 90 |
| `icl-claude-opus-4.7` | blind_spectrum_monitoring | 0 | 35.165 | 19.760 | 90.000 | 15.405 | 0.219 | 0.219 | $8.73 | 16.06 | 90 |
| `icl-claude-opus-4.7` | blind_spectrum_monitoring | 1 | 28.625 | 19.760 | 90.000 | 8.865 | 0.126 | 0.126 | $7.54 | 12.57 | 90 |
| `icl-claude-opus-4.7` | blind_spectrum_monitoring | 2 | 39.492 | 19.760 | 90.000 | 19.732 | 0.281 | 0.281 | $6.66 | 11.15 | 90 |
| `icl-claude-opus-4.7` | blind_spectrum_monitoring | 3 | 40.183 | 19.760 | 90.000 | 20.423 | 0.291 | 0.291 | $6.65 | 11.03 | 90 |
| `icl-claude-opus-4.7` | blind_spectrum_monitoring | 4 | 24.397 | 19.760 | 90.000 | 4.637 | 0.066 | 0.066 | $8.30 | 14.52 | 90 |
| `icl-claude-sonnet-4.6` | blind_spectrum_monitoring | 0 | 37.505 | 19.760 | 90.000 | 17.746 | 0.253 | 0.253 | $3.70 | 9.84 | 90 |
| `icl-claude-sonnet-4.6` | blind_spectrum_monitoring | 1 | 36.330 | 19.760 | 90.000 | 16.570 | 0.236 | 0.236 | $3.40 | 9.27 | 90 |
| `icl-claude-sonnet-4.6` | blind_spectrum_monitoring | 2 | 37.136 | 19.760 | 90.000 | 17.376 | 0.247 | 0.247 | $3.13 | 7.74 | 90 |
| `icl-claude-sonnet-4.6` | blind_spectrum_monitoring | 3 | 32.108 | 19.760 | 90.000 | 12.349 | 0.176 | 0.176 | $3.62 | 10.37 | 90 |
| `icl-claude-sonnet-4.6` | blind_spectrum_monitoring | 4 | 39.842 | 19.760 | 90.000 | 20.082 | 0.286 | 0.286 | $4.13 | 12.00 | 90 |
| `icl-gemini-3-flash` | blind_spectrum_monitoring | 0 | 35.371 | 19.760 | 90.000 | 15.611 | 0.222 | 0.222 | $0.73 | 4.42 | 90 |
| `icl-gemini-3-flash` | blind_spectrum_monitoring | 1 | 32.267 | 19.760 | 90.000 | 12.507 | 0.178 | 0.178 | $0.67 | 4.18 | 90 |
| `icl-gemini-3-flash` | blind_spectrum_monitoring | 2 | 31.509 | 19.760 | 90.000 | 11.750 | 0.167 | 0.167 | $0.68 | 3.64 | 90 |
| `icl-gemini-3-flash` | blind_spectrum_monitoring | 3 | 34.907 | 19.760 | 90.000 | 15.147 | 0.216 | 0.216 | $0.60 | 3.50 | 90 |
| `icl-gemini-3-flash` | blind_spectrum_monitoring | 4 | 31.141 | 19.760 | 90.000 | 11.381 | 0.162 | 0.162 | $0.72 | 4.17 | 90 |
| `icl-gemini-3.1-pro-preview` | blind_spectrum_monitoring | 0 | 37.086 | 19.760 | 90.000 | 17.326 | 0.247 | 0.247 | $3.44 | 13.69 | 90 |
| `icl-gemini-3.1-pro-preview` | blind_spectrum_monitoring | 1 | 30.058 | 19.760 | 90.000 | 10.298 | 0.147 | 0.147 | $4.02 | 16.56 | 90 |
| `icl-gemini-3.1-pro-preview` | blind_spectrum_monitoring | 2 | 32.506 | 19.760 | 90.000 | 12.747 | 0.181 | 0.181 | $3.59 | 12.40 | 90 |
| `icl-gemini-3.1-pro-preview` | blind_spectrum_monitoring | 3 | 32.980 | 19.760 | 90.000 | 13.220 | 0.188 | 0.188 | $3.72 | 15.23 | 90 |
| `icl-gemini-3.1-pro-preview` | blind_spectrum_monitoring | 4 | 32.532 | 19.760 | 90.000 | 12.772 | 0.182 | 0.182 | $4.40 | 20.69 | 90 |
| `icl-gpt-5.4` | blind_spectrum_monitoring | 0 | 49.695 | 19.761 | 90.000 | 29.934 | 0.426 | 0.426 | $2.02 | 6.38 | 90 |
| `icl-gpt-5.4` | blind_spectrum_monitoring | 1 | 47.164 | 19.761 | 90.000 | 27.403 | 0.390 | 0.390 | $2.00 | 6.06 | 90 |
| `icl-gpt-5.4` | blind_spectrum_monitoring | 2 | 44.404 | 19.761 | 90.000 | 24.643 | 0.351 | 0.351 | $1.78 | 5.84 | 90 |
| `icl-gpt-5.4` | blind_spectrum_monitoring | 3 | 45.089 | 19.761 | 90.000 | 25.328 | 0.361 | 0.361 | $1.99 | 6.31 | 90 |
| `icl-gpt-5.4` | blind_spectrum_monitoring | 4 | 44.637 | 19.761 | 90.000 | 24.876 | 0.354 | 0.354 | $1.88 | 6.10 | 90 |
| `icl-notepad-claude-sonnet-4-6` | blind_spectrum_monitoring | 0 | 36.496 | 19.760 | 90.000 | 16.737 | 0.238 | 0.238 | $2.87 | 18.75 | 90 |
| `icl-notepad-claude-sonnet-4-6` | blind_spectrum_monitoring | 1 | 32.805 | 19.760 | 90.000 | 13.045 | 0.186 | 0.186 | $2.49 | 14.36 | 90 |
| `icl-notepad-claude-sonnet-4-6` | blind_spectrum_monitoring | 2 | 28.931 | 19.760 | 90.000 | 9.172 | 0.131 | 0.131 | $2.39 | 13.11 | 90 |
| `icl-notepad-claude-sonnet-4-6` | blind_spectrum_monitoring | 3 | 38.776 | 19.760 | 90.000 | 19.016 | 0.271 | 0.271 | $3.33 | 21.97 | 90 |
| `icl-notepad-claude-sonnet-4-6` | blind_spectrum_monitoring | 4 | 42.957 | 19.760 | 90.000 | 23.197 | 0.330 | 0.330 | $3.85 | 23.46 | 90 |
| `icl-notepad-gemini-3.1-pro-preview` | blind_spectrum_monitoring | 0 | 38.428 | 19.760 | 90.000 | 18.669 | 0.266 | 0.266 | $1.74 | 11.85 | 90 |
| `icl-notepad-gemini-3.1-pro-preview` | blind_spectrum_monitoring | 1 | 28.529 | 19.760 | 90.000 | 8.770 | 0.125 | 0.125 | $2.29 | 15.26 | 90 |
| `icl-notepad-gemini-3.1-pro-preview` | blind_spectrum_monitoring | 2 | 21.444 | 19.760 | 90.000 | 1.684 | 0.024 | 0.024 | $2.75 | 18.27 | 90 |
| `icl-notepad-gemini-3.1-pro-preview` | blind_spectrum_monitoring | 3 | 32.779 | 19.760 | 90.000 | 13.019 | 0.185 | 0.185 | $2.41 | 15.77 | 90 |
| `icl-notepad-gemini-3.1-pro-preview` | blind_spectrum_monitoring | 4 | 24.428 | 19.760 | 90.000 | 4.669 | 0.066 | 0.066 | $4.80 | 29.46 | 90 |
| `icl-notepad-gpt-5.4` | blind_spectrum_monitoring | 0 | 35.235 | 19.761 | 90.000 | 15.474 | 0.220 | 0.220 | $1.07 | 7.45 | 90 |
| `icl-notepad-gpt-5.4` | blind_spectrum_monitoring | 1 | 34.245 | 19.761 | 90.000 | 14.484 | 0.206 | 0.206 | $1.20 | 8.23 | 90 |
| `icl-notepad-gpt-5.4` | blind_spectrum_monitoring | 2 | 29.613 | 19.761 | 90.000 | 9.852 | 0.140 | 0.140 | $0.91 | 6.68 | 90 |
| `icl-notepad-gpt-5.4` | blind_spectrum_monitoring | 3 | 24.642 | 19.761 | 90.000 | 4.881 | 0.069 | 0.069 | $0.91 | 6.98 | 90 |
| `icl-notepad-gpt-5.4` | blind_spectrum_monitoring | 4 | 35.838 | 19.761 | 90.000 | 16.077 | 0.229 | 0.229 | $1.03 | 7.44 | 90 |
| `mem0-gpt-5.4` | blind_spectrum_monitoring | 0 | 37.087 | 19.760 | 90.000 | 17.326 | 0.247 | 0.247 | $1.15 | 5.34 | 90 |
| `mem0-gpt-5.4` | blind_spectrum_monitoring | 1 | 24.384 | 19.760 | 90.000 | 4.624 | 0.066 | 0.066 | $1.30 | 6.72 | 90 |
| `mem0-gpt-5.4` | blind_spectrum_monitoring | 2 | 41.496 | 19.760 | 90.000 | 21.736 | 0.309 | 0.309 | $1.45 | 5.77 | 90 |
| `mem0-gpt-5.4` | blind_spectrum_monitoring | 3 | 29.969 | 19.760 | 90.000 | 10.209 | 0.145 | 0.145 | $1.45 | 5.42 | 90 |
| `mem0-gpt-5.4` | blind_spectrum_monitoring | 4 | 36.031 | 19.760 | 90.000 | 16.271 | 0.232 | 0.232 | $1.59 | 5.46 | 90 |
| `ace-gpt-5.4` | codebase_adaptation | 0 | 11.450 | 10.000 | 19.000 | 1.450 | 0.209 | 0.161 | $18.20 | 3.72 | 19 |
| `ace-gpt-5.4` | codebase_adaptation | 1 | 11.750 | 10.000 | 19.000 | 1.750 | 0.241 | 0.194 | $15.96 | 3.85 | 19 |
| `ace-gpt-5.4` | codebase_adaptation | 2 | 12.200 | 10.000 | 19.000 | 2.200 | 0.288 | 0.244 | $13.33 | 3.81 | 19 |
| `ace-gpt-5.4` | codebase_adaptation | 3 | 11.200 | 10.000 | 19.000 | 1.200 | 0.183 | 0.133 | $15.82 | 3.90 | 19 |
| `ace-gpt-5.4` | codebase_adaptation | 4 | 11.300 | 10.000 | 19.000 | 1.300 | 0.194 | 0.144 | $14.98 | 3.84 | 19 |
| `claude-code-sonnet-4.6` | codebase_adaptation | 0 | 9.350 | 5.900 | 19.000 | 3.450 | -0.010 | 0.263 | $6.88 | 5.88 | 19 |
| `claude-code-sonnet-4.6` | codebase_adaptation | 1 | 9.075 | 5.900 | 19.000 | 3.175 | -0.039 | 0.242 | $8.29 | 5.86 | 19 |
| `claude-code-sonnet-4.6` | codebase_adaptation | 2 | 8.625 | 5.900 | 19.000 | 2.725 | -0.086 | 0.208 | $6.66 | 5.35 | 19 |
| `claude-code-sonnet-4.6` | codebase_adaptation | 3 | 0.000 | 5.900 | 19.000 | -5.900 | -0.990 | -0.450 | $5.66 | 54.76 | 19 |
| `claude-code-sonnet-4.6` | codebase_adaptation | 4 | 6.100 | 5.900 | 19.000 | 0.200 | -0.351 | 0.015 | $6.41 | 5.87 | 19 |
| `codex-gpt-5.4` | codebase_adaptation | 0 | 7.450 | 8.250 | 19.000 | -0.800 | -0.209 | -0.074 | $3.79 | 6.21 | 19 |
| `icl-claude-opus-4.7` | codebase_adaptation | 0 | 11.200 | 8.875 | 19.000 | 2.325 | 0.183 | 0.230 | $8.40 | 5.10 | 19 |
| `icl-claude-opus-4.7` | codebase_adaptation | 1 | 9.550 | 8.875 | 19.000 | 0.675 | 0.010 | 0.067 | $8.67 | 4.83 | 19 |
| `icl-claude-opus-4.7` | codebase_adaptation | 2 | 7.175 | 8.875 | 19.000 | -1.700 | -0.238 | -0.168 | $7.51 | 4.38 | 19 |
| `icl-claude-opus-4.7` | codebase_adaptation | 3 | 11.400 | 8.875 | 19.000 | 2.525 | 0.204 | 0.249 | $12.30 | 5.17 | 19 |
| `icl-claude-opus-4.7` | codebase_adaptation | 4 | 12.475 | 8.875 | 19.000 | 3.600 | 0.317 | 0.356 | $3.29 | 4.22 | 19 |
| `icl-claude-sonnet-4.6` | codebase_adaptation | 0 | 9.250 | 7.050 | 19.000 | 2.200 | -0.021 | 0.184 | $4.30 | 4.81 | 19 |
| `icl-claude-sonnet-4.6` | codebase_adaptation | 1 | 10.425 | 7.050 | 19.000 | 3.375 | 0.102 | 0.282 | $7.04 | 5.38 | 19 |
| `icl-claude-sonnet-4.6` | codebase_adaptation | 2 | 9.700 | 7.050 | 19.000 | 2.650 | 0.026 | 0.222 | $6.88 | 4.94 | 19 |
| `icl-claude-sonnet-4.6` | codebase_adaptation | 3 | 9.325 | 7.050 | 19.000 | 2.275 | -0.013 | 0.190 | $7.53 | 5.48 | 19 |
| `icl-claude-sonnet-4.6` | codebase_adaptation | 4 | 10.075 | 7.050 | 19.000 | 3.025 | 0.065 | 0.253 | $8.85 | 5.40 | 19 |
| `icl-gemini-3-flash` | codebase_adaptation | 0 | 6.175 | 7.750 | 19.000 | -1.575 | -0.343 | -0.140 | $2.92 | 6.22 | 19 |
| `icl-gemini-3-flash` | codebase_adaptation | 1 | 4.850 | 7.750 | 19.000 | -2.900 | -0.482 | -0.258 | $1.59 | 3.84 | 19 |
| `icl-gemini-3-flash` | codebase_adaptation | 2 | 7.000 | 7.750 | 19.000 | -0.750 | -0.257 | -0.067 | $3.17 | 7.41 | 19 |
| `icl-gemini-3-flash` | codebase_adaptation | 3 | 9.800 | 7.750 | 19.000 | 2.050 | 0.037 | 0.182 | $2.42 | 5.67 | 19 |
| `icl-gemini-3-flash` | codebase_adaptation | 4 | 9.275 | 7.750 | 19.000 | 1.525 | -0.018 | 0.136 | $3.67 | 9.55 | 19 |
| `icl-gemini-3.1-pro-preview` | codebase_adaptation | 0 | 5.350 | 6.675 | 19.000 | -1.325 | -0.429 | -0.108 | $5.92 | 4.90 | 19 |
| `icl-gemini-3.1-pro-preview` | codebase_adaptation | 1 | 5.275 | 6.675 | 19.000 | -1.400 | -0.437 | -0.114 | $1.41 | 4.54 | 19 |
| `icl-gemini-3.1-pro-preview` | codebase_adaptation | 2 | 5.425 | 6.675 | 19.000 | -1.250 | -0.421 | -0.101 | $5.11 | 5.34 | 19 |
| `icl-gemini-3.1-pro-preview` | codebase_adaptation | 3 | 2.775 | 6.675 | 19.000 | -3.900 | -0.699 | -0.316 | $1.15 | 4.07 | 19 |
| `icl-gemini-3.1-pro-preview` | codebase_adaptation | 4 | 6.800 | 6.675 | 19.000 | 0.125 | -0.277 | 0.010 | $3.42 | 4.24 | 19 |
| `icl-gpt-5.4` | codebase_adaptation | 0 | 6.175 | 9.450 | 19.000 | -3.275 | -0.343 | -0.343 | $2.09 | 5.10 | 19 |
| `icl-gpt-5.4` | codebase_adaptation | 1 | 14.100 | 9.450 | 19.000 | 4.650 | 0.487 | 0.487 | $6.10 | 6.49 | 19 |
| `icl-gpt-5.4` | codebase_adaptation | 2 | 4.925 | 9.450 | 19.000 | -4.525 | -0.474 | -0.474 | $3.08 | 5.89 | 19 |
| `icl-gpt-5.4` | codebase_adaptation | 3 | 11.250 | 9.450 | 19.000 | 1.800 | 0.188 | 0.188 | $3.84 | 6.62 | 19 |
| `icl-gpt-5.4` | codebase_adaptation | 4 | 14.250 | 9.450 | 19.000 | 4.800 | 0.503 | 0.503 | $2.69 | 5.80 | 19 |
| `icl-notepad-claude-sonnet-4-6` | codebase_adaptation | 0 | 9.800 | 7.875 | 19.000 | 1.925 | 0.037 | 0.173 | $3.81 | 5.14 | 19 |
| `icl-notepad-claude-sonnet-4-6` | codebase_adaptation | 1 | 8.450 | 7.875 | 19.000 | 0.575 | -0.105 | 0.052 | $6.34 | 10.17 | 19 |
| `icl-notepad-claude-sonnet-4-6` | codebase_adaptation | 2 | 7.275 | 7.875 | 19.000 | -0.600 | -0.228 | -0.054 | $3.93 | 4.38 | 19 |
| `icl-notepad-claude-sonnet-4-6` | codebase_adaptation | 3 | 9.625 | 7.875 | 19.000 | 1.750 | 0.018 | 0.157 | $3.56 | 4.39 | 19 |
| `icl-notepad-claude-sonnet-4-6` | codebase_adaptation | 4 | 8.675 | 7.875 | 19.000 | 0.800 | -0.081 | 0.072 | $3.57 | 5.12 | 19 |
| `icl-notepad-gemini-3.1-pro-preview` | codebase_adaptation | 0 | 6.050 | 8.425 | 19.000 | -2.375 | -0.356 | -0.225 | $3.58 | 4.35 | 19 |
| `icl-notepad-gemini-3.1-pro-preview` | codebase_adaptation | 1 | 8.650 | 8.425 | 19.000 | 0.225 | -0.084 | 0.021 | $3.37 | 4.29 | 19 |
| `icl-notepad-gemini-3.1-pro-preview` | codebase_adaptation | 2 | 8.675 | 8.425 | 19.000 | 0.250 | -0.081 | 0.024 | $3.46 | 4.20 | 19 |
| `icl-notepad-gemini-3.1-pro-preview` | codebase_adaptation | 3 | 6.100 | 8.425 | 19.000 | -2.325 | -0.351 | -0.220 | $3.46 | 4.40 | 19 |
| `icl-notepad-gemini-3.1-pro-preview` | codebase_adaptation | 4 | 6.075 | 8.425 | 19.000 | -2.350 | -0.353 | -0.222 | $3.66 | 4.20 | 19 |
| `icl-notepad-gpt-5.4` | codebase_adaptation | 0 | 9.825 | 8.250 | 19.000 | 1.575 | 0.039 | 0.147 | $3.73 | 10.11 | 19 |
| `icl-notepad-gpt-5.4` | codebase_adaptation | 1 | 10.200 | 8.250 | 19.000 | 1.950 | 0.079 | 0.181 | $3.29 | 9.65 | 19 |
| `icl-notepad-gpt-5.4` | codebase_adaptation | 2 | 9.875 | 8.250 | 19.000 | 1.625 | 0.045 | 0.151 | $3.33 | 9.61 | 19 |
| `icl-notepad-gpt-5.4` | codebase_adaptation | 3 | 6.950 | 8.250 | 19.000 | -1.300 | -0.262 | -0.121 | $3.00 | 8.38 | 19 |
| `icl-notepad-gpt-5.4` | codebase_adaptation | 4 | 7.450 | 8.250 | 19.000 | -0.800 | -0.209 | -0.074 | $4.12 | 10.11 | 19 |
| `mem0-gpt-5.4` | codebase_adaptation | 0 | 10.750 | 8.125 | 19.000 | 2.625 | 0.136 | 0.241 | $2.54 | 4.88 | 19 |
| `mem0-gpt-5.4` | codebase_adaptation | 1 | 13.375 | 8.125 | 19.000 | 5.250 | 0.411 | 0.483 | $2.87 | 4.79 | 19 |
| `mem0-gpt-5.4` | codebase_adaptation | 2 | 7.450 | 8.125 | 19.000 | -0.675 | -0.209 | -0.062 | $3.06 | 5.92 | 19 |
| `mem0-gpt-5.4` | codebase_adaptation | 3 | 10.550 | 8.125 | 19.000 | 2.425 | 0.115 | 0.223 | $2.52 | 5.55 | 19 |
| `mem0-gpt-5.4` | codebase_adaptation | 4 | 13.400 | 8.125 | 19.000 | 5.275 | 0.414 | 0.485 | $2.13 | 4.75 | 19 |
| `ace-gpt-5.4` | cohort_studies | 0 | 0.952 | 0.376 | 3.244 | 0.576 | -0.019 | 0.201 | $13.02 | 6.56 | 20 |
| `ace-gpt-5.4` | cohort_studies | 1 | 0.341 | 0.376 | 3.244 | -0.035 | -0.291 | -0.012 | $12.19 | 6.47 | 20 |
| `ace-gpt-5.4` | cohort_studies | 2 | 1.021 | 0.376 | 3.244 | 0.646 | 0.012 | 0.225 | $15.19 | 6.08 | 20 |
| `ace-gpt-5.4` | cohort_studies | 3 | 0.599 | 0.376 | 3.244 | 0.223 | -0.176 | 0.078 | $8.92 | 6.31 | 20 |
| `ace-gpt-5.4` | cohort_studies | 4 | 0.881 | 0.376 | 3.244 | 0.506 | -0.050 | 0.176 | $14.67 | 6.25 | 20 |
| `claude-code-sonnet-4.6` | cohort_studies | 0 | 0.267 | 0.805 | 3.244 | -0.538 | -0.323 | -0.220 | $7.82 | 25.34 | 20 |
| `claude-code-sonnet-4.6` | cohort_studies | 1 | 0.110 | 0.805 | 3.244 | -0.694 | -0.393 | -0.285 | $7.70 | 24.91 | 20 |
| `claude-code-sonnet-4.6` | cohort_studies | 2 | 0.685 | 0.805 | 3.244 | -0.120 | -0.138 | -0.049 | $5.99 | 21.51 | 20 |
| `claude-code-sonnet-4.6` | cohort_studies | 3 | 1.298 | 0.805 | 3.244 | 0.493 | 0.135 | 0.202 | $6.10 | 20.28 | 20 |
| `claude-code-sonnet-4.6` | cohort_studies | 4 | 0.121 | 0.805 | 3.244 | -0.684 | -0.388 | -0.280 | $8.46 | 33.65 | 20 |
| `codex-gpt-5.4` | cohort_studies | 0 | 0.821 | 0.505 | 3.244 | 0.316 | -0.077 | 0.115 | $7.76 | 13.10 | 20 |
| `icl-claude-opus-4.7` | cohort_studies | 0 | -0.200 | -0.030 | 3.244 | -0.170 | -0.531 | -0.052 | $7.02 | 8.29 | 20 |
| `icl-claude-opus-4.7` | cohort_studies | 1 | 0.035 | -0.030 | 3.244 | 0.065 | -0.426 | 0.020 | $6.98 | 8.02 | 20 |
| `icl-claude-opus-4.7` | cohort_studies | 2 | -0.043 | -0.030 | 3.244 | -0.013 | -0.461 | -0.004 | $6.44 | 7.10 | 20 |
| `icl-claude-opus-4.7` | cohort_studies | 3 | 0.049 | -0.030 | 3.244 | 0.079 | -0.420 | 0.024 | $6.67 | 8.27 | 20 |
| `icl-claude-opus-4.7` | cohort_studies | 4 | -0.446 | -0.030 | 3.244 | -0.416 | -0.640 | -0.127 | $7.73 | 8.70 | 20 |
| `icl-claude-sonnet-4.6` | cohort_studies | 0 | 0.451 | 0.576 | 3.244 | -0.125 | -0.241 | -0.047 | $5.70 | 13.04 | 20 |
| `icl-claude-sonnet-4.6` | cohort_studies | 1 | 0.010 | 0.576 | 3.244 | -0.566 | -0.437 | -0.212 | $5.37 | 13.45 | 20 |
| `icl-claude-sonnet-4.6` | cohort_studies | 2 | 1.241 | 0.576 | 3.244 | 0.665 | 0.110 | 0.249 | $7.09 | 22.76 | 20 |
| `icl-claude-sonnet-4.6` | cohort_studies | 3 | 0.840 | 0.576 | 3.244 | 0.264 | -0.069 | 0.099 | $5.20 | 13.93 | 20 |
| `icl-claude-sonnet-4.6` | cohort_studies | 4 | 1.265 | 0.576 | 3.244 | 0.689 | 0.120 | 0.258 | $4.74 | 16.45 | 20 |
| `icl-gemini-3-flash` | cohort_studies | 0 | 0.218 | 0.140 | 3.244 | 0.078 | -0.345 | 0.025 | $1.11 | 3.54 | 20 |
| `icl-gemini-3-flash` | cohort_studies | 1 | 0.211 | 0.140 | 3.244 | 0.072 | -0.348 | 0.023 | $1.59 | 7.33 | 20 |
| `icl-gemini-3-flash` | cohort_studies | 2 | 0.380 | 0.140 | 3.244 | 0.240 | -0.273 | 0.077 | $1.21 | 3.97 | 20 |
| `icl-gemini-3-flash` | cohort_studies | 3 | 1.352 | 0.140 | 3.244 | 1.213 | 0.159 | 0.391 | $0.77 | 3.70 | 20 |
| `icl-gemini-3-flash` | cohort_studies | 4 | 0.719 | 0.140 | 3.244 | 0.580 | -0.122 | 0.187 | $1.54 | 4.04 | 20 |
| `icl-gemini-3.1-pro-preview` | cohort_studies | 0 | 0.131 | 0.820 | 3.244 | -0.689 | -0.384 | -0.284 | $1.99 | 7.88 | 20 |
| `icl-gemini-3.1-pro-preview` | cohort_studies | 1 | 0.409 | 0.820 | 3.244 | -0.411 | -0.260 | -0.170 | $1.82 | 8.75 | 20 |
| `icl-gemini-3.1-pro-preview` | cohort_studies | 2 | 0.001 | 0.820 | 3.244 | -0.819 | -0.442 | -0.338 | $2.07 | 7.99 | 20 |
| `icl-gemini-3.1-pro-preview` | cohort_studies | 3 | 0.746 | 0.820 | 3.244 | -0.075 | -0.111 | -0.031 | $1.82 | 8.52 | 20 |
| `icl-gemini-3.1-pro-preview` | cohort_studies | 4 | 0.000 | 0.820 | 3.244 | -0.820 | -0.442 | -0.338 | $1.47 | 16.64 | 20 |
| `icl-gpt-5.4` | cohort_studies | 0 | 0.152 | 0.994 | 3.244 | -0.842 | -0.374 | -0.374 | $3.96 | 15.90 | 20 |
| `icl-gpt-5.4` | cohort_studies | 1 | 1.682 | 0.994 | 3.244 | 0.688 | 0.306 | 0.306 | $3.09 | 9.48 | 20 |
| `icl-gpt-5.4` | cohort_studies | 2 | 0.668 | 0.994 | 3.244 | -0.327 | -0.145 | -0.145 | $3.75 | 8.23 | 20 |
| `icl-gpt-5.4` | cohort_studies | 3 | 0.203 | 0.994 | 3.244 | -0.791 | -0.352 | -0.352 | $4.87 | 11.83 | 20 |
| `icl-gpt-5.4` | cohort_studies | 4 | 2.080 | 0.994 | 3.244 | 1.085 | 0.482 | 0.482 | $2.99 | 10.64 | 20 |
| `icl-notepad-claude-sonnet-4-6` | cohort_studies | 0 | -0.141 | -1.579 | 3.244 | 1.439 | -0.505 | 0.298 | $11.27 | 25.01 | 20 |
| `icl-notepad-claude-sonnet-4-6` | cohort_studies | 1 | -0.391 | -1.579 | 3.244 | 1.188 | -0.616 | 0.246 | $11.65 | 24.40 | 20 |
| `icl-notepad-claude-sonnet-4-6` | cohort_studies | 2 | -2.112 | -1.579 | 3.244 | -0.532 | -1.381 | -0.110 | $11.10 | 23.09 | 20 |
| `icl-notepad-claude-sonnet-4-6` | cohort_studies | 3 | -0.944 | -1.579 | 3.244 | 0.635 | -0.862 | 0.132 | $12.39 | 25.89 | 20 |
| `icl-notepad-claude-sonnet-4-6` | cohort_studies | 4 | -0.332 | -1.579 | 3.244 | 1.247 | -0.590 | 0.259 | $11.39 | 23.24 | 20 |
| `icl-notepad-gemini-3.1-pro-preview` | cohort_studies | 0 | 0.240 | 0.692 | 3.244 | -0.453 | -0.335 | -0.177 | $1.46 | 6.63 | 20 |
| `icl-notepad-gemini-3.1-pro-preview` | cohort_studies | 1 | 0.656 | 0.692 | 3.244 | -0.036 | -0.150 | -0.014 | $1.66 | 5.86 | 20 |
| `icl-notepad-gemini-3.1-pro-preview` | cohort_studies | 2 | 0.194 | 0.692 | 3.244 | -0.498 | -0.356 | -0.195 | $1.65 | 6.19 | 20 |
| `icl-notepad-gemini-3.1-pro-preview` | cohort_studies | 3 | 0.207 | 0.692 | 3.244 | -0.486 | -0.350 | -0.190 | $1.34 | 7.34 | 20 |
| `icl-notepad-gemini-3.1-pro-preview` | cohort_studies | 4 | 0.336 | 0.692 | 3.244 | -0.357 | -0.293 | -0.140 | $1.55 | 6.24 | 20 |
| `icl-notepad-gpt-5.4` | cohort_studies | 0 | 0.623 | 1.376 | 3.244 | -0.753 | -0.165 | -0.403 | $4.60 | 10.60 | 20 |
| `icl-notepad-gpt-5.4` | cohort_studies | 1 | 0.504 | 1.376 | 3.244 | -0.871 | -0.218 | -0.466 | $4.44 | 10.19 | 20 |
| `icl-notepad-gpt-5.4` | cohort_studies | 2 | 0.372 | 1.376 | 3.244 | -1.003 | -0.277 | -0.537 | $4.36 | 9.46 | 20 |
| `icl-notepad-gpt-5.4` | cohort_studies | 3 | 0.493 | 1.376 | 3.244 | -0.883 | -0.223 | -0.472 | $4.31 | 11.61 | 20 |
| `icl-notepad-gpt-5.4` | cohort_studies | 4 | 0.386 | 1.376 | 3.244 | -0.990 | -0.271 | -0.530 | $4.49 | 10.86 | 20 |
| `mem0-gpt-5.4` | cohort_studies | 0 | 0.236 | 0.845 | 3.244 | -0.608 | -0.337 | -0.254 | $6.91 | 7.59 | 20 |
| `mem0-gpt-5.4` | cohort_studies | 1 | 0.654 | 0.845 | 3.244 | -0.191 | -0.152 | -0.080 | $4.61 | 9.30 | 20 |
| `mem0-gpt-5.4` | cohort_studies | 2 | 1.822 | 0.845 | 3.244 | 0.977 | 0.368 | 0.407 | $4.90 | 8.95 | 20 |
| `mem0-gpt-5.4` | cohort_studies | 3 | 0.081 | 0.845 | 3.244 | -0.764 | -0.406 | -0.318 | $7.44 | 11.80 | 20 |
| `mem0-gpt-5.4` | cohort_studies | 4 | 1.098 | 0.845 | 3.244 | 0.253 | 0.046 | 0.105 | $6.13 | 7.76 | 20 |
| `ace-gpt-5.4` | database_exploration | 0 | 5.600 | 5.467 | 40.000 | 0.133 | 0.002 | 0.004 | $10.07 | 3.65 | 40 |
| `ace-gpt-5.4` | database_exploration | 1 | 8.000 | 5.467 | 40.000 | 2.533 | 0.072 | 0.073 | $8.33 | 3.79 | 40 |
| `ace-gpt-5.4` | database_exploration | 2 | 12.733 | 5.467 | 40.000 | 7.267 | 0.209 | 0.210 | $6.43 | 3.72 | 40 |
| `ace-gpt-5.4` | database_exploration | 3 | 7.000 | 5.467 | 40.000 | 1.533 | 0.043 | 0.044 | $10.31 | 3.69 | 40 |
| `ace-gpt-5.4` | database_exploration | 4 | 5.933 | 5.467 | 40.000 | 0.467 | 0.012 | 0.014 | $8.74 | 3.67 | 40 |
| `claude-code-sonnet-4.6` | database_exploration | 0 | 24.533 | 8.200 | 40.000 | 16.333 | 0.551 | 0.514 | $3.72 | 6.96 | 40 |
| `claude-code-sonnet-4.6` | database_exploration | 1 | 17.600 | 8.200 | 40.000 | 9.400 | 0.350 | 0.296 | $2.69 | 5.28 | 40 |
| `claude-code-sonnet-4.6` | database_exploration | 2 | 22.133 | 8.200 | 40.000 | 13.933 | 0.482 | 0.438 | $3.91 | 5.85 | 40 |
| `claude-code-sonnet-4.6` | database_exploration | 3 | 23.267 | 8.200 | 40.000 | 15.067 | 0.515 | 0.474 | $3.40 | 5.70 | 40 |
| `claude-code-sonnet-4.6` | database_exploration | 4 | 22.733 | 8.200 | 40.000 | 14.533 | 0.499 | 0.457 | $2.97 | 5.14 | 40 |
| `codex-gpt-5.4` | database_exploration | 0 | 9.600 | 3.467 | 40.000 | 6.133 | 0.118 | 0.168 | $1.85 | 4.74 | 40 |
| `icl-claude-opus-4.7` | database_exploration | 0 | 15.333 | 6.067 | 40.000 | 9.267 | 0.284 | 0.273 | $5.92 | 4.20 | 40 |
| `icl-claude-opus-4.7` | database_exploration | 1 | 12.000 | 6.067 | 40.000 | 5.933 | 0.188 | 0.175 | $5.42 | 4.72 | 40 |
| `icl-claude-opus-4.7` | database_exploration | 2 | 21.200 | 6.067 | 40.000 | 15.133 | 0.455 | 0.446 | $4.88 | 4.79 | 40 |
| `icl-claude-opus-4.7` | database_exploration | 3 | 11.867 | 6.067 | 40.000 | 5.800 | 0.184 | 0.171 | $5.26 | 4.19 | 40 |
| `icl-claude-opus-4.7` | database_exploration | 4 | 17.867 | 6.067 | 40.000 | 11.800 | 0.358 | 0.348 | $4.62 | 4.45 | 40 |
| `icl-claude-sonnet-4.6` | database_exploration | 0 | 14.667 | 6.533 | 40.000 | 8.133 | 0.265 | 0.243 | $2.18 | 2.12 | 40 |
| `icl-claude-sonnet-4.6` | database_exploration | 1 | 16.800 | 6.533 | 40.000 | 10.267 | 0.327 | 0.307 | $1.47 | 1.87 | 40 |
| `icl-claude-sonnet-4.6` | database_exploration | 2 | 17.333 | 6.533 | 40.000 | 10.800 | 0.342 | 0.323 | $2.21 | 2.16 | 40 |
| `icl-claude-sonnet-4.6` | database_exploration | 3 | 9.600 | 6.533 | 40.000 | 3.067 | 0.118 | 0.092 | $2.22 | 2.04 | 40 |
| `icl-claude-sonnet-4.6` | database_exploration | 4 | 16.667 | 6.533 | 40.000 | 10.133 | 0.323 | 0.303 | $1.67 | 1.93 | 40 |
| `icl-gemini-3-flash` | database_exploration | 0 | 14.267 | 3.533 | 40.000 | 10.733 | 0.253 | 0.294 | $0.41 | 1.43 | 40 |
| `icl-gemini-3-flash` | database_exploration | 1 | 13.400 | 3.533 | 40.000 | 9.867 | 0.228 | 0.271 | $0.46 | 1.40 | 40 |
| `icl-gemini-3-flash` | database_exploration | 2 | 14.000 | 3.533 | 40.000 | 10.467 | 0.246 | 0.287 | $0.39 | 1.40 | 40 |
| `icl-gemini-3-flash` | database_exploration | 3 | 16.600 | 3.533 | 40.000 | 13.067 | 0.321 | 0.358 | $0.38 | 1.59 | 40 |
| `icl-gemini-3-flash` | database_exploration | 4 | 16.867 | 3.533 | 40.000 | 13.333 | 0.329 | 0.366 | $0.45 | 1.54 | 40 |
| `icl-gemini-3.1-pro-preview` | database_exploration | 0 | 10.800 | 4.733 | 40.000 | 6.067 | 0.153 | 0.172 | $1.58 | 6.86 | 40 |
| `icl-gemini-3.1-pro-preview` | database_exploration | 1 | 1.067 | 4.733 | 40.000 | -3.667 | -0.130 | -0.104 | $0.77 | 5.56 | 40 |
| `icl-gemini-3.1-pro-preview` | database_exploration | 2 | 12.200 | 4.733 | 40.000 | 7.467 | 0.193 | 0.212 | $1.42 | 5.60 | 40 |
| `icl-gemini-3.1-pro-preview` | database_exploration | 3 | 17.000 | 4.733 | 40.000 | 12.267 | 0.333 | 0.348 | $1.57 | 6.69 | 40 |
| `icl-gemini-3.1-pro-preview` | database_exploration | 4 | 16.733 | 4.733 | 40.000 | 12.000 | 0.325 | 0.340 | $1.27 | 5.71 | 40 |
| `icl-gpt-5.4` | database_exploration | 0 | 15.467 | 5.533 | 40.000 | 9.933 | 0.288 | 0.288 | $1.19 | 3.96 | 40 |
| `icl-gpt-5.4` | database_exploration | 1 | 7.467 | 5.533 | 40.000 | 1.933 | 0.056 | 0.056 | $0.91 | 3.79 | 40 |
| `icl-gpt-5.4` | database_exploration | 2 | 14.800 | 5.533 | 40.000 | 9.267 | 0.269 | 0.269 | $0.99 | 3.59 | 40 |
| `icl-gpt-5.4` | database_exploration | 3 | 14.000 | 5.533 | 40.000 | 8.467 | 0.246 | 0.246 | $0.98 | 3.55 | 40 |
| `icl-gpt-5.4` | database_exploration | 4 | 17.667 | 5.533 | 40.000 | 12.133 | 0.352 | 0.352 | $1.08 | 3.98 | 40 |
| `icl-notepad-claude-sonnet-4-6` | database_exploration | 0 | 12.933 | 7.333 | 40.000 | 5.600 | 0.215 | 0.171 | $2.11 | 2.37 | 40 |
| `icl-notepad-claude-sonnet-4-6` | database_exploration | 1 | 10.400 | 7.333 | 40.000 | 3.067 | 0.141 | 0.094 | $2.26 | 2.36 | 40 |
| `icl-notepad-claude-sonnet-4-6` | database_exploration | 2 | 13.000 | 7.333 | 40.000 | 5.667 | 0.217 | 0.173 | $2.15 | 2.47 | 40 |
| `icl-notepad-claude-sonnet-4-6` | database_exploration | 3 | 10.267 | 7.333 | 40.000 | 2.933 | 0.137 | 0.090 | $2.34 | 2.37 | 40 |
| `icl-notepad-claude-sonnet-4-6` | database_exploration | 4 | 8.400 | 7.333 | 40.000 | 1.067 | 0.083 | 0.033 | $2.34 | 2.27 | 40 |
| `icl-notepad-gemini-3.1-pro-preview` | database_exploration | 0 | 9.800 | 4.200 | 40.000 | 5.600 | 0.124 | 0.156 | $2.26 | 4.47 | 40 |
| `icl-notepad-gemini-3.1-pro-preview` | database_exploration | 1 | 8.200 | 4.200 | 40.000 | 4.000 | 0.077 | 0.112 | $2.57 | 4.70 | 40 |
| `icl-notepad-gemini-3.1-pro-preview` | database_exploration | 2 | 10.267 | 4.200 | 40.000 | 6.067 | 0.137 | 0.169 | $2.35 | 4.60 | 40 |
| `icl-notepad-gemini-3.1-pro-preview` | database_exploration | 3 | 7.067 | 4.200 | 40.000 | 2.867 | 0.044 | 0.080 | $2.91 | 4.41 | 40 |
| `icl-notepad-gemini-3.1-pro-preview` | database_exploration | 4 | 7.267 | 4.200 | 40.000 | 3.067 | 0.050 | 0.086 | $3.07 | 4.33 | 40 |
| `icl-notepad-gpt-5.4` | database_exploration | 0 | 11.400 | 6.000 | 40.000 | 5.400 | 0.170 | 0.159 | $1.30 | 3.43 | 40 |
| `icl-notepad-gpt-5.4` | database_exploration | 1 | 10.867 | 6.000 | 40.000 | 4.867 | 0.155 | 0.143 | $1.61 | 3.87 | 40 |
| `icl-notepad-gpt-5.4` | database_exploration | 2 | 15.867 | 6.000 | 40.000 | 9.867 | 0.300 | 0.290 | $1.36 | 3.60 | 40 |
| `icl-notepad-gpt-5.4` | database_exploration | 3 | 14.200 | 6.000 | 40.000 | 8.200 | 0.251 | 0.241 | $1.36 | 3.33 | 40 |
| `icl-notepad-gpt-5.4` | database_exploration | 4 | 9.533 | 6.000 | 40.000 | 3.533 | 0.116 | 0.104 | $1.61 | 3.47 | 40 |
| `mem0-gpt-5.4` | database_exploration | 0 | 16.800 | 4.333 | 40.000 | 12.467 | 0.327 | 0.350 | $2.09 | 2.91 | 40 |
| `mem0-gpt-5.4` | database_exploration | 1 | 17.267 | 4.333 | 40.000 | 12.933 | 0.340 | 0.363 | $2.28 | 3.11 | 40 |
| `mem0-gpt-5.4` | database_exploration | 2 | 18.400 | 4.333 | 40.000 | 14.067 | 0.373 | 0.394 | $1.69 | 2.97 | 40 |
| `mem0-gpt-5.4` | database_exploration | 3 | 12.867 | 4.333 | 40.000 | 8.533 | 0.213 | 0.239 | $1.81 | 3.08 | 40 |
| `mem0-gpt-5.4` | database_exploration | 4 | 20.867 | 4.333 | 40.000 | 16.533 | 0.445 | 0.464 | $1.97 | 3.02 | 40 |
| `ace-gpt-5.4` | exploitable_poker | 0 | 0.300 | 141.900 | 1138.500 | -141.600 | -0.133 | -0.142 | $15.75 | 4.98 | 120 |
| `ace-gpt-5.4` | exploitable_poker | 1 | 240.500 | 141.900 | 1138.500 | 98.600 | 0.106 | 0.099 | $10.93 | 5.08 | 120 |
| `ace-gpt-5.4` | exploitable_poker | 2 | 160.300 | 141.900 | 1138.500 | 18.400 | 0.027 | 0.018 | $12.27 | 4.66 | 120 |
| `ace-gpt-5.4` | exploitable_poker | 3 | 117.500 | 141.900 | 1138.500 | -24.400 | -0.016 | -0.024 | $13.89 | 4.75 | 120 |
| `ace-gpt-5.4` | exploitable_poker | 4 | 198.900 | 141.900 | 1138.500 | 57.000 | 0.065 | 0.057 | $13.18 | 4.77 | 120 |
| `claude-code-sonnet-4.6` | exploitable_poker | 0 | 363.800 | 284.500 | 1138.500 | 79.300 | 0.229 | 0.093 | $7.99 | 7.04 | 120 |
| `claude-code-sonnet-4.6` | exploitable_poker | 1 | 381.600 | 284.500 | 1138.500 | 97.100 | 0.247 | 0.114 | $9.65 | 6.24 | 120 |
| `claude-code-sonnet-4.6` | exploitable_poker | 2 | 403.600 | 284.500 | 1138.500 | 119.100 | 0.269 | 0.139 | $8.93 | 6.25 | 120 |
| `claude-code-sonnet-4.6` | exploitable_poker | 3 | 330.300 | 284.500 | 1138.500 | 45.800 | 0.196 | 0.054 | $8.60 | 5.75 | 120 |
| `claude-code-sonnet-4.6` | exploitable_poker | 4 | 235.800 | 284.500 | 1138.500 | -48.700 | 0.102 | -0.057 | $8.09 | 7.77 | 120 |
| `codex-gpt-5.4` | exploitable_poker | 0 | 85.000 | 64.500 | 1138.500 | 20.500 | -0.048 | 0.019 | $8.27 | 5.88 | 120 |
| `icl-claude-opus-4.7` | exploitable_poker | 0 | 124.000 | 157.700 | 1138.500 | -33.700 | -0.010 | -0.034 | $17.65 | 4.73 | 120 |
| `icl-claude-opus-4.7` | exploitable_poker | 1 | 62.000 | 157.700 | 1138.500 | -95.700 | -0.071 | -0.098 | $19.47 | 4.90 | 120 |
| `icl-claude-opus-4.7` | exploitable_poker | 2 | 109.800 | 157.700 | 1138.500 | -47.900 | -0.024 | -0.049 | $15.33 | 5.15 | 120 |
| `icl-claude-opus-4.7` | exploitable_poker | 3 | 75.000 | 157.700 | 1138.500 | -82.700 | -0.058 | -0.084 | $17.25 | 4.82 | 120 |
| `icl-claude-opus-4.7` | exploitable_poker | 4 | 212.600 | 157.700 | 1138.500 | 54.900 | 0.079 | 0.056 | $17.63 | 5.14 | 120 |
| `icl-claude-sonnet-4.6` | exploitable_poker | 0 | 379.000 | 316.700 | 1138.500 | 62.300 | 0.244 | 0.076 | $8.77 | 8.09 | 120 |
| `icl-claude-sonnet-4.6` | exploitable_poker | 1 | 314.500 | 316.700 | 1138.500 | -2.200 | 0.180 | -0.003 | $8.90 | 8.14 | 120 |
| `icl-claude-sonnet-4.6` | exploitable_poker | 2 | 323.800 | 316.700 | 1138.500 | 7.100 | 0.189 | 0.009 | $10.98 | 9.40 | 120 |
| `icl-claude-sonnet-4.6` | exploitable_poker | 3 | 290.200 | 316.700 | 1138.500 | -26.500 | 0.156 | -0.032 | $8.93 | 8.19 | 120 |
| `icl-claude-sonnet-4.6` | exploitable_poker | 4 | 392.100 | 316.700 | 1138.500 | 75.400 | 0.257 | 0.092 | $9.34 | 8.35 | 120 |
| `icl-gemini-3-flash` | exploitable_poker | 0 | 64.000 | 196.800 | 1138.500 | -132.800 | -0.069 | -0.141 | $1.74 | 2.30 | 120 |
| `icl-gemini-3-flash` | exploitable_poker | 1 | 129.500 | 196.800 | 1138.500 | -67.300 | -0.004 | -0.071 | $1.57 | 2.43 | 120 |
| `icl-gemini-3-flash` | exploitable_poker | 2 | 137.000 | 196.800 | 1138.500 | -59.800 | 0.003 | -0.064 | $1.72 | 2.47 | 120 |
| `icl-gemini-3-flash` | exploitable_poker | 3 | 91.000 | 196.800 | 1138.500 | -105.800 | -0.042 | -0.112 | $2.21 | 2.29 | 120 |
| `icl-gemini-3-flash` | exploitable_poker | 4 | 52.700 | 196.800 | 1138.500 | -144.100 | -0.081 | -0.153 | $2.33 | 2.73 | 120 |
| `icl-gemini-3.1-pro-preview` | exploitable_poker | 0 | 74.500 | 43.500 | 1138.500 | 31.000 | -0.059 | 0.028 | $3.82 | 7.53 | 120 |
| `icl-gemini-3.1-pro-preview` | exploitable_poker | 1 | 64.500 | 43.500 | 1138.500 | 21.000 | -0.069 | 0.019 | $4.14 | 5.69 | 120 |
| `icl-gemini-3.1-pro-preview` | exploitable_poker | 2 | 88.500 | 43.500 | 1138.500 | 45.000 | -0.045 | 0.041 | $3.75 | 6.27 | 120 |
| `icl-gemini-3.1-pro-preview` | exploitable_poker | 3 | 75.500 | 43.500 | 1138.500 | 32.000 | -0.058 | 0.029 | $4.05 | 6.65 | 120 |
| `icl-gemini-3.1-pro-preview` | exploitable_poker | 4 | 79.000 | 43.500 | 1138.500 | 35.500 | -0.054 | 0.032 | $4.19 | 6.32 | 120 |
| `icl-gpt-5.4` | exploitable_poker | 0 | 127.500 | 133.600 | 1138.500 | -6.100 | -0.006 | -0.006 | $4.50 | 6.44 | 120 |
| `icl-gpt-5.4` | exploitable_poker | 1 | 64.000 | 133.600 | 1138.500 | -69.600 | -0.069 | -0.069 | $4.91 | 6.66 | 120 |
| `icl-gpt-5.4` | exploitable_poker | 2 | 21.000 | 133.600 | 1138.500 | -112.600 | -0.112 | -0.112 | $4.57 | 6.48 | 120 |
| `icl-gpt-5.4` | exploitable_poker | 3 | 91.000 | 133.600 | 1138.500 | -42.600 | -0.042 | -0.042 | $4.38 | 6.12 | 120 |
| `icl-gpt-5.4` | exploitable_poker | 4 | 175.300 | 133.600 | 1138.500 | 41.700 | 0.041 | 0.041 | $4.69 | 6.54 | 120 |
| `icl-notepad-claude-sonnet-4-6` | exploitable_poker | 0 | 166.800 | 317.000 | 1138.500 | -150.200 | 0.033 | -0.183 | $6.56 | 16.28 | 120 |
| `icl-notepad-claude-sonnet-4-6` | exploitable_poker | 1 | 149.700 | 317.000 | 1138.500 | -167.300 | 0.016 | -0.204 | $6.62 | 15.91 | 120 |
| `icl-notepad-claude-sonnet-4-6` | exploitable_poker | 2 | 20.500 | 317.000 | 1138.500 | -296.500 | -0.113 | -0.361 | $6.02 | 15.88 | 120 |
| `icl-notepad-claude-sonnet-4-6` | exploitable_poker | 3 | 170.300 | 317.000 | 1138.500 | -146.700 | 0.037 | -0.179 | $6.64 | 16.68 | 120 |
| `icl-notepad-claude-sonnet-4-6` | exploitable_poker | 4 | 69.800 | 317.000 | 1138.500 | -247.200 | -0.063 | -0.301 | $7.89 | 20.52 | 120 |
| `icl-notepad-gemini-3.1-pro-preview` | exploitable_poker | 0 | 75.500 | 36.500 | 1138.500 | 39.000 | -0.058 | 0.035 | $1.28 | 5.41 | 120 |
| `icl-notepad-gemini-3.1-pro-preview` | exploitable_poker | 1 | 41.000 | 36.500 | 1138.500 | 4.500 | -0.092 | 0.004 | $1.35 | 5.33 | 120 |
| `icl-notepad-gemini-3.1-pro-preview` | exploitable_poker | 2 | 62.500 | 36.500 | 1138.500 | 26.000 | -0.071 | 0.024 | $1.33 | 5.20 | 120 |
| `icl-notepad-gemini-3.1-pro-preview` | exploitable_poker | 3 | 43.500 | 36.500 | 1138.500 | 7.000 | -0.090 | 0.006 | $1.40 | 5.51 | 120 |
| `icl-notepad-gemini-3.1-pro-preview` | exploitable_poker | 4 | 45.000 | 36.500 | 1138.500 | 8.500 | -0.088 | 0.008 | $1.28 | 5.29 | 120 |
| `icl-notepad-gpt-5.4` | exploitable_poker | 0 | 86.500 | 81.100 | 1138.500 | 5.400 | -0.047 | 0.005 | $1.52 | 4.31 | 120 |
| `icl-notepad-gpt-5.4` | exploitable_poker | 1 | 92.000 | 81.100 | 1138.500 | 10.900 | -0.041 | 0.010 | $1.54 | 4.29 | 120 |
| `icl-notepad-gpt-5.4` | exploitable_poker | 2 | 112.000 | 81.100 | 1138.500 | 30.900 | -0.021 | 0.029 | $1.64 | 4.64 | 120 |
| `icl-notepad-gpt-5.4` | exploitable_poker | 3 | 23.000 | 81.100 | 1138.500 | -58.100 | -0.110 | -0.055 | $1.51 | 4.31 | 120 |
| `icl-notepad-gpt-5.4` | exploitable_poker | 4 | 93.200 | 81.100 | 1138.500 | 12.100 | -0.040 | 0.011 | $1.38 | 4.27 | 120 |
| `mem0-gpt-5.4` | exploitable_poker | 0 | 66.200 | 90.400 | 1138.500 | -24.200 | -0.067 | -0.023 | $3.76 | 4.67 | 120 |
| `mem0-gpt-5.4` | exploitable_poker | 1 | 230.400 | 90.400 | 1138.500 | 140.000 | 0.096 | 0.134 | $3.88 | 4.66 | 120 |
| `mem0-gpt-5.4` | exploitable_poker | 2 | 49.000 | 90.400 | 1138.500 | -41.400 | -0.084 | -0.040 | $3.58 | 5.08 | 120 |
| `mem0-gpt-5.4` | exploitable_poker | 3 | -15.500 | 90.400 | 1138.500 | -105.900 | -0.148 | -0.101 | $3.37 | 4.59 | 120 |
| `mem0-gpt-5.4` | exploitable_poker | 4 | 37.000 | 90.400 | 1138.500 | -53.400 | -0.096 | -0.051 | $3.28 | 4.35 | 120 |
| `ace-gpt-5.4` | sales_prediction | 0 | 6.338 | 5.203 | 12.000 | 1.135 | 0.116 | 0.167 | $7.11 | 7.45 | 12 |
| `ace-gpt-5.4` | sales_prediction | 1 | 6.340 | 5.203 | 12.000 | 1.137 | 0.116 | 0.167 | $7.96 | 7.05 | 12 |
| `ace-gpt-5.4` | sales_prediction | 2 | 6.274 | 5.203 | 12.000 | 1.071 | 0.106 | 0.158 | $8.71 | 7.23 | 12 |
| `ace-gpt-5.4` | sales_prediction | 3 | 5.922 | 5.203 | 12.000 | 0.719 | 0.051 | 0.106 | $8.05 | 7.37 | 12 |
| `ace-gpt-5.4` | sales_prediction | 4 | 5.705 | 5.203 | 12.000 | 0.502 | 0.017 | 0.074 | $9.99 | 7.25 | 12 |
| `claude-code-sonnet-4.6` | sales_prediction | 0 | 9.148 | 5.036 | 12.000 | 4.112 | 0.555 | 0.590 | $2.40 | 14.83 | 12 |
| `claude-code-sonnet-4.6` | sales_prediction | 1 | 10.488 | 5.036 | 12.000 | 5.452 | 0.764 | 0.783 | $2.08 | 10.52 | 12 |
| `claude-code-sonnet-4.6` | sales_prediction | 2 | 9.730 | 5.036 | 12.000 | 4.694 | 0.645 | 0.674 | $2.01 | 11.09 | 12 |
| `claude-code-sonnet-4.6` | sales_prediction | 3 | 9.384 | 5.036 | 12.000 | 4.348 | 0.591 | 0.624 | $2.42 | 12.77 | 12 |
| `claude-code-sonnet-4.6` | sales_prediction | 4 | 9.116 | 5.036 | 12.000 | 4.080 | 0.550 | 0.586 | $2.16 | 11.54 | 12 |
| `codex-gpt-5.4` | sales_prediction | 0 | 8.445 | 5.176 | 12.000 | 3.269 | 0.445 | 0.479 | $2.65 | 10.88 | 12 |
| `codex-gpt-5.4` | sales_prediction | 1 | 7.674 | 5.176 | 12.000 | 2.497 | 0.324 | 0.366 | $1.51 | 11.99 | 12 |
| `codex-gpt-5.4` | sales_prediction | 2 | 8.112 | 5.176 | 12.000 | 2.936 | 0.393 | 0.430 | $1.73 | 11.50 | 12 |
| `codex-gpt-5.4` | sales_prediction | 3 | 8.828 | 5.176 | 12.000 | 3.651 | 0.505 | 0.535 | $2.74 | 10.28 | 12 |
| `codex-gpt-5.4` | sales_prediction | 4 | 8.541 | 5.176 | 12.000 | 3.365 | 0.460 | 0.493 | $3.29 | 10.49 | 12 |
| `icl-claude-opus-4.7` | sales_prediction | 0 | 9.549 | 4.390 | 12.000 | 5.159 | 0.617 | 0.678 | $4.40 | 9.48 | 12 |
| `icl-claude-opus-4.7` | sales_prediction | 1 | 8.113 | 4.390 | 12.000 | 3.724 | 0.393 | 0.489 | $4.22 | 9.52 | 12 |
| `icl-claude-opus-4.7` | sales_prediction | 2 | 8.597 | 4.390 | 12.000 | 4.208 | 0.469 | 0.553 | $4.83 | 9.14 | 12 |
| `icl-claude-opus-4.7` | sales_prediction | 3 | 9.364 | 4.390 | 12.000 | 4.974 | 0.588 | 0.654 | $3.99 | 9.33 | 12 |
| `icl-claude-opus-4.7` | sales_prediction | 4 | 9.571 | 4.390 | 12.000 | 5.181 | 0.621 | 0.681 | $4.35 | 10.26 | 12 |
| `icl-claude-sonnet-4.6` | sales_prediction | 0 | 10.024 | 5.178 | 12.000 | 4.846 | 0.691 | 0.710 | $3.47 | 17.21 | 12 |
| `icl-claude-sonnet-4.6` | sales_prediction | 1 | 9.465 | 5.178 | 12.000 | 4.287 | 0.604 | 0.628 | $2.82 | 15.79 | 12 |
| `icl-claude-sonnet-4.6` | sales_prediction | 2 | 9.780 | 5.178 | 12.000 | 4.601 | 0.653 | 0.675 | $2.84 | 12.51 | 12 |
| `icl-claude-sonnet-4.6` | sales_prediction | 3 | 10.052 | 5.178 | 12.000 | 4.874 | 0.696 | 0.714 | $2.84 | 18.70 | 12 |
| `icl-claude-sonnet-4.6` | sales_prediction | 4 | 10.769 | 5.178 | 12.000 | 5.591 | 0.808 | 0.820 | $2.82 | 14.14 | 12 |
| `icl-gemini-3-flash` | sales_prediction | 0 | 8.636 | 5.271 | 12.000 | 3.366 | 0.475 | 0.500 | $0.53 | 3.83 | 12 |
| `icl-gemini-3-flash` | sales_prediction | 1 | 8.072 | 5.271 | 12.000 | 2.801 | 0.386 | 0.416 | $0.59 | 3.64 | 12 |
| `icl-gemini-3-flash` | sales_prediction | 2 | 8.017 | 5.271 | 12.000 | 2.746 | 0.378 | 0.408 | $0.78 | 3.44 | 12 |
| `icl-gemini-3-flash` | sales_prediction | 3 | 9.173 | 5.271 | 12.000 | 3.903 | 0.559 | 0.580 | $0.53 | 3.60 | 12 |
| `icl-gemini-3-flash` | sales_prediction | 4 | 8.506 | 5.271 | 12.000 | 3.235 | 0.454 | 0.481 | $0.52 | 3.35 | 12 |
| `icl-gemini-3.1-pro-preview` | sales_prediction | 0 | 6.588 | 3.908 | 12.000 | 2.680 | 0.155 | 0.331 | $0.86 | 9.40 | 12 |
| `icl-gemini-3.1-pro-preview` | sales_prediction | 1 | 6.270 | 3.908 | 12.000 | 2.362 | 0.105 | 0.292 | $0.85 | 10.49 | 12 |
| `icl-gemini-3.1-pro-preview` | sales_prediction | 2 | 6.469 | 3.908 | 12.000 | 2.561 | 0.136 | 0.316 | $0.82 | 9.70 | 12 |
| `icl-gemini-3.1-pro-preview` | sales_prediction | 3 | 6.469 | 3.908 | 12.000 | 2.561 | 0.136 | 0.316 | $0.86 | 9.13 | 12 |
| `icl-gemini-3.1-pro-preview` | sales_prediction | 4 | 6.543 | 3.908 | 12.000 | 2.635 | 0.148 | 0.326 | $0.86 | 8.96 | 12 |
| `icl-gpt-5.4` | sales_prediction | 0 | 8.386 | 5.597 | 12.000 | 2.788 | 0.436 | 0.436 | $3.43 | 10.04 | 12 |
| `icl-gpt-5.4` | sales_prediction | 1 | 9.709 | 5.597 | 12.000 | 4.111 | 0.642 | 0.642 | $3.54 | 13.61 | 12 |
| `icl-gpt-5.4` | sales_prediction | 2 | 8.905 | 5.597 | 12.000 | 3.308 | 0.517 | 0.517 | $3.76 | 11.63 | 12 |
| `icl-gpt-5.4` | sales_prediction | 3 | 9.566 | 5.597 | 12.000 | 3.969 | 0.620 | 0.620 | $3.50 | 10.91 | 12 |
| `icl-gpt-5.4` | sales_prediction | 4 | 9.585 | 5.597 | 12.000 | 3.987 | 0.623 | 0.623 | $3.36 | 10.53 | 12 |
| `icl-notepad-claude-sonnet-4-6` | sales_prediction | 0 | 10.410 | 4.282 | 12.000 | 6.127 | 0.752 | 0.794 | $4.10 | 17.31 | 12 |
| `icl-notepad-claude-sonnet-4-6` | sales_prediction | 1 | 10.045 | 4.282 | 12.000 | 5.763 | 0.695 | 0.747 | $3.82 | 18.24 | 12 |
| `icl-notepad-claude-sonnet-4-6` | sales_prediction | 2 | 9.957 | 4.282 | 12.000 | 5.675 | 0.681 | 0.735 | $3.66 | 19.92 | 12 |
| `icl-notepad-claude-sonnet-4-6` | sales_prediction | 3 | 9.844 | 4.282 | 12.000 | 5.562 | 0.663 | 0.721 | $3.57 | 19.44 | 12 |
| `icl-notepad-claude-sonnet-4-6` | sales_prediction | 4 | 10.106 | 4.282 | 12.000 | 5.824 | 0.704 | 0.755 | $3.61 | 18.19 | 12 |
| `icl-notepad-gemini-3.1-pro-preview` | sales_prediction | 0 | 8.619 | 3.156 | 12.000 | 5.463 | 0.472 | 0.618 | $1.28 | 11.37 | 12 |
| `icl-notepad-gemini-3.1-pro-preview` | sales_prediction | 1 | 8.380 | 3.156 | 12.000 | 5.224 | 0.435 | 0.591 | $1.54 | 9.08 | 12 |
| `icl-notepad-gemini-3.1-pro-preview` | sales_prediction | 2 | 6.934 | 3.156 | 12.000 | 3.778 | 0.209 | 0.427 | $1.29 | 8.57 | 12 |
| `icl-notepad-gemini-3.1-pro-preview` | sales_prediction | 3 | 7.730 | 3.156 | 12.000 | 4.574 | 0.333 | 0.517 | $1.66 | 8.45 | 12 |
| `icl-notepad-gemini-3.1-pro-preview` | sales_prediction | 4 | 8.876 | 3.156 | 12.000 | 5.720 | 0.512 | 0.647 | $1.86 | 8.09 | 12 |
| `icl-notepad-gpt-5.4` | sales_prediction | 0 | 7.752 | 4.483 | 12.000 | 3.269 | 0.336 | 0.435 | $2.52 | 13.77 | 12 |
| `icl-notepad-gpt-5.4` | sales_prediction | 1 | 9.556 | 4.483 | 12.000 | 5.073 | 0.618 | 0.675 | $1.97 | 16.33 | 12 |
| `icl-notepad-gpt-5.4` | sales_prediction | 2 | 8.801 | 4.483 | 12.000 | 4.318 | 0.500 | 0.574 | $2.55 | 9.94 | 12 |
| `icl-notepad-gpt-5.4` | sales_prediction | 3 | 8.076 | 4.483 | 12.000 | 3.593 | 0.387 | 0.478 | $2.38 | 13.70 | 12 |
| `icl-notepad-gpt-5.4` | sales_prediction | 4 | 8.244 | 4.483 | 12.000 | 3.761 | 0.413 | 0.500 | $2.33 | 13.41 | 12 |
| `mem0-gpt-5.4` | sales_prediction | 0 | 7.796 | 4.750 | 12.000 | 3.045 | 0.343 | 0.420 | $2.69 | 8.76 | 12 |
| `mem0-gpt-5.4` | sales_prediction | 1 | 8.323 | 4.750 | 12.000 | 3.573 | 0.426 | 0.493 | $1.99 | 9.68 | 12 |
| `mem0-gpt-5.4` | sales_prediction | 2 | 8.077 | 4.750 | 12.000 | 3.326 | 0.387 | 0.459 | $2.44 | 10.07 | 12 |
| `mem0-gpt-5.4` | sales_prediction | 3 | 7.146 | 4.750 | 12.000 | 2.395 | 0.242 | 0.330 | $3.11 | 11.24 | 12 |
| `mem0-gpt-5.4` | sales_prediction | 4 | 7.732 | 4.750 | 12.000 | 2.982 | 0.333 | 0.411 | $3.70 | 10.73 | 12 |
