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
