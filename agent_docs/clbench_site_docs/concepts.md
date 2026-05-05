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
