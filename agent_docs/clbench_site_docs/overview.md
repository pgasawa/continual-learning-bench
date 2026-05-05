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
