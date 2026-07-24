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
