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
