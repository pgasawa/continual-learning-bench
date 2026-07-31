# Knowledge Cards

End-of-instance reflected knowledge cards for continual learning runs.

During an instance the model only acts/answers. After feedback, a reflector
rebuilds the card notebook (add/edit/remove) from the episode + prior cards.
Cards are injected immediately after the system prompt as trusted memory
learned earlier in the run.

## Behavior

- **Stateful**: memory message after system prompt; reflect on instance end.
- **Trust**: prompt tells the model to prefer facts already on cards (unless
  empty, STALE, or contradicted).
- **Stateless**: no cards / no reflection (baseline).
- **Drift NOTICE** (database_exploration): clear cards by default so reflection
  rebuilds from scratch. Keep+mark-stale with `--system.drop-stale-cards false`.
- **`reflection_prompt`**: generic default in code; override via
  `--system.reflection-prompt` or config `system.params.reflection_prompt`
  when a task needs a tuned storage policy (BSM transmitter registry; cohort
  studies population registry). See [notes.md](notes.md) for hypotheses and
  experiment notes.
- **Artifacts**: final cards + per-reflection snapshots under `artifacts/<trace>/`.

## Run

Default (generic reflection prompt):

```bash
clbench run database_exploration --system knowledge_cards --schedule default \
  --system.model <model> --runs 1
```

Task-tuned reflection prompts:

```bash
clbench run --config configs/blind_spectrum_monitoring/bsm_mixed_grid_knowledge_cards.json
uv run clbench run --config configs/cohort_studies/cohort_studies_knowledge_cards.json
```

The cohort config pins `system.params.model`, `task.params.schedule`,
`runs`, and the population-registry `reflection_prompt`.

Configs are only loaded when passed with `--config`; they are not auto-selected
by task name. On exploitable poker, the default reflection prompt has been
sufficient in early runs — prefer default unless A/B shows otherwise.
