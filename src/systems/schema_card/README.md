# Schema Card

End-of-instance reflected schema cards for `database_exploration`.

During a question the model only QUERY/ANSWER. After feedback, a reflector
rebuilds the card notebook (add/edit/remove) from the episode + prior cards.
Cards are injected immediately after the system prompt as trusted memory.

## Behavior

- **Stateful**: memory message after system prompt; reflect on instance end.
- **Stateless**: no cards / no reflection (baseline).
- **Drift NOTICE**: clear cards by default so reflection rebuilds from scratch.
  Keep+mark-stale with `--system.drop-stale-cards false`.
- **Artifacts**: final cards + per-reflection snapshots under `artifacts/<trace>/`.

## Run

```bash
clbench run database_exploration --system schema_card --schedule default \
  --system.model <model> --runs 1
```
