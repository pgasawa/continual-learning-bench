# Schema Card

End-of-instance reflected schema cards for `database_exploration`.

During a question the model only QUERY/ANSWER. After feedback, a reflector
rebuilds the card notebook (add/edit/remove) from the episode + prior cards.
Cards are injected immediately after the system prompt as trusted memory
learned earlier in the run. Each card carries a confidence score that rises
(+1 upvote) when the instance answer is marked CORRECT by the task feedback.

## Behavior

- **Stateful**: memory message after system prompt; reflect on instance end.
- **Trust**: prompt tells the model not to rediscover facts already on cards
  (unless empty, STALE, or a query error). Drift clears cards entirely by
  default, so confidence and content reset together.
- **Confidence**: exact-text card matches keep their score across reflection;
  CORRECT feedback upvotes every card in the rebuilt notebook.
- **Stateless**: no cards / no reflection (baseline).
- **Drift NOTICE**: clear cards by default so reflection rebuilds from scratch.
  Keep+mark-stale with `--system.drop-stale-cards false`.
- **Artifacts**: final cards + per-reflection snapshots under `artifacts/<trace>/`.

## Run

```bash
clbench run database_exploration --system schema_card --schedule default \
  --system.model <model> --runs 1
```
