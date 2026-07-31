# Knowledge cards — design notes

## Generalization from schema_card

The system was renamed from `schema_card` to `knowledge_cards` so it is not
tied to database exploration. The mechanism is general: answer during an
instance, reflect at instance end, inject durable cards after the system
prompt on later turns.

## Upvote / confidence removed

Earlier schema cards carried a Stack Overflow–style confidence score that
rose when feedback text contained `CORRECT` (and not `INCORRECT`). That
signal does not generalize cleanly across tasks:

- Feedback wording differs by task.
- `InstanceOutcome.success` semantics also differ (binary correctness vs
  reward thresholds vs chip profit).
- Wiring success into `observe` would need a framework contract change.

We removed upvote/confidence entirely. Cards are plain strings. Revisit only
if a cross-task success signal becomes available on observations.

## What belongs on cards

The default `reflection_prompt` prefers concrete reusable state (with a
per-card `Use when:` line), forbids low-value schema/strategy/transcript
cards, and asks the reflector to merge conflicts and drop weak one-offs.
Task-tuned prompts (e.g. BSM transmitter registry) fully replace the default
via `reflection_prompt` / `--system.reflection-prompt`.

### Suggested A/B

1. **Default** (generic prompt in code):

   ```bash
   clbench run database_exploration --system knowledge_cards --system.model <model>
   ```

2. **Custom** (paste a task-tuned prompt via ctor / CLI / config):

   ```bash
   clbench run database_exploration --system knowledge_cards \
     --system.model <model> \
     --system.reflection-prompt "$(cat path/to/prompt.txt)"
   ```

   Or put `reflection_prompt` under `system.params` in a JSON `--config` file.
   Configs are not auto-loaded.

Compare scores. A large gap suggests the reflection prompt (and thus storage
policy) matters a lot for that task.

## Hypotheses (verify with knowledge_cards runs)

Use this system to state, run, and check the following on the tasks we care
about (full multi-run bakeoffs later; early single-/few-run signals below).
Keep same-model comparisons when claiming KC vs ICL.

### H1 — Knowledge cards help (task-dependent)

**Claim:** End-of-instance reflected cards improve continual performance vs a
no-memory / non-learning baseline, but the gain depends on whether the task
has durable cross-instance state worth storing.

**Status (early):** Supported where durable environment state exists (e.g.
BSM occupancy map; schema-like DB facts). On exploitable poker, early runs
showed no meaningful lift — plausible that ~40 hands is too few to pin down
a specific opponent’s tendencies; needs longer horizons / more evidence
before calling H1 false for poker.

### H2 — Knowledge cards can beat ICL

**Claim:** With a suitable reflection policy, KC ≥ ICL on tasks tried so far
(same model), because cards compress durable state instead of relying only on
raw dialogue history.

**Status (early):** Promising on the first tasks tried (early KC runs looked
better than ICL). Not confirmed until full runs on the three planned tasks
with matched models and enough seeds. Poker may be an exception until the
horizon is long enough (see H1).

### H3 — Reflection prompt matters

**Claim:** What the reflector is told to capture (and forbid) materially
changes learning. Default vs task-tuned prompts are not interchangeable.

**Status (early):** Strongly supported on BSM. Default reflection produced
schema/strategy cards only → flat ~0.22 IoU, no learning. Task-tuned
transmitter-registry prompt → ~0.50 mean IoU and clear learning Δ. Storage
policy is part of the method; tune per task via `reflection_prompt`.

### H4 — Stronger models learn better (with KC)

**Claim:** Holding the KC system fixed, stronger models yield stronger
learning curves; weaker models still learn when the memory content is right,
but plateau lower.

**Status (early):** Directionally consistent (e.g. stronger ICL/official
models above weaker same-setup KC). Learning is present even on weaker
models once cards store the right state — capacity modulates the ceiling,
not whether reflection can help at all. Verify with same-task KC sweeps
across model tiers.

### Verification plan

- Run knowledge_cards on the chosen tasks with matched baselines (ICL,
  stateless KC) and matched models.
- A/B default vs task `reflection_prompt` where H3 is in doubt.
- Prefer multi-seed / full schedules before locking H1–H2; treat short poker
  runs as under-powered for opponent modeling.

## Drift marker

The string `NOTICE: The live database schema or contents may have changed`
remains a hard-coded trigger for clearing or marking cards STALE. Only
`database_exploration` emits it today; other tasks are unaffected.

## Archived database_exploration reflection_prompt

The former schema-card reflector prompt, kept for later A/B against the
generic default. Pass it as `--system.reflection-prompt` or
`system.params.reflection_prompt` in a run config.

```
You maintain a notebook of durable schema cards for a SQLite database agent.
Given the prior cards and one completed episode (SQL, results, feedback),
produce a FULL replacement notebook.

Rules:
- Keep only durable environment facts: tables/columns, types, encodings (cents vs dollars, timestamp units), joins, group identities, missing tables, migration/legacy/soft-delete notes.
- Do NOT store questions, SQL text, result row dumps, submitted/correct answers, or ephemeral plans.
- You may add, edit, merge, or remove cards. Newer episode evidence overrides older cards when they conflict. Prefer fewer, consistent cards.
- If feedback shows an incorrect answer caused by a bad schema assumption, fix or remove that assumption.
- If cards were stale after a migration, rewrite them from episode evidence.
- Confidence scores are maintained outside this step; return card text only.

Write slightly longer, more complete cards than a one-line gloss. For each durable fact, include enough detail that a later agent can apply it without re-deriving it: what the fact is, when it applies, related columns/keys that play the same role (if any), coverage or sparsity if you observed it, and common misuses to avoid. Prefer one thorough card over several thin ones when they describe the same area. Do not invent measurements you did not see in the episode or prior cards; if coverage is unknown, say so briefly.
```
