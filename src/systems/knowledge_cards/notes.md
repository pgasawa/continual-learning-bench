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

Poker and database exploration use the default reflection prompt. Committed
task-tuned overrides:
- BSM (`configs/blind_spectrum_monitoring/bsm_mixed_grid_knowledge_cards.json`)
- Cohort studies
  (`configs/cohort_studies/cohort_studies_knowledge_cards.json`; prompt source
  `configs/cohort_studies/prompts/population_registry.txt`)

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

**Status:** Supported on BSM with a matched model (run still ongoing at time
of writing): same-model ICL underperforms the tuned KC registry setup.
Stronger official ICL numbers elsewhere are largely a **model** effect, not
evidence that ICL beats KC. Still confirm on the other planned tasks with
matched models / full seeds; poker may remain an exception until the horizon
is long enough (see H1).

### H3 — Reflection prompt matters

**Claim:** What the reflector is told to capture (and forbid) materially
changes learning. Default vs task-tuned prompts are not interchangeable.

**Status:** Supported by BSM alone — no further DB-exploration custom prompt
needed to establish this. Default reflection → schema/strategy cards only,
flat ~0.22 IoU. Task-tuned transmitter-registry prompt → ~0.50 mean IoU and
clear learning Δ. Cohort studies showed the same failure mode under default
reflection (schema + in-sample CASE/KL cards); population-registry override is
in-repo for the A/B. Use default elsewhere unless a task shows this pattern.

### H4 — Stronger models learn better (with KC)

**Claim:** Holding the KC system fixed, stronger models yield stronger
learning curves; weaker models still learn when the memory content is right,
but plateau lower.

**Status:** Reinforced by the BSM ICL comparison: official/high-model ICL
looks strong mainly because the model is stronger, not because ICL dominates
KC. Same weaker model + tuned KC already beats same-model ICL on BSM.
Learning still appears on weaker models when cards store the right state —
capacity modulates the ceiling. Verify with same-task KC sweeps across model
tiers.

### Verification plan

- Run knowledge_cards on the chosen tasks with matched baselines (ICL,
  stateless KC) and matched models.
- H3 treated as established via BSM; no DB custom-prompt A/B required.
- Prefer multi-seed / full schedules before locking H1–H2; treat short poker
  runs as under-powered for opponent modeling.

## Drift marker

The string `NOTICE: The live database schema or contents may have changed`
remains a hard-coded trigger for clearing or marking cards STALE. Only
`database_exploration` emits it today; other tasks are unaffected.
