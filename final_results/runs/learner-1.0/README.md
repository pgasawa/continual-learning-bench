# Results: Learner 1.0 (LearnerLabs.ai) on database exploration, cohort studies and sales prediction

Results for **Learner 1.0**, a foundation model from Learner Labs that learns continually through its weights. **Mean normalized reward: 0.192. Mean normalized gain: 0.229.**

## Learning purely through weights

Learner 1.0 is the only system on the leaderboard, and among the open results submissions, that learns purely through its weights. Every other entry carries what it learns in its context, a memory store, a notepad or agent files. Learner 1.0 keeps no notes, memory or transcript between task instances. Its context is reset before every instance, so everything it learns on this benchmark is carried by its weights.

## Learner 1.0

Learner 1.0 is a new foundation model from [Learner Labs](https://learnerlabs.ai) that learns continually. It learns incrementally from a stream of data, learns efficiently from few examples, and retains everything it learned before along with its base capabilities.

- **Continual and incremental.** One example, one update, in the order the data arrives, with no replay and a fixed number of trainable parameters.
- **Efficient.** On a stream of four text domains it acquired each domain in under 10% of the training steps a rank-matched LoRA needed.
- **Retains prior learning and base capabilities.** Taught ten unrelated skills in sequence with 63,282 single-example updates, it retained every skill it acquired. A 16,481-item benchmark battery covering MMLU, ARC-Challenge, HellaSwag and WinoGrande held constant at every checkpoint.

Reports, training data and every recorded answer are at [learnerlabs.ai/demos](https://learnerlabs.ai/demos).

## Benchmark Results

| Task | Normalized reward | Normalized gain | Runs |
|---|---:|---:|---:|
| sales_prediction | 0.444 | 0.430 | 1 |
| database_exploration | 0.130 | 0.214 | 3 |
| cohort_studies | 0.002 | 0.044 | 3 |

| Task | Reward (mean) | Stateless baseline | Gain (mean) |
|---|---:|---:|---:|
| sales_prediction | 8.4398 | 5.7572 | 2.6826 |
| database_exploration | 10.0222 | 1.8667 | 8.1556 |
| cohort_studies | -0.0894 | -0.2408 | 0.1514 |

On database exploration the three runs answer 21, 15 and 16 of 40 questions correctly. The stateless baseline answers 7.

## The system

- After every instance, a teacher LLM writes training examples from the states the learner actually reached, and the learner trains on each example once.
- The teacher sees the benchmark's released feedback, the learner's earlier trajectories and the procedures it wrote before.
- The stateless baseline is the same model with learning off.

## Models

Learner 1.0 (Learner Labs), served through the Learner Labs API. Training examples are written by a teacher LLM.

## Evaluation

Runs use the benchmark's `run_single` and `run_baseline` at upstream commit `5f8c50e`, with instance orders from the default schedule for each run index. Scores come from the benchmark's own scorers.

Normalized reward = (reward − gpt-5.4 stateless baseline) / (reference max − that baseline). Normalized gain = gain / (reference max − our stateless baseline).

## Artifacts

[learnerlabs-ai/replications/clbench](https://github.com/learnerlabs-ai/replications/tree/main/clbench) holds every trajectory, per-instance outcome and training example behind these scores.

## Evaluation dates

- Database exploration: 2026-10-08
- Cohort studies: 2026-10-06 to 2026-10-07
- Sales prediction: 2026-10-06 to 2026-10-07
