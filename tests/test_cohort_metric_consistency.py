"""cohort_studies reward is normalized to the current signed metric before ranking.

cohort_studies moved from a clamped relative skill score (reward = max(0, 1 -
mean_kl_divergence / mean_reference_kl)) to the signed information gain (reward =
mean_reference_kl - mean_kl_divergence). Reference runs recorded under the two metrics must be
compared on the same signed scale, so the leaderboard recomputes cohort reward from the stored
cohort-mean KLs in canonical_outcome_reward (mirroring the database_exploration recompute). The
fixed gpt-5.4 normalization baseline is derived through the same chokepoint
(artifact_baseline_reward -> canonical_reward_sum -> canonical_outcome_reward), so a single fix
point corrects both the per-system numerator and the reference baseline.
"""

from __future__ import annotations

from scripts.analyze_final_results import (
    artifact_baseline_reward,
    canonical_outcome_reward,
)


def _cohort_outcome(reward, ref_kl, kl):
    return {
        "reward": reward,
        "metadata": {"mean_reference_kl": ref_kl, "mean_kl_divergence": kl},
    }


def test_cohort_old_clamped_artifact_is_recomputed_signed():
    # Old kl_skill_score artifact: reward was clamped (max(0, 1 - 0.90/0.16) == 0.0), but the
    # metadata carries the KLs the signed metric needs.
    outcome = _cohort_outcome(reward=0.0, ref_kl=0.16, kl=0.90)
    got = canonical_outcome_reward("cohort_studies", outcome, database_budget=40.0)
    assert (
        abs(got - (-0.74)) < 1e-9
    )  # signed information gain 0.16 - 0.90 (negatives legitimate)


def test_cohort_new_signed_artifact_unchanged():
    # New kl_information_gain_bits artifact: reward already == ref_kl - kl; recompute reproduces it.
    outcome = _cohort_outcome(reward=-0.74, ref_kl=0.16, kl=0.90)
    got = canonical_outcome_reward("cohort_studies", outcome, database_budget=40.0)
    assert abs(got - (-0.74)) < 1e-9


def test_cohort_normalization_baseline_uses_signed_recompute():
    # The fixed gpt-5.4 normalization baseline is derived via artifact_baseline_reward ->
    # canonical_reward_sum -> canonical_outcome_reward. This is where the highest-impact effect
    # lives (the gpt-5.4 cohort baseline shifts from clamped 0.9945 to signed -0.0977, driving the
    # rank moves), so exercise the real entry point: the baseline must sum the SIGNED recompute,
    # not the clamped stored rewards.
    artifact = {
        "summary": {"task": {"name": "cohort_studies"}},
        "baseline_trace": {
            "result": {
                "instance_outcomes": [
                    _cohort_outcome(
                        reward=0.0, ref_kl=0.16, kl=0.90
                    ),  # stored 0.0 -> signed -0.74
                    _cohort_outcome(
                        reward=0.0, ref_kl=0.20, kl=0.25
                    ),  # stored 0.0 -> signed -0.05
                ]
            }
        },
    }
    baseline = artifact_baseline_reward(artifact, "cohort_studies")
    assert abs(baseline - (-0.79)) < 1e-9  # signed sum, NOT the stored 0.0 + 0.0


def test_cohort_missing_metadata_falls_back_to_stored_reward():
    got = canonical_outcome_reward(
        "cohort_studies", {"reward": 0.5}, database_budget=40.0
    )
    assert got == 0.5


def test_other_tasks_unaffected():
    assert (
        canonical_outcome_reward(
            "exploitable_poker", {"reward": 3.5}, database_budget=40.0
        )
        == 3.5
    )
    # database_exploration keeps its own regret recompute, unchanged by the cohort branch.
    assert (
        canonical_outcome_reward(
            "database_exploration",
            {"reward": 0.9, "raw_metric_value": 8},
            database_budget=40.0,
        )
        == 0.8
    )
