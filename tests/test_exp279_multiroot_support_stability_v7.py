from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.experiments.exp279_multiroot_support_stability_v7 import (
    CANONICAL_INDICES,
    PROBE_INDICES,
    ROOT_PREFIX,
    TRAIN_BUDGETS,
    canonical_root,
    classify_budget_support,
    diagnostic_fold,
    expected_root_map,
    probe_root,
    run_exp279_multiroot_support_stability_v7_shard,
    summarize_canonical_support,
    summarize_probe_support,
    wilson_lower_bound,
)
from nolane_ai.experiments.exp279_routing_worlds import STRATA


def _fold(*, episodes: int = 8, rescues: int = 1) -> dict[str, int]:
    assert 0 <= rescues <= episodes
    return {
        "episodes": episodes,
        "branch_rescues": rescues,
        "branch_harms": 0,
        "both_success": 1 if episodes - rescues else 0,
        "both_failure": episodes - rescues - (1 if episodes - rescues else 0),
    }


def _probe(*, fold_rescues: tuple[int, int, int] = (1, 1, 1)) -> dict:
    return summarize_probe_support([_fold(rescues=value) for value in fold_rescues])


def _canonical(*, probe_fold_rescues: tuple[tuple[int, int, int], ...] | None = None) -> dict:
    values = probe_fold_rescues or ((1, 1, 1),) * 4
    return summarize_canonical_support([_probe(fold_rescues=value) for value in values])


def test_v7_root_namespaces_are_frozen_and_cross_budget_disjoint() -> None:
    assert ROOT_PREFIX == "20260912-exp279-multiroot-support-v7-dev"
    assert TRAIN_BUDGETS == (60, 120)
    assert CANONICAL_INDICES == (0, 1, 2, 3)
    assert PROBE_INDICES == (0, 1, 2, 3)
    assert canonical_root(60, 2) == (
        "20260912-exp279-multiroot-support-v7-dev::train::60::canonical::2"
    )
    assert probe_root(120, 3, 1) == (
        "20260912-exp279-multiroot-support-v7-dev::train::120::probe::3::1"
    )

    roots60 = expected_root_map(60)
    roots120 = expected_root_map(120)
    flat60 = set(roots60["canonical_roots"]) | set(roots60["probe_roots"])
    flat120 = set(roots120["canonical_roots"]) | set(roots120["probe_roots"])
    assert len(roots60["canonical_roots"]) == 4
    assert len(roots60["probe_roots"]) == 16
    assert len(flat60) == 20
    assert len(flat120) == 20
    assert flat60.isdisjoint(flat120)


@pytest.mark.parametrize(
    ("fn", "args"),
    [
        (canonical_root, (15, 0)),
        (canonical_root, (60, -1)),
        (canonical_root, (60, 4)),
        (probe_root, (120, 0, -1)),
        (probe_root, (120, 4, 0)),
        (probe_root, (999, 0, 0)),
    ],
)
def test_v7_root_helpers_reject_unregistered_identifiers(fn, args) -> None:
    with pytest.raises(ValueError):
        fn(*args)


def test_v7_diagnostic_fold_assignment_is_exact() -> None:
    assert diagnostic_fold(0) == 0
    assert diagnostic_fold(len(STRATA) - 1) == 0
    assert diagnostic_fold(len(STRATA)) == 1
    assert diagnostic_fold(2 * len(STRATA)) == 2
    assert diagnostic_fold(3 * len(STRATA)) == 0
    with pytest.raises(ValueError):
        diagnostic_fold(-1)


def test_v7_one_sided_wilson_lower_bound_matches_frozen_formula() -> None:
    assert wilson_lower_bound(0, 100) == 0.0
    assert wilson_lower_bound(1, 100) == pytest.approx(0.0022340928334547497)
    assert wilson_lower_bound(10, 100) == pytest.approx(0.06071866701544723)
    assert wilson_lower_bound(31, 1584) == pytest.approx(0.014611611807305332)

    for rescues, episodes in [(-1, 10), (11, 10), (0, 0)]:
        with pytest.raises(ValueError):
            wilson_lower_bound(rescues, episodes)


def test_v7_probe_and_canonical_support_summaries_close_counts() -> None:
    probe = _probe(fold_rescues=(1, 0, 2))
    assert probe["episodes"] == 24
    assert probe["branch_rescues"] == 3
    assert probe["nonrescue_count"] == 21
    assert probe["probe_rescue_supported"] is True
    assert probe["probe_nonrescue_supported"] is True
    assert probe["fold_support_closed"] is False
    assert probe["wilson_lower_bound"] > 0.0

    canonical = summarize_canonical_support([probe, _probe(), _probe(), _probe()])
    assert canonical["canonical_total_episodes"] == 96
    assert canonical["canonical_total_rescues"] == 12
    assert canonical["canonical_has_any_rescue"] is True
    assert canonical["canonical_all_probe_roots_supported"] is True
    assert canonical["canonical_all_folds_supported"] is False
    assert canonical["pooled_wilson_lower_bound"] > 0.0


def test_v7_budget_classification_priority_is_frozen() -> None:
    recurrent = [_canonical() for _ in range(4)]
    assert classify_budget_support(recurrent) == "SUPPORT_RECURRENT"

    intermittent = list(recurrent)
    intermittent[2] = _canonical(
        probe_fold_rescues=((1, 1, 1), (1, 0, 1), (1, 1, 1), (1, 1, 1))
    )
    assert classify_budget_support(intermittent) == "PROBE_SUPPORT_INTERMITTENT"

    collapse = list(recurrent)
    collapse[3] = _canonical(probe_fold_rescues=((0, 0, 0),) * 4)
    assert collapse[3]["canonical_total_rescues"] == 0
    assert collapse[3]["pooled_wilson_lower_bound"] == 0.0
    assert classify_budget_support(collapse) == "MODEL_ROOT_SUPPORT_COLLAPSE"


def test_v7_support_summaries_reject_nonclosing_partitions() -> None:
    bad = _fold()
    bad["both_failure"] += 1
    with pytest.raises(ValueError, match="partition"):
        summarize_probe_support([bad, _fold(), _fold()])

    with pytest.raises(ValueError, match="four"):
        summarize_canonical_support([_probe()] * 3)


def test_v7_tiny_shard_uses_augmentation_only_and_exports_sufficient_statistics() -> None:
    receipt = run_exp279_multiroot_support_stability_v7_shard(
        train_replicates=60,
        canonical_index=0,
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        route_threshold=0.5,
        probe_replicates=9,
        batch_size=2,
        timesteps=3,
        variables=4,
        constraints=2,
        noise_std=0.05,
        lr=1e-3,
        weight_decay=0.0,
        protocol_digest="p" * 64,
        code_digest="c" * 64,
    )

    assert receipt["schema"] == "NLM-EXP-279-MULTIROOT-SUPPORT-SHARD-V7"
    assert receipt["evidence_level"] == "EV-E2"
    assert receipt["decision"] == "UNVERIFIED"
    assert receipt["scientific_evidence_eligible"] is False
    assert receipt["train_replicates"] == 60
    assert receipt["canonical_index"] == 0
    assert receipt["training_rng_stream"] == "augmentation"
    assert receipt["probe_rng_stream"] == "augmentation"
    assert receipt["evaluation_rng_stream_used"] is False
    assert receipt["evaluation_targets_used"] is False
    assert receipt["raw_examples_exported"] is False
    assert receipt["raw_model_outputs_exported"] is False
    assert receipt["selector_trained"] is False
    assert receipt["cdd_distiller_trained"] is False
    assert receipt["branch_preview_selector_trained"] is False
    assert receipt["threshold_tuned"] is False
    assert receipt["fresh_evaluation_lineage_may_be_reserved"] is False
    assert receipt["fresh_evaluation_lineage_consumed"] is False
    assert receipt["confirmatory_data_consumed"] is False
    assert receipt["challenge_materialized"] is False
    assert receipt["promotion_claimed"] is False
    assert len(receipt["probe_receipts"]) == 4
    assert sum(len(item["folds"]) for item in receipt["probe_receipts"]) == 12
    assert receipt["canonical_support"]["canonical_total_episodes"] == 4 * 9 * 2
    assert len(receipt["final_canonical_digest"]) == 64
    encoded = repr(receipt).lower()
    for forbidden in (
        "stop_logits",
        "branch_logits",
        "raw_targets",
        "predictions",
        "route_scores",
    ):
        assert forbidden not in encoded


def test_v7_sample_size_reference_formula_used_by_future_cross_cell_contract() -> None:
    p_floor = 0.01
    expected = math.ceil(math.log(0.01) / math.log(1.0 - p_floor))
    assert expected == 459
