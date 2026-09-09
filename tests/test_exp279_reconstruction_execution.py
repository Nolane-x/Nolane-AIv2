from __future__ import annotations

from copy import deepcopy

import pytest

pytest.importorskip("torch")


STRATA = ["PROPAGATION_FIT", "BRANCH_FIT", "MIXED_RESIDUAL"]
BEACON_TIME = "2026-09-09T12:01:00Z"


def _fixture(tmp_path):
    from nolane_ai.experiments.exp279_beacon import build_test_beacon_receipt
    from nolane_ai.experiments.exp279_reconstruction_court import (
        build_exp279_reconstruction_authorization,
    )
    from tests.test_exp279_confirmatory_authorization import _build

    execution, registry, prep, checkpoint, authorization, seal = _build(tmp_path)
    reconstruction = build_exp279_reconstruction_authorization(
        seal=seal,
        reconstruction_code_digest=authorization["machinery_digests"][
            "reconstruction_code_digest"
        ],
    )
    beacon = build_test_beacon_receipt(
        source="synthetic-exp279-reconstruction-test",
        beacon_id="test-round-279-reconstruction",
        published_at_utc=BEACON_TIME,
        entropy_hex="73" * 32,
        evidence_reference="test-only://exp279-reconstruction",
    )
    return {
        "execution": execution,
        "registry": registry,
        "prep": prep,
        "checkpoint_receipt": checkpoint,
        "checkpoint_path": tmp_path / "exp279-checkpoint.pt",
        "authorization": authorization,
        "seal": seal,
        "reconstruction": reconstruction,
        "beacon": beacon,
    }


def _reconstruct(fixture, replicate=None):
    from nolane_ai.experiments.exp279_reconstruction_court import (
        reconstruct_exp279_expected_row,
    )

    reconstruction = fixture["reconstruction"]
    selected = (
        reconstruction["reserved_replicate_ids"][0]
        if replicate is None
        else replicate
    )
    return reconstruct_exp279_expected_row(
        reconstruction_authorization=reconstruction,
        seal=fixture["seal"],
        beacon_receipt=fixture["beacon"],
        checkpoint_path=fixture["checkpoint_path"],
        checkpoint_receipt=fixture["checkpoint_receipt"],
        replicate=selected,
    )


def test_exp279_reconstruction_replays_sealed_seed_stratum_and_checkpoint_without_writes(tmp_path) -> None:
    from nolane_ai.experiments.exp279_beacon import derive_exp279_challenge_seed
    from nolane_ai.protocol.identity import file_sha256

    fixture = _fixture(tmp_path)
    reconstruction = fixture["reconstruction"]
    replicate = reconstruction["reserved_replicate_ids"][0]
    checkpoint_sha_before = file_sha256(fixture["checkpoint_path"])

    row = _reconstruct(fixture, replicate=replicate)
    assert row["schema"] == "NLM-EXP-279-CONFIRMATORY-RAW-ROW-V1"
    assert row["experiment_id"] == "EXP-279"
    assert row["replicate"] == replicate
    assert row["replicate_ordinal"] == 0
    assert row["stratum"] == STRATA[0]
    expected_seed = derive_exp279_challenge_seed(
        protocol_digest=reconstruction["protocol_digest"],
        beacon_receipt=fixture["beacon"],
        stream="challenge",
        replicate=replicate,
        stratum=STRATA[0],
        freeze_commit_timestamp_utc=reconstruction["freeze_commit_timestamp_utc"],
        checkpoint_seal_created_at_utc=reconstruction[
            "checkpoint_seal_created_at_utc"
        ],
    )
    assert row["challenge_seed"] == expected_seed
    assert row["challenge_digest"]
    assert row["checkpoint_functional_state_before"] == row[
        "checkpoint_functional_state_after"
    ]
    assert row["checkpoint_functional_state_before"] == {
        "propagation_only": fixture["checkpoint_receipt"]["final_state"][
            "propagation_only_digest"
        ],
        "branch_only": fixture["checkpoint_receipt"]["final_state"][
            "branch_only_digest"
        ],
        "hybrid": fixture["checkpoint_receipt"]["final_state"]["hybrid_digest"],
    }
    assert file_sha256(fixture["checkpoint_path"]) == checkpoint_sha_before


def test_exp279_reconstruction_binds_information_surface_route_receipt_and_path_dependent_cost(tmp_path) -> None:
    fixture = _fixture(tmp_path)
    row = _reconstruct(fixture)
    costs = fixture["reconstruction"]["accounted_flops_contract"]

    assert row["information_separation"] == fixture["reconstruction"][
        "information_separation"
    ]
    assert row["arm_input_receipt"] == {
        "same_surface_events": True,
        "same_variable_states": True,
        "propagation_only_received_incidence": True,
        "branch_only_received_incidence": False,
        "hybrid_received_incidence": True,
        "evaluator_targets_withheld_from_arms": True,
    }
    assert row["propagation_only"]["accounted_flops_per_episode"] == costs[
        "propagation_only"
    ]["accounted_flops_per_episode"]
    assert row["branch_only"]["accounted_flops_per_episode"] == costs[
        "branch_only"
    ]["accounted_flops_per_episode"]

    route = row["hybrid_route_receipt"]
    assert route["threshold"] == pytest.approx(
        fixture["reconstruction"]["route_threshold"]
    )
    assert route["routed_episodes"] == sum(bool(item) for item in route["branch_route_mask"])
    assert route["route_fraction"] == pytest.approx(
        route["routed_episodes"] / route["batch_size"]
    )
    hybrid_cost = costs["hybrid"]
    expected = hybrid_cost["stop_accounted_flops_per_episode"] + route[
        "route_fraction"
    ] * (
        hybrid_cost["branch_accounted_flops_per_episode"]
        - hybrid_cost["stop_accounted_flops_per_episode"]
    )
    assert row["hybrid"]["accounted_flops_per_episode"] == pytest.approx(expected)
    assert route["charged_accounted_flops_per_episode"] == pytest.approx(expected)
    for arm in ("propagation_only", "branch_only", "hybrid"):
        assert row[arm]["analytical_cost_receipt"]["hardware_profiler_flops_claimed"] is False


def test_exp279_reconstruction_row_validator_detects_rehashed_seed_stratum_cost_and_metric_tamper(tmp_path) -> None:
    from nolane_ai.experiments.exp279_reconstruction_court import (
        _row_digest,
        validate_exp279_raw_row_against_reconstruction,
    )

    fixture = _fixture(tmp_path)
    original = _reconstruct(fixture)

    for mutation, needle in (
        (lambda row: row.__setitem__("challenge_seed", row["challenge_seed"] + 1), "seed"),
        (lambda row: row.__setitem__("stratum", STRATA[1]), "stratum"),
        (
            lambda row: row["hybrid"].__setitem__(
                "accounted_flops_per_episode",
                float(row["hybrid"]["accounted_flops_per_episode"]) + 1.0,
            ),
            "flop",
        ),
        (
            lambda row: row["propagation_only"].__setitem__(
                "verified_solution_rate", 0.123456
            ),
            "solution",
        ),
    ):
        bad = deepcopy(original)
        mutation(bad)
        bad["row_digest"] = _row_digest(bad)
        errors = validate_exp279_raw_row_against_reconstruction(
            raw_row=bad,
            reconstruction_authorization=fixture["reconstruction"],
            seal=fixture["seal"],
            beacon_receipt=fixture["beacon"],
            checkpoint_path=fixture["checkpoint_path"],
            checkpoint_receipt=fixture["checkpoint_receipt"],
        )
        assert any(needle in error.lower() for error in errors), errors


def test_exp279_reconstruction_refuses_replica_outside_sealed_reserved_lineage(tmp_path) -> None:
    fixture = _fixture(tmp_path)
    reconstruction = fixture["reconstruction"]
    outside = reconstruction["reserved_replicate_ids"][-1] + 1
    with pytest.raises(ValueError, match="reserved|lineage"):
        _reconstruct(fixture, replicate=outside)
