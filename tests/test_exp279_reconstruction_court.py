from __future__ import annotations

from copy import deepcopy

import pytest

pytest.importorskip("torch")


STRATA = ["PROPAGATION_FIT", "BRANCH_FIT", "MIXED_RESIDUAL"]


def _reconstruction(tmp_path):
    from nolane_ai.experiments.exp279_reconstruction_court import (
        build_exp279_reconstruction_authorization,
    )
    from tests.test_exp279_confirmatory_authorization import _build

    _, _, _, _, authorization, seal = _build(tmp_path)
    reconstruction = build_exp279_reconstruction_authorization(
        seal=seal,
        reconstruction_code_digest=authorization["machinery_digests"][
            "reconstruction_code_digest"
        ],
    )
    return authorization, seal, reconstruction


def test_exp279_reconstruction_authorization_binds_gate_a_without_materializing_challenge(tmp_path) -> None:
    from nolane_ai.experiments.exp279_reconstruction_court import (
        validate_exp279_reconstruction_authorization,
    )

    authorization, seal, reconstruction = _reconstruction(tmp_path)

    assert reconstruction["schema"] == "NLM-EXP-279-CONFIRMATORY-RECONSTRUCTION-AUTH-V1"
    assert reconstruction["status"] == "RECONSTRUCTION_AUTHORIZED_NOT_EXECUTED"
    assert reconstruction["evidence_level"] == "EV-E2"
    assert reconstruction["decision"] == "UNVERIFIED"
    assert reconstruction["confirmatory_ready"] is False
    assert reconstruction["confirmatory_data_consumed"] is False
    assert reconstruction["challenge_materialized"] is False
    assert reconstruction["seed_materialization_status"] == "NOT_EXECUTED"
    assert reconstruction["decision_rule_executed"] is False
    assert reconstruction["seal_digest"] == seal["seal_digest"]
    assert reconstruction["protocol_digest"] == authorization["protocol_digest"]
    assert reconstruction["source_tree_digest"] == authorization["source_tree_digest"]
    assert reconstruction["confirmatory_n"] == authorization["confirmatory_n"]
    assert reconstruction["reserved_replicate_ids"] == authorization["reserved_replicate_ids"]
    assert reconstruction["stratum_schedule"] == STRATA
    assert reconstruction["route_threshold"] == pytest.approx(authorization["route_threshold"])
    assert validate_exp279_reconstruction_authorization(reconstruction) == []

    rendered = repr(reconstruction).lower()
    assert "beacon_receipt" not in rendered
    assert "challenge_seed" not in rendered
    assert "challenge_batch" not in rendered


def test_exp279_reconstruction_binds_checkpoint_route_dependent_cost_and_information_court(tmp_path) -> None:
    authorization, seal, reconstruction = _reconstruction(tmp_path)
    checkpoint = authorization["checkpoint"]

    assert reconstruction["seal_snapshot_digest"]
    assert reconstruction["seal_snapshot"] == seal
    assert reconstruction["checkpoint_scientific_identity_digest"] == checkpoint[
        "scientific_identity_digest"
    ]
    assert reconstruction["checkpoint_receipt_digest"] == checkpoint["receipt_digest"]
    assert reconstruction["checkpoint_replay_contract_digest"] == checkpoint[
        "replay_contract_digest"
    ]
    assert reconstruction["checkpoint_final_state_digest"] == checkpoint[
        "final_state_digest"
    ]
    assert reconstruction["checkpoint_file_sha256"] == checkpoint[
        "checkpoint_file_sha256"
    ]
    assert reconstruction["accounted_flops_contract"] == authorization[
        "resource_court"
    ]["accounted_flops_contract"]
    assert reconstruction["model_geometry"] == authorization["model_geometry"]
    assert reconstruction["information_separation"] == authorization[
        "information_separation"
    ]
    assert reconstruction["reconstruction_code_digest"] == authorization[
        "machinery_digests"
    ]["reconstruction_code_digest"]


def test_exp279_reconstruction_rejects_wrong_machinery_digest(tmp_path) -> None:
    from nolane_ai.experiments.exp279_reconstruction_court import (
        build_exp279_reconstruction_authorization,
    )
    from tests.test_exp279_confirmatory_authorization import _build

    _, _, _, _, _, seal = _build(tmp_path)
    with pytest.raises(ValueError, match="machinery|code digest"):
        build_exp279_reconstruction_authorization(
            seal=seal,
            reconstruction_code_digest="9" * 64,
        )


def test_exp279_reconstruction_validator_rejects_rehashed_route_strata_and_cost_tamper(tmp_path) -> None:
    from nolane_ai.experiments.exp279_reconstruction_court import (
        _binding_digest,
        _reconstruction_digest,
        validate_exp279_reconstruction_authorization,
    )

    _, _, reconstruction = _reconstruction(tmp_path)

    bad = deepcopy(reconstruction)
    bad["route_threshold"] = 0.25
    bad["binding_digest"] = _binding_digest(bad)
    bad["reconstruction_digest"] = _reconstruction_digest(bad)
    assert any("route" in error.lower() for error in validate_exp279_reconstruction_authorization(bad))

    bad = deepcopy(reconstruction)
    bad["stratum_schedule"] = list(reversed(STRATA))
    bad["binding_digest"] = _binding_digest(bad)
    bad["reconstruction_digest"] = _reconstruction_digest(bad)
    assert any("stratum" in error.lower() for error in validate_exp279_reconstruction_authorization(bad))

    bad = deepcopy(reconstruction)
    bad["accounted_flops_contract"]["hybrid"]["stop_accounted_flops_per_episode"] += 1
    bad["binding_digest"] = _binding_digest(bad)
    bad["reconstruction_digest"] = _reconstruction_digest(bad)
    assert any("flop" in error.lower() or "cost" in error.lower() for error in validate_exp279_reconstruction_authorization(bad))
