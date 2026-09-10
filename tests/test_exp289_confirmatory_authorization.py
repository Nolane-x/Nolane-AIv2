from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil

import pytest

pytest.importorskip("torch")

from nolane_ai.protocol.evidence import canonical_sha256
from tests.exp289_confirmatory_fixtures import (
    CODE_DIGEST,
    TEST_GEOMETRY_DIGEST,
    task5_inputs,
)


def _api():
    try:
        from nolane_ai.experiments.exp289_challenge_worlds import (
            challenge_contract_digest,
            frozen_exp289_challenge_contract,
        )
        from nolane_ai.experiments.exp289_confirmatory_authorization import (
            authorize_exp289_confirmatory_execution,
            validate_exp289_confirmatory_execution_authorization,
        )
    except ModuleNotFoundError:
        pytest.fail("EXP-289 execution authorization / challenge contract authority is missing")
    return (
        frozen_exp289_challenge_contract,
        challenge_contract_digest,
        authorize_exp289_confirmatory_execution,
        validate_exp289_confirmatory_execution_authorization,
    )


def _authorize(inputs: dict, **overrides):
    _, _, authorize, _ = _api()
    kwargs = {
        "execution_artifact": inputs["authoritative_execution"],
        "prep_artifact": inputs["prep"],
        "checkpoint_path": inputs["checkpoint_path"],
        "checkpoint_receipt": inputs["checkpoint_receipt"],
        "expected_geometry_digest": inputs["expected_geometry_digest"],
        "expected_geometry_configuration": inputs["expected_geometry_configuration"],
        "execution_code_digest": inputs["execution_code_digest"],
    }
    kwargs.update(overrides)
    return authorize(**kwargs)


def test_exp289_challenge_contract_freezes_rder_family_without_materializing_entropy() -> None:
    frozen_contract, digest_fn, _, _ = _api()
    contract = frozen_contract()
    assert contract["schema"] == "NLM-EXP-289-POST-FREEZE-CHALLENGE-CONTRACT-V1"
    assert contract["experiment_id"] == "EXP-289"
    assert contract["lane"] == "POST_FREEZE_CHALLENGE"
    assert contract["arms"] == ["no_nogood", "local_nogood"]
    assert contract["primary_endpoint"] == "repeat_dead_end_rate"
    assert contract["primary_direction"] == "lower"
    assert contract["mesi_relative_reduction"] == 0.25
    assert contract["paired"] is True
    assert contract["zero_opportunity_policy"] == "retain_raw_episode_exclude_from_rder_denominator"
    assert contract["epsilon_denominator_rescue"] is False
    assert contract["protected_overprune_ceiling"] == 0.005
    assert contract["protected_solution_floor_delta"] == -0.01
    assert contract["memory_scope"] == "episode_local"
    assert contract["cross_episode_reuse"] is False
    assert contract["cross_problem_reuse"] is False
    assert contract["exact_subset_matching_only"] is True
    assert contract["seed_authority"] == "FUTURE_PUBLIC_BEACON_AFTER_IMMUTABLE_GATE_A_FREEZE"
    rendered = repr(contract).lower()
    assert "entropy_hex" not in rendered
    assert "challenge_seed" not in rendered
    assert "confirmatory_observations" not in rendered
    assert digest_fn() == canonical_sha256(contract)


def test_exp289_execution_authorization_binds_checkpoint_and_all_pre_beacon_lineage(task5_inputs: dict) -> None:
    _, challenge_digest, _, validate = _api()
    authorization = _authorize(task5_inputs)
    receipt = task5_inputs["checkpoint_receipt"]
    execution = task5_inputs["authoritative_execution"]
    prep = task5_inputs["prep"]

    assert authorization["schema"] == "NLM-EXP-289-CONFIRMATORY-EXECUTION-AUTHORIZATION-V1"
    assert authorization["status"] == "AUTHORIZED_NOT_EXECUTED"
    assert authorization["evidence_level"] == "EV-E2"
    assert authorization["decision"] == "UNVERIFIED"
    assert authorization["confirmatory_data_consumed"] is False
    assert authorization["seed_materialization_status"] == "NOT_EXECUTED"
    assert authorization["challenge_materialized"] is False
    assert 32 <= authorization["confirmatory_n"] <= 128
    assert authorization["confirmatory_n"] == prep["sample_size_freeze"]["confirmatory_n"]
    assert authorization["preflight"]["all_checks_passed"] is True

    lineage = authorization["lineage"]
    assert lineage["protocol_digest"] == task5_inputs["protocol_digest"]
    assert lineage["development_geometry_digest"] == TEST_GEOMETRY_DIGEST
    assert lineage["development_execution_digest"] == execution["artifact_digest"]
    assert lineage["scientific_execution_digest"] == execution["development_geometry_authority"]["scientific_execution_digest"]
    assert lineage["prep_digest"] == prep["prep_digest"]
    assert lineage["execution_code_digest"] == CODE_DIGEST
    assert lineage["challenge_contract_digest"] == challenge_digest()
    assert lineage["checkpoint_receipt_digest"] == receipt["receipt_digest"]
    assert lineage["checkpoint_scientific_identity_digest"] == receipt["scientific_identity_digest"]
    assert lineage["checkpoint_file_sha256"] == receipt["checkpoint_file_sha256"]
    assert lineage["checkpoint_execution_contract_digest"] == receipt["execution_contract_digest"]
    assert lineage["checkpoint_no_nogood_final_digest"] == receipt["no_nogood_final_digest"]
    assert lineage["checkpoint_local_nogood_final_digest"] == receipt["local_nogood_final_digest"]

    rendered = repr(authorization).lower()
    for forbidden in ("beacon_receipt", "entropy_hex", "challenge_seed", "confirmatory_observations"):
        assert forbidden not in rendered
    assert validate(authorization) == []


def test_exp289_execution_authorization_rejects_unenveloped_smoke_even_if_prep_is_ready(task5_inputs: dict) -> None:
    with pytest.raises(ValueError, match="authoritative|geometry"):
        _authorize(
            task5_inputs,
            execution_artifact=task5_inputs["scientific_execution"],
        )


def test_exp289_execution_authorization_rejects_geometry_code_and_prep_drift(task5_inputs: dict) -> None:
    wrong_geometry = deepcopy(task5_inputs["expected_geometry_configuration"])
    wrong_geometry["variables"] = int(wrong_geometry["variables"]) + 1
    with pytest.raises(ValueError, match="geometry"):
        _authorize(task5_inputs, expected_geometry_configuration=wrong_geometry)

    with pytest.raises(ValueError, match="code"):
        _authorize(task5_inputs, execution_code_digest="b" * 64)

    changed_prep = deepcopy(task5_inputs["prep"])
    changed_prep["sample_size_freeze"]["confirmatory_n"] += 1
    changed_prep["prep_digest"] = "0" * 64
    with pytest.raises(ValueError, match="prep|sample|digest"):
        _authorize(task5_inputs, prep_artifact=changed_prep)


def test_exp289_execution_authorization_rejects_checkpoint_identity_tamper(task5_inputs: dict) -> None:
    receipt = deepcopy(task5_inputs["checkpoint_receipt"])
    receipt["scientific_identity_digest"] = "0" * 64
    clean = deepcopy(receipt)
    clean.pop("receipt_digest", None)
    receipt["receipt_digest"] = canonical_sha256(clean)
    with pytest.raises(ValueError, match="checkpoint|scientific identity"):
        _authorize(task5_inputs, checkpoint_receipt=receipt)

    receipt = deepcopy(task5_inputs["checkpoint_receipt"])
    receipt["execution_contract_digest"] = "0" * 64
    clean = deepcopy(receipt)
    clean.pop("receipt_digest", None)
    receipt["receipt_digest"] = canonical_sha256(clean)
    with pytest.raises(ValueError, match="checkpoint|contract"):
        _authorize(task5_inputs, checkpoint_receipt=receipt)


def test_exp289_execution_authorization_rejects_checkpoint_file_tamper(task5_inputs: dict, tmp_path: Path) -> None:
    tampered_path = tmp_path / "tampered-trained.pt"
    shutil.copy2(task5_inputs["checkpoint_path"], tampered_path)
    with tampered_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="checkpoint file|sha256|hash"):
        _authorize(task5_inputs, checkpoint_path=tampered_path)
