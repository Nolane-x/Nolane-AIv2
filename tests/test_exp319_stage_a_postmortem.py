from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from nolane_ai.experiments.exp319_contract import canonical_json_bytes
from nolane_ai.experiments.exp319_stage_a_postmortem import (
    EFFORTS,
    POSTMORTEM_SCHEMA,
    EffortMetric,
    seal_postmortem,
    validate_selection_binding,
)


def _digest(char: str) -> str:
    return char * 64


def _receipt(**overrides):
    values = {
        "stage": "A_SANITY",
        "root": 0,
        "cumulative_step": 1024,
        "source_commit_sha": "2" * 40,
        "run_identity": "github-run-35311529822-exp319-v1",
        "arm_id": "A_FIXED",
        "learning_rate": 0.0001,
        "artifact_digest": _digest("a"),
        "training_contract_digest": _digest("b"),
        "model_state_digest": _digest("c"),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _selection(receipt=None):
    receipt = receipt or _receipt()
    payload = {
        "schema": "EXP319-STAGE-A-SELECTION-AUTHORITY-V1",
        "selections": [
            {
                "arm_id": receipt.arm_id,
                "passes_floor": False,
                "selected_learning_rate": receipt.learning_rate,
                "selected_record": {
                    "arm_id": receipt.arm_id,
                    "root": 0,
                    "step": 1024,
                    "learning_rate": receipt.learning_rate,
                    "answer_only_loss": 2.3,
                    "answer_token_accuracy": 0.77,
                    "exact_match": 0.28,
                    "eos_correctness": 1.0,
                    "invalid_output_rate": 0.0,
                    "nonfinite_events": 0,
                    "initial_answer_only_loss": 343.0,
                },
                "selection_digest": _digest("d"),
            }
        ],
    }
    payload["authority_digest"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def _metric(
    effort: int,
    *,
    exact: float,
    token: float,
    loss: float,
) -> EffortMetric:
    return EffortMetric(
        effort=effort,
        answer_only_loss=loss,
        answer_token_accuracy=token,
        exact_match=exact,
        family_balanced_exact_match=exact,
        family_exact=(
            ("algorithmic-sequence-transform", exact),
            ("generator-heldout-abstract-transformation", exact),
            ("iterative-grid-and-maze", exact),
            ("language-sequence-control", exact),
        ),
        eos_correctness=1.0,
        invalid_output_rate=0.0,
        nonzero_exact_families=4 if exact > 0.0 else 0,
    )


def test_selection_binding_reproduces_authority_and_selected_checkpoint() -> None:
    receipt = _receipt()
    selection = _selection(receipt)

    selected = validate_selection_binding(
        selection,
        receipt,
        expected_authority_digest=selection["authority_digest"],
        expected_source_commit_sha=receipt.source_commit_sha,
        expected_run_identity=receipt.run_identity,
    )

    assert selected["arm_id"] == "A_FIXED"
    assert selected["selected_learning_rate"] == 0.0001
    assert selected["selected_record"]["step"] == 1024


def test_selection_binding_rejects_wrong_selected_learning_rate() -> None:
    receipt = _receipt()
    selection = _selection(receipt)
    selection["selections"][0]["selected_learning_rate"] = 0.0003
    selection["authority_digest"] = hashlib.sha256(
        canonical_json_bytes(
            {
                "schema": selection["schema"],
                "selections": selection["selections"],
            }
        )
    ).hexdigest()

    with pytest.raises(ValueError, match="selected LR|learning rate"):
        validate_selection_binding(
            selection,
            receipt,
            expected_authority_digest=selection["authority_digest"],
            expected_source_commit_sha=receipt.source_commit_sha,
            expected_run_identity=receipt.run_identity,
        )


def test_seal_postmortem_is_deterministic_and_uses_effort4_as_gate() -> None:
    receipt = _receipt()
    metrics = (
        _metric(1, exact=0.50, token=0.82, loss=1.8),
        _metric(2, exact=0.40, token=0.80, loss=1.9),
        _metric(4, exact=0.28, token=0.77, loss=2.3),
        _metric(8, exact=0.35, token=0.79, loss=2.0),
    )

    first = seal_postmortem(
        scientific_run_id=35311529822,
        selection_authority_digest=_digest("e"),
        receipt=receipt,
        metrics=metrics,
        model_state_digest_before=_digest("c"),
        model_state_digest_after=_digest("c"),
    )
    second = seal_postmortem(
        scientific_run_id=35311529822,
        selection_authority_digest=_digest("e"),
        receipt=receipt,
        metrics=tuple(reversed(metrics)),
        model_state_digest_before=_digest("c"),
        model_state_digest_after=_digest("c"),
    )

    assert first.schema == POSTMORTEM_SCHEMA
    assert first.gate_effort == 4
    assert first.highest_exact_effort == 1
    assert tuple(item.effort for item in first.efforts) == EFFORTS
    assert first.evidence_digest == second.evidence_digest
    assert dict(first.exact_match_delta_from_gate)[4] == 0.0
    assert dict(first.exact_match_delta_from_gate)[1] == pytest.approx(0.22)


def test_seal_postmortem_rejects_any_model_state_mutation() -> None:
    receipt = _receipt()
    metrics = tuple(
        _metric(effort, exact=0.25, token=0.75, loss=2.0)
        for effort in EFFORTS
    )

    with pytest.raises(ValueError, match="mutated model state"):
        seal_postmortem(
            scientific_run_id=35311529822,
            selection_authority_digest=_digest("e"),
            receipt=receipt,
            metrics=metrics,
            model_state_digest_before=_digest("c"),
            model_state_digest_after=_digest("f"),
        )


def test_postmortem_output_contains_no_authorization_or_replacement_fields() -> None:
    receipt = _receipt()
    report = seal_postmortem(
        scientific_run_id=35311529822,
        selection_authority_digest=_digest("e"),
        receipt=receipt,
        metrics=tuple(
            _metric(effort, exact=0.25, token=0.75, loss=2.0)
            for effort in EFFORTS
        ),
        model_state_digest_before=_digest("c"),
        model_state_digest_after=_digest("c"),
    )

    keys = set(report.to_json_dict())
    assert "authorization" not in " ".join(keys).lower()
    assert "replacement_disposition" not in keys
    assert "pass" not in keys
