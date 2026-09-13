from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")


STAGE_A_DIGEST = "c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440"


def _tiny_geometry() -> dict[str, object]:
    return {
        "batch_size": 2,
        "canonical_indices": [0, 1, 2, 3],
        "d_model": 64,
        "decoys": 3,
        "eval_replicates": 1,
        "eval_start_replicate": 40000,
        "hidden_size": 48,
        "lr": 0.002,
        "max_search_steps": 24,
        "noise_std": 0.05,
        "restarts": 4,
        "root_prefix": "20260913-exp290-structural-clause-transfer-v1-dev",
        "target_parameters": 500000,
        "timesteps": 4,
        "train_replicates": 1,
        "variables": 8,
        "weight_decay": 0.0,
    }


def test_exp290_production_receipt_publishes_validator_episode_counts() -> None:
    from nolane_ai.experiments.exp290_receipts import seal_root_receipt, validate_root_receipt
    from nolane_ai.experiments.exp290_structural_clause_transfer import run_exp290_root

    geometry = _tiny_geometry()
    result = run_exp290_root(
        canonical_index=0,
        geometry=geometry,
        protocol_digest=STAGE_A_DIGEST,
        geometry_digest="0" * 64,
        code_digest="1" * 64,
    )

    evaluation = result["evaluation"]
    expected_episode_count = int(geometry["eval_replicates"]) * int(geometry["batch_size"])
    assert evaluation["evaluation_episode_count"] == expected_episode_count
    assert evaluation["heldout_nonidentity_episode_count"] == expected_episode_count
    for mode in evaluation["modes"].values():
        assert mode["episodes"] == evaluation["evaluation_episode_count"]
        assert len(mode["per_episode"]) == evaluation["evaluation_episode_count"]

    # Match the publication layer in scripts/run_exp290_structural_clause_transfer_dev.py:
    # repository identity is bound after the scientific runner returns and before sealing.
    result["repository_head"] = "f" * 40
    sealed = seal_root_receipt(result)
    validate_root_receipt(sealed)
