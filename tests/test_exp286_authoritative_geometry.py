from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = ROOT / "protocols" / "exp286_development_geometry_v1.json"
GEOMETRY_DIGEST = ROOT / "protocols" / "exp286_development_geometry_v1.sha256"
STAGE_A_DIGEST = ROOT / "protocols" / "stage_a_v1.sha256"


def test_exp286_authoritative_development_geometry_is_predeclared_and_digest_locked() -> None:
    payload = json.loads(GEOMETRY.read_text(encoding="utf-8"))

    assert payload == {
        "schema": "NLM-EXP-286-DEVELOPMENT-GEOMETRY-V1",
        "experiment_id": "EXP-286",
        "authority_scope": "DEVELOPMENT_PILOT_ONLY",
        "protocol_digest": STAGE_A_DIGEST.read_text(encoding="utf-8").strip(),
        "predeclaration": "BEFORE_AUTHORITATIVE_DEVELOPMENT_RUN",
        "confirmatory_authority": False,
        "challenge_materialized": False,
        "pilot_reuse_as_confirmatory": False,
        "runner": "scripts/run_exp286_paired_dev.py",
        "geometry": {
            "tiny": False,
            "root_seed": "20260906-exp286-paired-dev",
            "d_model": 64,
            "hidden_size": 48,
            "target_parameters": 500000,
            "train_replicates": 16,
            "eval_replicates": 32,
            "eval_start_replicate": 30000,
            "batch_size": 8,
            "timesteps": 4,
            "variables": 8,
            "decoys": 3,
            "max_search_steps": 16,
            "noise_std": 0.05,
            "lr": 0.002,
            "weight_decay": 0.0,
            "max_accounted_flops_per_episode": None,
        },
    }

    actual_digest = hashlib.sha256(GEOMETRY.read_bytes()).hexdigest()
    assert GEOMETRY_DIGEST.read_text(encoding="utf-8").strip() == actual_digest


def test_exp286_authoritative_geometry_cannot_be_mistaken_for_confirmatory_entropy() -> None:
    payload = json.loads(GEOMETRY.read_text(encoding="utf-8"))

    assert payload["authority_scope"] == "DEVELOPMENT_PILOT_ONLY"
    assert payload["confirmatory_authority"] is False
    assert payload["challenge_materialized"] is False
    assert payload["pilot_reuse_as_confirmatory"] is False
    assert "beacon" not in payload
    assert "challenge_seed" not in payload
    assert payload["geometry"]["eval_replicates"] >= 32
