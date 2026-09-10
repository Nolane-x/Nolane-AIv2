from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp289_checkpoint import build_exp289_trained_checkpoint
from nolane_ai.experiments.exp289_confirmatory_prep import (
    _prep_digest,
    build_exp289_confirmatory_prep,
    validate_exp289_confirmatory_prep,
)
from nolane_ai.experiments.exp289_paired_runner import (
    _artifact_digest,
    run_exp289_paired_development,
)
from nolane_ai.experiments.neural_arm_registry import build_neural_arm_registry
from nolane_ai.model.audit import audit_model
from nolane_ai.model.config import NLMConfig
from nolane_ai.model.nlm import NolaneLivingModel


ROOT = Path(__file__).resolve().parents[1]
CODE_DIGEST = "a" * 64
TEST_GEOMETRY_DIGEST = "9" * 64
FREEZE_SHA = "1" * 40
FREEZE_TIME = "2026-09-10T10:45:00Z"


def _protocol() -> dict:
    return json.loads((ROOT / "protocols" / "stage_a_v1.json").read_text(encoding="utf-8"))


def _protocol_digest() -> str:
    return (ROOT / "protocols" / "stage_a_v1.sha256").read_text(encoding="utf-8").strip()


def _exp289(protocol: dict) -> dict:
    return next(item for item in protocol["experiments"] if item["experiment_id"] == "EXP-289")


def _model_audit():
    return audit_model(NolaneLivingModel(NLMConfig.stage_a_pilot_16m(), device="meta"))


def build_task5_inputs(tmp_path: Path) -> dict:
    protocol = _protocol()
    protocol_digest = _protocol_digest()
    execution = run_exp289_paired_development(
        root_seed="exp289-task5-test-only-authority",
        d_model=8,
        hidden_size=6,
        target_parameters=5_000,
        train_replicates=2,
        eval_replicates=32,
        eval_start_replicate=900,
        batch_size=2,
        timesteps=3,
        restarts=3,
        variables=6,
        decoys=2,
        max_search_steps=12,
        noise_std=0.05,
        lr=1e-3,
        weight_decay=0.0,
        protocol_digest=protocol_digest,
        code_digest=CODE_DIGEST,
    )
    scientific_execution = deepcopy(execution)
    scientific_digest = execution["artifact_digest"]
    registry = build_neural_arm_registry(
        protocol=protocol,
        protocol_digest=protocol_digest,
        model_audit=_model_audit(),
        exp289_pair_audit=execution["resource_match"]["pair_audit"],
        exp289_execution_artifact=execution,
    )
    prep = build_exp289_confirmatory_prep(
        experiment=_exp289(protocol),
        execution_artifact=execution,
        arm_registry=registry,
        analysis_code_digest=CODE_DIGEST,
        familywise_alpha=protocol["global_sample_size_plan"]["familywise_alpha"],
    )
    assert prep["status"] == "CONFIRMATORY_GATE_A_PREPARED"

    authority = {
        "schema": "NLM-EXP-289-AUTHORITATIVE-DEVELOPMENT-V1",
        "authority_scope": "DEVELOPMENT_PILOT_ONLY",
        "manifest_digest": TEST_GEOMETRY_DIGEST,
        "scientific_execution_digest": scientific_digest,
        "confirmatory_authority": False,
    }
    authoritative_execution = deepcopy(execution)
    authoritative_execution["development_geometry_authority"] = deepcopy(authority)
    authoritative_execution["artifact_digest"] = _artifact_digest(authoritative_execution)

    geometry_binding = deepcopy(authority)
    geometry_binding["authoritative_execution_digest"] = authoritative_execution["artifact_digest"]
    geometry_binding["geometry_configuration_match"] = True
    prep["development_geometry_authority"] = geometry_binding
    prep["prep_digest"] = _prep_digest(prep)
    assert validate_exp289_confirmatory_prep(prep) == []

    checkpoint_path = tmp_path / "exp289-task5-trained.pt"
    checkpoint_receipt = build_exp289_trained_checkpoint(
        execution_artifact=authoritative_execution,
        checkpoint_path=checkpoint_path,
    )
    return {
        "protocol": protocol,
        "protocol_digest": protocol_digest,
        "scientific_execution": scientific_execution,
        "authoritative_execution": authoritative_execution,
        "registry": registry,
        "prep": prep,
        "checkpoint_path": checkpoint_path,
        "checkpoint_receipt": checkpoint_receipt,
        "expected_geometry_digest": TEST_GEOMETRY_DIGEST,
        "expected_geometry_configuration": deepcopy(scientific_execution["configuration"]),
        "execution_code_digest": CODE_DIGEST,
    }


@pytest.fixture(scope="module")
def task5_inputs(tmp_path_factory: pytest.TempPathFactory) -> dict:
    return build_task5_inputs(tmp_path_factory.mktemp("exp289-task5"))
