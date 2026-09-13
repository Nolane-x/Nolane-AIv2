from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from nolane_ai.protocol.evidence import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = ROOT / "protocols" / "exp299_native_fidelity_geometry_v1.json"
GEOMETRY_SHA = ROOT / "protocols" / "exp299_native_fidelity_geometry_v1.sha256"
SCRIPT = ROOT / "scripts" / "run_exp299_native_fidelity_dev.py"
RELEASE = ROOT / "protocols" / "exp299-native-fidelity-release.lock"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("exp299_cli_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_geometry_is_exact_and_self_digesting() -> None:
    payload = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    declared = GEOMETRY_SHA.read_text(encoding="ascii").strip()
    assert canonical_sha256(payload) == declared
    assert payload == {
        "schema": "NLM-EXP-299-NATIVE-FIDELITY-GEOMETRY-V1",
        "experiment_id": "EXP-299",
        "authority_scope": "DEVELOPMENT_EV_E2_ONLY",
        "canonical_indices": [0, 1, 2, 3],
        "root_seed_prefix": "20260913-exp299-native-fidelity-v1-dev",
        "protocol_digest": "c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440",
        "domains": [
            "code_invariant",
            "causal_diagnosis",
            "grounded_language_ambiguity",
        ],
        "fit_replicates": 96,
        "fit_start_replicate": 0,
        "eval_replicates": 32,
        "eval_start_replicate": 10000,
        "candidates_per_replicate": 16,
        "structural_feature_width": 128,
        "pair_feature_width": 512,
        "d_model": 64,
        "hidden_size": 64,
        "target_parameters": 500000,
        "batch_size": 32,
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
        "fit_passes": 1,
        "control_auxiliary_loss_weight": 0.0,
        "teacher_auxiliary_loss_weight": 0.25,
        "authority_threshold": 0.5,
        "max_exact_probes": 4096,
        "native_ba_floor": 0.7,
        "native_gain_mesi": 0.1,
        "wrong_authority_ceiling": 0.05,
        "faithful_rejection_ceiling": 0.1,
        "heldout_generalization_scope": "instance_lineage_within_frozen_domain_family",
        "successor_scope_if_recurrent": "DESIGN_EXP300_INTEGRATED_COURT_ONLY",
        "scientific_evidence_eligible": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
        "stage_a_protocol_modified": False,
        "exp300_execution_authorized": False,
    }


def test_cli_loader_rejects_geometry_or_stage_a_drift() -> None:
    module = _load_script_module()
    payload, digest = module._load_frozen_geometry()
    assert payload["experiment_id"] == "EXP-299"
    assert digest == GEOMETRY_SHA.read_text(encoding="ascii").strip()


def test_cli_atomic_writer_publishes_sha_sidecar_and_never_overwrites(tmp_path) -> None:
    module = _load_script_module()
    output = tmp_path / "receipt.json"
    receipt = {"artifact_digest": "a" * 64, "decision": "TEST_ONLY"}
    module._write_canonical_atomic(output, receipt)
    raw = output.read_bytes()
    sidecar = output.with_name(output.name + ".sha256")
    assert sidecar.read_text(encoding="ascii") == hashlib.sha256(raw).hexdigest() + "\n"
    with pytest.raises(FileExistsError):
        module._write_canonical_atomic(output, receipt)


def test_cli_cannot_create_or_mutate_release_authority() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "exp299-native-fidelity-release.lock" not in text
    assert "workflow_dispatch" not in text
    # The release marker may legitimately exist later as a marker-only commit;
    # scientific execution authority belongs to the DEVELOPMENT workflow, not CLI.
    if RELEASE.exists():
        assert RELEASE.read_text(encoding="utf-8").startswith(
            "EXP299_NATIVE_FIDELITY_RELEASE_V1\n"
        )
