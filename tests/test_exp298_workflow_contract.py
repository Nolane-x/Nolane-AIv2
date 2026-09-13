from __future__ import annotations

import json
from pathlib import Path

from nolane_ai.protocol.evidence import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "exp298-cross-domain-fidelity-ci.yml"
DEVELOPMENT = ROOT / ".github" / "workflows" / "exp298-cross-domain-fidelity-development.yml"
FREEZE_GUARD = ROOT / ".github" / "workflows" / "exp298-cross-domain-fidelity-freeze-guard.yml"
GEOMETRY = ROOT / "protocols" / "exp298_cross_domain_fidelity_geometry_v1.json"
GEOMETRY_SHA = ROOT / "protocols" / "exp298_cross_domain_fidelity_geometry_v1.sha256"
MARKER = ROOT / "protocols" / "exp298-cross-domain-fidelity-release.lock"
MARKER_REL = "protocols/exp298-cross-domain-fidelity-release.lock"
PROTOCOL_DIGEST = "c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440"

EXPECTED_GEOMETRY = {
    "schema": "NLM-EXP-298-CROSS-DOMAIN-FIDELITY-GEOMETRY-V1",
    "experiment_id": "EXP-298",
    "authority_scope": "DEVELOPMENT_EV_E2_ONLY",
    "canonical_indices": [0, 1, 2, 3],
    "domains": [
        "code_invariant",
        "causal_diagnosis",
        "grounded_language_ambiguity",
    ],
    "eval_replicates": 32,
    "eval_start_replicate": 50000,
    "candidates_per_replicate": 16,
    "d_model": 64,
    "hidden_size": 64,
    "target_parameters": 500000,
    "max_exact_probes": 4096,
    "balanced_accuracy_gain_mesi": 0.10,
    "wrong_authority_ceiling": 0.05,
    "faithful_rejection_ceiling": 0.10,
    "root_seed_prefix": "20260913-exp298-cross-domain-fidelity-v1-dev",
    "protocol_digest": PROTOCOL_DIGEST,
    "scientific_evidence_eligible": False,
    "confirmatory_data_consumed": False,
    "challenge_materialized": False,
    "promotion_claimed": False,
    "stage_a_protocol_modified": False,
    "successor_scope_if_recurrent": "DESIGN_EXP299_SCAFFOLD_REMOVAL_COURT_ONLY",
    "exp300_authorized": False,
}


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_frozen_geometry_manifest_and_digest_are_exact() -> None:
    payload = json.loads(_text(GEOMETRY))
    assert payload == EXPECTED_GEOMETRY
    assert GEOMETRY_SHA.read_text(encoding="ascii").strip() == canonical_sha256(payload)


def test_dedicated_ci_covers_all_exp298_surfaces_without_write_authority() -> None:
    text = _text(CI)
    assert "name: exp298-cross-domain-fidelity-ci" in text
    assert "pull_request:" in text
    assert "src/nolane_ai/experiments/exp298_*.py" in text
    assert "src/nolane_ai/experiments/matched_cross_domain_fidelity_arms.py" in text
    assert "scripts/run_exp298_cross_domain_fidelity_dev.py" in text
    assert "tests/test_exp298_*.py" in text
    assert "protocols/exp298_cross_domain_fidelity_geometry_v1.json" in text
    assert "protocols/exp298_cross_domain_fidelity_geometry_v1.sha256" in text
    assert "contents: read" in text
    assert "workflow_dispatch" not in text


def test_development_workflow_is_marker_only_exact_head_bound_and_four_root() -> None:
    text = _text(DEVELOPMENT)
    assert "name: exp298-cross-domain-fidelity-development" in text
    assert "workflow_dispatch" not in text
    assert "pull_request:" in text
    assert MARKER_REL in text
    assert "github.event.pull_request.head.sha" in text
    assert "LAST_TOUCH" in text
    assert 'test "$LAST_TOUCH" = "${{ github.event.pull_request.head.sha }}"' in text
    assert "EXP298_CROSS_DOMAIN_FIDELITY_RELEASE_V1" in text
    assert "canonical_indices=0,1,2,3" in text
    assert "eval_replicates=32" in text
    assert "eval_start_replicate=50000" in text
    assert "candidates_per_replicate=16" in text
    assert "d_model=64" in text
    assert "hidden_size=64" in text
    assert "target_parameters=500000" in text
    assert "max_exact_probes=4096" in text
    assert "balanced_accuracy_gain_mesi=0.1" in text
    assert "wrong_authority_ceiling=0.05" in text
    assert "faithful_rejection_ceiling=0.1" in text
    assert "root_seed_prefix=20260913-exp298-cross-domain-fidelity-v1-dev" in text
    assert f"protocol_digest={PROTOCOL_DIGEST}" in text
    assert "authority_scope=DEVELOPMENT_EV_E2_ONLY" in text
    assert "source_tree_digest=" in text
    assert "pre_marker_head=" in text
    assert "canonical_index: [0, 1, 2, 3]" in text
    assert "python scripts/run_exp298_cross_domain_fidelity_dev.py root" in text
    assert "python scripts/run_exp298_cross_domain_fidelity_dev.py cross" in text
    assert "pattern: exp298-root-*-${{ github.sha }}" in text
    assert "name: exp298-cross-${{ github.sha }}" in text


def test_development_preflight_blocks_reruns_and_duplicates_before_root_execution() -> None:
    text = _text(DEVELOPMENT)
    assert "actions: read" in text
    assert "EXP298_DEVELOPMENT_WORKFLOW_NAME" in text
    assert 'current_run_attempt = int(os.environ["GITHUB_RUN_ATTEMPT"])' in text
    assert "current_run_attempt != 1" in text
    assert "rerun DEVELOPMENT attempt blocked before root execution" in text
    assert "candidates.sort" in text
    assert 'authoritative = int(candidates[0]["id"])' in text
    assert "current_run_id != authoritative" in text
    assert "duplicate DEVELOPMENT run blocked before root execution" in text
    assert text.index("rerun DEVELOPMENT attempt blocked before root execution") < text.index("root:")
    assert text.index("duplicate DEVELOPMENT run blocked before root execution") < text.index("root:")


def test_development_has_no_scientific_escape_hatch() -> None:
    text = _text(DEVELOPMENT)
    for token in (
        "secrets.",
        "workflow_dispatch",
        "derive_challenge_seed",
        "challenge_seed",
        "beacon_pulse",
        "actions: write",
    ):
        assert token not in text
    assert "confirmatory" not in text.lower()
    assert "contents: read" in text


def test_release_marker_is_absent_before_predata_green() -> None:
    assert not MARKER.exists(), "EXP-298 release marker must not exist before exact-head pre-data gates are GREEN"
