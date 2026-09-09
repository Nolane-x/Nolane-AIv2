from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "exp277-persist-real-gate-b-evidence.yml"

RUN_ID = "34323835734"
ARTIFACT_ID = "10093277653"
ARTIFACT_NAME = "exp277-real-gate-b-evidence-dd37ab9d0c59f9b86a9e75e190acd6436c3401ac"
ARTIFACT_DIGEST = "sha256:043ad0015258a03cadbd7d6705474756425f1753af7bfba0ceb26d23ecbb247b"
BASELINE_SHA = "b449099f4a8ac5673a9bb55be6c735511ebd786d"
CEREMONY_SHA = "dd37ab9d0c59f9b86a9e75e190acd6436c3401ac"
SOURCE_TREE_DIGEST = "c13f85d290204ed5355b4651d6751c589310468dd7c0e751f2a893a1cdc30900"
PROTOCOL_DIGEST = "c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440"
PRIOR_RUN_ID = "34318307479"
PRIOR_ARM_SHA = "a24b1d"


def _text() -> str:
    assert WORKFLOW.is_file(), f"missing EXP-277 persistence workflow: {WORKFLOW}"
    return WORKFLOW.read_text(encoding="utf-8")


def test_exp277_persistence_pins_exact_authoritative_artifact_and_lineage() -> None:
    text = _text()
    for token in (
        RUN_ID,
        ARTIFACT_ID,
        ARTIFACT_NAME,
        ARTIFACT_DIGEST,
        BASELINE_SHA,
        CEREMONY_SHA,
        SOURCE_TREE_DIGEST,
        PROTOCOL_DIGEST,
        PRIOR_RUN_ID,
        PRIOR_ARM_SHA,
        "PRE_BEACON_PRE_INFERENCE",
        "evidence/exp277/2026-09-09-real-gate-b",
    ):
        assert token in text


def test_exp277_persistence_is_evidence_only_and_never_reexecutes_science() -> None:
    text = _text()
    assert "contents: write" in text
    assert "actions: read" in text
    assert "SHA256SUMS" in text
    assert "actions-artifact-metadata.json" in text
    assert "actions-run-metadata.json" in text
    assert "PROVENANCE.md" in text
    assert "PERSISTENCE-RECEIPT.md" in text
    assert "KILL_SUBSYSTEM" in text
    assert "EV-E3" in text
    assert "cryptographic_signature_verified_by_ceremony" in text
    assert "run_exp277_confirmatory_gate_b.py" not in text
    assert "api.drand.sh/public" not in text
    assert "api2.drand.sh/public" not in text


def test_exp277_persistence_verifies_primary_and_protected_endpoints() -> None:
    text = _text()
    for token in (
        "0.0573178199632578",
        "0.1",
        "0.13671875",
        "-0.005",
        "lower_floor_pass",
        "confirmatory_n",
        "10000",
        "wall_energy_per_episode",
    ):
        assert token in text


def test_exp277_persistence_source_boundary_check_cannot_fail_open_on_shallow_clone() -> None:
    text = _text()
    assert "fetch-depth: 0" in text
    assert "git cat-file -e \"$FREEZE_SHA^{commit}\"" in text


def test_exp277_persistence_uses_summary_fields_that_exist_and_cross_links_lineage() -> None:
    text = _text()
    assert "summary['analysis_digest']" not in text
    assert "summary['seal_digest'] == analysis['lineage']['seal_digest']" in text
    assert "summary['beacon_receipt_digest'] == analysis['lineage']['beacon_receipt_digest']" in text
    assert "summary['source_tree_digest'] == analysis['lineage']['source_tree_digest']" in text
