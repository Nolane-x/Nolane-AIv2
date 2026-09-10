from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "protocols" / "exp289_authoritative_development_v1.json"
DIGEST = ROOT / "protocols" / "exp289_authoritative_development_v1.sha256"
PROTOCOL_DIGEST = "c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440"


def test_authoritative_geometry_is_predeclared_non_tiny_and_digest_bound() -> None:
    from nolane_ai.experiments.exp289_development_geometry import (
        load_exp289_development_geometry,
    )

    geometry, digest = load_exp289_development_geometry(
        MANIFEST,
        DIGEST,
        protocol_digest=PROTOCOL_DIGEST,
    )

    assert digest == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert geometry["tiny"] is False
    assert geometry["eval_replicates"] >= 32
    assert geometry["eval_start_replicate"] >= geometry["train_replicates"]
    assert geometry["restarts"] >= 2
    assert geometry["decoys"] > 0


def test_authoritative_geometry_digest_tamper_fails_closed(tmp_path: Path) -> None:
    from nolane_ai.experiments.exp289_development_geometry import (
        load_exp289_development_geometry,
    )

    manifest = tmp_path / "geometry.json"
    digest = tmp_path / "geometry.sha256"
    manifest.write_bytes(MANIFEST.read_bytes())
    digest.write_text("0" * 64 + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="geometry digest mismatch"):
        load_exp289_development_geometry(
            manifest,
            digest,
            protocol_digest=PROTOCOL_DIGEST,
        )


def test_focused_workflow_executes_authoritative_geometry_as_e2_only() -> None:
    workflow = (ROOT / ".github" / "workflows" / "exp289-confirmatory-ci.yml").read_text(
        encoding="utf-8"
    )

    assert "protocols/exp289_authoritative_development_v1.json" in workflow
    assert "protocols/exp289_authoritative_development_v1.sha256" in workflow
    assert "Run locked authoritative EXP-289 DEVELOPMENT geometry" in workflow
    assert "--geometry-manifest protocols/exp289_authoritative_development_v1.json" in workflow
    assert "development_geometry_authority" in workflow
    assert "Upload authoritative EXP-289 DEVELOPMENT Gate A evidence" in workflow
    assert 'assert e["evidence_level"] == "EV-E2"' in workflow
    assert 'assert e["confirmatory_data_consumed"] is False' in workflow
    assert 'assert e["challenge_materialized"] is False' in workflow
