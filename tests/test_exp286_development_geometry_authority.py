from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "protocols" / "exp286_development_geometry_v1.json"
DIGEST = ROOT / "protocols" / "exp286_development_geometry_v1.sha256"
PROTOCOL_DIGEST = (ROOT / "protocols" / "stage_a_v1.sha256").read_text(encoding="utf-8").strip()


def test_exp286_geometry_loader_verifies_locked_manifest_and_protocol_lineage() -> None:
    from nolane_ai.experiments.exp286_development_geometry import load_exp286_development_geometry

    geometry, digest = load_exp286_development_geometry(
        MANIFEST,
        DIGEST,
        protocol_digest=PROTOCOL_DIGEST,
    )

    assert digest == DIGEST.read_text(encoding="utf-8").strip()
    assert geometry["tiny"] is False
    assert geometry["eval_replicates"] == 32
    assert geometry["eval_start_replicate"] == 30000


def test_exp286_geometry_loader_rejects_digest_drift(tmp_path: Path) -> None:
    from nolane_ai.experiments.exp286_development_geometry import load_exp286_development_geometry

    manifest = tmp_path / "geometry.json"
    manifest.write_bytes(MANIFEST.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="geometry digest mismatch"):
        load_exp286_development_geometry(
            manifest,
            DIGEST,
            protocol_digest=PROTOCOL_DIGEST,
        )


def test_exp286_geometry_loader_rejects_protocol_lineage_drift() -> None:
    from nolane_ai.experiments.exp286_development_geometry import load_exp286_development_geometry

    with pytest.raises(ValueError, match="protocol lineage mismatch"):
        load_exp286_development_geometry(
            MANIFEST,
            DIGEST,
            protocol_digest="0" * 64,
        )
