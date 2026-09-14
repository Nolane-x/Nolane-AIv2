from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

from .exp301_analysis import Exp301AnalysisSummary, reduce_exp301
from .exp301_evidence import RootScientificEvidenceArtifact, load_and_audit_root_evidence
from .exp301_execution import EXP301_BOOTSTRAP_SAMPLES, EXP301_ROOTS


CROSS_ROOT_SCHEMA = "EXP301-CROSS-ROOT-SCIENTIFIC-EVIDENCE-V1"
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class CrossRootScientificEvidenceArtifact:
    schema: str
    frozen_implementation_digest: str
    roots: tuple[int, ...]
    root_artifact_digests: tuple[str, ...]
    root_run_identities: tuple[str, ...]
    bootstrap_samples: int
    analysis: Exp301AnalysisSummary
    scientific_evidence_eligible: bool
    artifact_digest: str


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _require_hex(value: str, *, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in _HEX for ch in value):
        raise ValueError(f"{field} must be 64 lowercase hexadecimal characters")


def _payload(artifact: CrossRootScientificEvidenceArtifact) -> dict[str, object]:
    payload = asdict(artifact)
    payload.pop("artifact_digest", None)
    return payload


def canonical_cross_root_digest(artifact: CrossRootScientificEvidenceArtifact) -> str:
    return _digest(_payload(artifact))


def _ordered_roots(
    artifacts: Iterable[RootScientificEvidenceArtifact],
) -> tuple[RootScientificEvidenceArtifact, ...]:
    materialized = tuple(artifacts)
    roots = tuple(item.root for item in materialized)
    if len(materialized) != len(EXP301_ROOTS) or set(roots) != set(EXP301_ROOTS):
        raise ValueError(f"cross-root analysis requires exactly roots {EXP301_ROOTS}")
    if len(set(roots)) != len(roots):
        raise ValueError("cross-root roots must be unique")
    ordered = tuple(sorted(materialized, key=lambda item: item.root))
    frozen = {item.frozen_implementation_digest for item in ordered}
    if len(frozen) != 1:
        raise ValueError("all roots must bind the same frozen implementation")
    run_ids = tuple(item.run_identity for item in ordered)
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("cross-root run identities must be unique")
    for item in ordered:
        if not item.scientific_evidence_eligible:
            raise ValueError("all root artifacts must be scientific-evidence eligible")
        _require_hex(item.frozen_implementation_digest, field="frozen_implementation_digest")
        _require_hex(item.run_identity, field="run_identity")
        _require_hex(item.artifact_digest, field="artifact_digest")
    return ordered


def build_cross_root_artifact(
    artifacts: Iterable[RootScientificEvidenceArtifact],
    *,
    reducer: Callable[..., Exp301AnalysisSummary] = reduce_exp301,
) -> CrossRootScientificEvidenceArtifact:
    ordered = _ordered_roots(artifacts)
    rows = tuple(row for artifact in ordered for row in artifact.evaluation_rows)
    analysis = reducer(rows, bootstrap_samples=EXP301_BOOTSTRAP_SAMPLES)
    if not isinstance(analysis, Exp301AnalysisSummary):
        raise ValueError("cross-root reducer returned an invalid analysis summary")
    # EXP-301 success may authorize only EXP-302 DESIGN/PREREGISTRATION.
    if analysis.exp302_implementation_authorized or analysis.scale_authorized:
        raise ValueError("EXP-301 cross-root analysis cannot authorize implementation or scale")

    values = dict(
        schema=CROSS_ROOT_SCHEMA,
        frozen_implementation_digest=ordered[0].frozen_implementation_digest,
        roots=EXP301_ROOTS,
        root_artifact_digests=tuple(item.artifact_digest for item in ordered),
        root_run_identities=tuple(item.run_identity for item in ordered),
        bootstrap_samples=EXP301_BOOTSTRAP_SAMPLES,
        analysis=analysis,
        scientific_evidence_eligible=True,
    )
    provisional = CrossRootScientificEvidenceArtifact(**values, artifact_digest="")
    return CrossRootScientificEvidenceArtifact(
        **values,
        artifact_digest=canonical_cross_root_digest(provisional),
    )


def reduce_root_evidence_files(
    paths: Iterable[str | Path],
    *,
    reducer: Callable[..., Exp301AnalysisSummary] = reduce_exp301,
) -> CrossRootScientificEvidenceArtifact:
    materialized_paths = tuple(Path(path) for path in paths)
    if len(materialized_paths) != len(EXP301_ROOTS):
        raise ValueError(f"cross-root analysis requires exactly {len(EXP301_ROOTS)} root artifact files")
    audited = tuple(load_and_audit_root_evidence(path) for path in materialized_paths)
    return build_cross_root_artifact(audited, reducer=reducer)


def write_cross_root_artifact(
    path: str | Path,
    artifact: CrossRootScientificEvidenceArtifact,
) -> None:
    if artifact.schema != CROSS_ROOT_SCHEMA:
        raise ValueError("cross-root artifact schema mismatch")
    if canonical_cross_root_digest(artifact) != artifact.artifact_digest:
        raise ValueError("cross-root artifact digest mismatch")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(asdict(artifact), handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
