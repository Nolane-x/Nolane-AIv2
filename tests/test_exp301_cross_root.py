from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from nolane_ai.experiments.exp301_analysis import Exp301AnalysisSummary
from nolane_ai.experiments.exp301_cross_root import (
    build_cross_root_artifact,
    reduce_root_evidence_files,
    write_cross_root_artifact,
)
from nolane_ai.experiments.exp301_execution import EXP301_BOOTSTRAP_SAMPLES


@dataclass(frozen=True, slots=True)
class RootStub:
    frozen_implementation_digest: str
    root: int
    run_identity: str
    artifact_digest: str
    evaluation_rows: tuple[object, ...]
    scientific_evidence_eligible: bool = True


def _roots(*, frozen: str = "a" * 64) -> tuple[RootStub, ...]:
    return tuple(
        RootStub(
            frozen_implementation_digest=frozen,
            root=root,
            run_identity=format(root + 1, "x") * 64,
            artifact_digest=format(root + 5, "x") * 64,
            evaluation_rows=(f"row-{root}",),
        )
        for root in range(4)
    )


def _summary() -> Exp301AnalysisSummary:
    return Exp301AnalysisSummary(
        decision="KILL_H_RD_01",
        aggregate_rcg=-0.01,
        bootstrap_ci_low=-0.02,
        bootstrap_ci_high=-0.001,
        positive_roots=0,
        reasoning_families_ge_5pp=0,
        language_regression=0.0,
        protected_floors_all_clear=True,
        infrastructure_valid=True,
        valid_primary_efforts=(1, 2, 4, 8),
        bootstrap_seed=301_170_017,
        exp302_implementation_authorized=False,
        scale_authorized=False,
    )


def test_cross_root_artifact_requires_exact_four_roots_and_frozen_10000_bootstraps() -> None:
    calls = {}

    def reducer(rows, *, bootstrap_samples):
        calls["rows"] = tuple(rows)
        calls["bootstrap_samples"] = bootstrap_samples
        return _summary()

    artifact = build_cross_root_artifact(_roots(), reducer=reducer)
    assert artifact.roots == (0, 1, 2, 3)
    assert artifact.bootstrap_samples == 10_000 == EXP301_BOOTSTRAP_SAMPLES
    assert calls["bootstrap_samples"] == 10_000
    assert calls["rows"] == ("row-0", "row-1", "row-2", "row-3")
    assert artifact.analysis.decision == "KILL_H_RD_01"
    assert artifact.analysis.exp302_implementation_authorized is False
    assert artifact.analysis.scale_authorized is False
    assert len(artifact.artifact_digest) == 64


def test_cross_root_rejects_missing_duplicate_or_mixed_identity_roots() -> None:
    roots = _roots()
    with pytest.raises(ValueError, match="exactly roots"):
        build_cross_root_artifact(roots[:3], reducer=lambda *_args, **_kwargs: _summary())
    with pytest.raises(ValueError, match="exactly roots|unique"):
        build_cross_root_artifact((roots[0], roots[0], roots[2], roots[3]), reducer=lambda *_args, **_kwargs: _summary())
    mixed = list(roots)
    mixed[3] = RootStub(
        frozen_implementation_digest="b" * 64,
        root=3,
        run_identity=mixed[3].run_identity,
        artifact_digest=mixed[3].artifact_digest,
        evaluation_rows=mixed[3].evaluation_rows,
    )
    with pytest.raises(ValueError, match="frozen implementation"):
        build_cross_root_artifact(tuple(mixed), reducer=lambda *_args, **_kwargs: _summary())


def test_file_reducer_audits_every_root_before_cross_analysis(monkeypatch, tmp_path: Path) -> None:
    roots = _roots()
    paths = tuple(tmp_path / f"root-{root}.json" for root in range(4))
    mapping = dict(zip(paths, roots))
    seen = []

    def loader(path):
        path = Path(path)
        seen.append(path)
        return mapping[path]

    monkeypatch.setattr("nolane_ai.experiments.exp301_cross_root.load_and_audit_root_evidence", loader)
    artifact = reduce_root_evidence_files(
        paths,
        reducer=lambda _rows, *, bootstrap_samples: _summary(),
    )
    assert tuple(seen) == paths
    assert artifact.root_artifact_digests == tuple(root.artifact_digest for root in roots)


def test_cross_root_artifact_is_write_once(tmp_path: Path) -> None:
    artifact = build_cross_root_artifact(
        _roots(),
        reducer=lambda _rows, *, bootstrap_samples: _summary(),
    )
    path = tmp_path / "cross-analysis.json"
    write_cross_root_artifact(path, artifact)
    assert path.is_file()
    with pytest.raises(FileExistsError):
        write_cross_root_artifact(path, artifact)
