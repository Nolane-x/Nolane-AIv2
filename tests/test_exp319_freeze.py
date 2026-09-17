from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import pytest

from nolane_ai.experiments import exp319_freeze as freeze
from nolane_ai.experiments.exp319_freeze import (
    EXP319_FROZEN_PATHS,
    EXP319_MARKER_PATHS,
    build_exp319_identity_from_components,
    canonical_marker_json_bytes,
    marker_sidecar_bytes,
    validate_frozen_changed_paths,
    validate_marker_changed_paths,
)


def _h(ch: str, n: int) -> str:
    return ch * n


def test_exp319_frozen_allowlist_is_explicit_and_rejects_unrelated_paths() -> None:
    expected_subset = {
        "src/nolane_ai/experiments/exp319_contract.py",
        "src/nolane_ai/experiments/exp319_worlds.py",
        "src/nolane_ai/experiments/exp319_training.py",
        "src/nolane_ai/experiments/exp319_selection.py",
        "src/nolane_ai/experiments/exp319_challenge.py",
        "src/nolane_ai/experiments/exp319_evidence.py",
        "src/nolane_ai/experiments/exp319_identity.py",
        "src/nolane_ai/experiments/exp319_freeze.py",
        ".github/workflows/exp319-learnability-foundation-diagnostic.yml",
        ".github/workflows/v017-exp319-contract.yml",
        "scripts/verify_exp319_freeze.py",
        *EXP319_MARKER_PATHS,
    }
    assert expected_subset <= set(EXP319_FROZEN_PATHS)

    validate_frozen_changed_paths(expected_subset)
    with pytest.raises(ValueError, match="unrelated|forbidden|EXP-319"):
        validate_frozen_changed_paths((*expected_subset, "README.md"))
    with pytest.raises(ValueError, match="unrelated|forbidden|EXP-319"):
        validate_frozen_changed_paths((*expected_subset, "src/nolane_ai/experiments/exp301_scientific.py"))


def test_execution_identity_from_components_hashes_exact_frozen_bytes() -> None:
    generator = b"generator-contract\n"
    training = b"training-contract\n"
    selection = b"selection-contract\n"
    scoring = b"scoring-contract\n"
    workflow = b"name: exp319\n"

    identity = build_exp319_identity_from_components(
        source_commit_sha=_h("a", 40),
        git_tree_sha=_h("b", 40),
        generator_contract_bytes=generator,
        training_contract_bytes=training,
        selection_contract_bytes=selection,
        scoring_contract_bytes=scoring,
        workflow_bytes=workflow,
    )

    assert identity.generator_contract_digest == hashlib.sha256(generator).hexdigest()
    assert identity.training_contract_digest == hashlib.sha256(training).hexdigest()
    assert identity.selection_contract_digest == hashlib.sha256(selection).hexdigest()
    assert identity.scoring_contract_digest == hashlib.sha256(scoring).hexdigest()
    assert identity.workflow_sha256 == hashlib.sha256(workflow).hexdigest()
    assert identity.source_commit_sha == _h("a", 40)
    assert len(identity.source_tree_digest) == 64


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def test_git_identity_freezes_explicit_component_path_groups_and_source_commit(tmp_path: Path) -> None:
    expected_groups = {
        "generator": ("src/nolane_ai/experiments/exp319_worlds.py",),
        "training": (
            "src/nolane_ai/experiments/exp319_contract.py",
            "src/nolane_ai/experiments/exp319_metrics.py",
            "src/nolane_ai/experiments/exp319_training.py",
        ),
        "selection": ("src/nolane_ai/experiments/exp319_selection.py",),
        "scoring": (
            "src/nolane_ai/experiments/exp319_challenge.py",
            "src/nolane_ai/experiments/exp319_evidence.py",
        ),
    }
    assert freeze.EXP319_COMPONENT_PATHS == expected_groups

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "exp319@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "EXP319 Test"], check=True)

    paths = {
        path
        for group in expected_groups.values()
        for path in group
    } | {freeze.EXP319_WORKFLOW_PATH}
    for index, relative in enumerate(sorted(paths)):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{relative}:{index}\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "source"], check=True, stdout=subprocess.PIPE)

    source_sha = _git(repo, "rev-parse", "HEAD")
    first = freeze.build_exp319_identity_from_git(repo, source_sha)
    assert first.source_commit_sha == source_sha

    generator = repo / expected_groups["generator"][0]
    generator.write_text(generator.read_text(encoding="utf-8") + "mutation\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", str(generator)], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "generator mutation"], check=True, stdout=subprocess.PIPE)
    second = freeze.build_exp319_identity_from_git(repo, "HEAD")

    assert second.source_commit_sha != first.source_commit_sha
    assert second.source_tree_digest != first.source_tree_digest
    assert second.generator_contract_digest != first.generator_contract_digest
    assert second.training_contract_digest == first.training_contract_digest
    assert second.selection_contract_digest == first.selection_contract_digest
    assert second.scoring_contract_digest == first.scoring_contract_digest
    assert second.workflow_sha256 == first.workflow_sha256


def test_marker_json_and_sidecar_are_reproducible() -> None:
    identity = build_exp319_identity_from_components(
        source_commit_sha=_h("a", 40),
        git_tree_sha=_h("b", 40),
        generator_contract_bytes=b"generator",
        training_contract_bytes=b"training",
        selection_contract_bytes=b"selection",
        scoring_contract_bytes=b"scoring",
        workflow_bytes=b"workflow\n",
    )
    first = canonical_marker_json_bytes(identity)
    second = canonical_marker_json_bytes(identity)
    assert first == second
    assert first.endswith(b"\n")

    expected_digest = hashlib.sha256(first).hexdigest()
    sidecar = marker_sidecar_bytes(identity)
    assert sidecar == f"{expected_digest}  {EXP319_MARKER_PATHS[0]}\n".encode("ascii")


def test_marker_commit_must_change_exactly_the_two_execution_identity_files() -> None:
    validate_marker_changed_paths(EXP319_MARKER_PATHS)
    validate_marker_changed_paths(tuple(reversed(EXP319_MARKER_PATHS)))
    with pytest.raises(ValueError, match="exactly the two"):
        validate_marker_changed_paths((EXP319_MARKER_PATHS[0],))
    with pytest.raises(ValueError, match="exactly the two"):
        validate_marker_changed_paths((*EXP319_MARKER_PATHS, "README.md"))
