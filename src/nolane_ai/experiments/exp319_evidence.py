from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Iterable, Mapping, Sequence

from . import exp319_challenge as challenge
from .exp319_contract import (
    AUTHORIZATION_FLAGS,
    CONTRACT,
    DISPOSITIONS,
    PRIMARY_ARMS,
    TASK_FAMILIES,
    canonical_json_bytes,
)


ROOT_EVIDENCE_SCHEMA = "EXP319-ROOT-EVIDENCE-V1"
CROSS_ROOT_EVIDENCE_SCHEMA = "EXP319-CROSS-ROOT-DIAGNOSTIC-EVIDENCE-V1"


@dataclass(frozen=True, slots=True)
class StageASelectionSeal:
    arm_id: str
    passes_floor: bool
    selected_learning_rate: float
    selection_digest: str


@dataclass(frozen=True, slots=True)
class StageCArmSummary:
    arm_id: str
    answer_token_accuracy: float
    family_exact: tuple[tuple[str, float], ...]
    family_balanced_exact_match: float
    nonzero_exact_families: int
    eos_correctness: float
    invalid_output_rate: float
    passes_floor: bool


@dataclass(frozen=True, slots=True)
class RootEvidence:
    schema: str
    root: int
    provenance_valid: bool
    source_digest: str
    training_contract_digest: str
    scoring_contract_digest: str
    stage_c_floor_digest: str
    stage_a_selections: tuple[StageASelectionSeal, ...]
    stage_b_selection_authority_digest: str
    stage_b_control_pass: bool
    stage_b_nrs_pass: bool
    stage_c_manifest_digest: str | None
    stage_c_commitment_grid_digest: str | None
    stage_c_scoring_digest: str | None
    stage_c_summaries: tuple[StageCArmSummary, ...]
    stage_c_control_pass: bool | None
    stage_c_nrs_pass: bool | None
    artifact_digest: str


@dataclass(frozen=True, slots=True)
class CrossRootDiagnosticEvidence:
    schema: str
    decision: str
    roots: tuple[int, ...]
    stage_b_control_pass_roots: int
    stage_b_nrs_pass_roots: int
    stage_c_control_pass_roots: int | None
    stage_c_nrs_pass_roots: int | None
    authorizations: dict[str, bool]
    exp302_implementation_authorized: bool
    exp320_implementation_authorized: bool
    scale_authorized: bool
    authorized_30m: bool
    authorized_100m: bool
    evidence_digest: str


def _digest_payload(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdefABCDEF" for ch in value)
    )


def _require_digest(value: object, *, label: str) -> None:
    if not _is_digest(value):
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")


def _unit(value: float, *, label: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be finite and in [0,1]")


def _stage_c_floor_digest() -> str:
    return _digest_payload(asdict(CONTRACT.stage_c.floor))


def _validate_stage_a_selections(selections: Sequence[StageASelectionSeal]) -> None:
    if len(selections) != len(PRIMARY_ARMS):
        raise ValueError("Stage A evidence requires exactly the two primary arms")
    if {item.arm_id for item in selections} != set(PRIMARY_ARMS):
        raise ValueError("Stage A evidence primary-arm coverage mismatch")
    for item in selections:
        if item.arm_id not in PRIMARY_ARMS:
            raise ValueError("Stage A evidence arm mismatch")
        if not isinstance(item.passes_floor, bool):
            raise ValueError("Stage A passes_floor must be boolean")
        if item.selected_learning_rate not in CONTRACT.stage_a.learning_rates:
            raise ValueError("Stage A selected learning rate is outside the preregistration")
        _require_digest(item.selection_digest, label="Stage A selection_digest")


def _validate_summary(summary: StageCArmSummary) -> None:
    if summary.arm_id not in PRIMARY_ARMS:
        raise ValueError("Stage C summary arm mismatch")
    _unit(summary.answer_token_accuracy, label="Stage C answer_token_accuracy")
    _unit(summary.family_balanced_exact_match, label="Stage C family_balanced_exact_match")
    _unit(summary.eos_correctness, label="Stage C eos_correctness")
    _unit(summary.invalid_output_rate, label="Stage C invalid_output_rate")
    if len(summary.family_exact) != len(TASK_FAMILIES):
        raise ValueError("Stage C summary requires all frozen families")
    if {name for name, _ in summary.family_exact} != set(TASK_FAMILIES):
        raise ValueError("Stage C summary family coverage mismatch")
    for family, score in summary.family_exact:
        _unit(score, label=f"Stage C family_exact[{family}]")
    recomputed = sum(score for _, score in summary.family_exact) / len(summary.family_exact)
    if not math.isclose(recomputed, summary.family_balanced_exact_match, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Stage C family-balanced exact mismatch")
    if summary.nonzero_exact_families != sum(score > 0.0 for _, score in summary.family_exact):
        raise ValueError("Stage C nonzero-family count mismatch")
    if summary.passes_floor != _stage_c_summary_passes(summary):
        raise ValueError("Stage C passes_floor does not match frozen thresholds")


def _stage_c_summary_passes(summary: StageCArmSummary) -> bool:
    floor = CONTRACT.stage_c.floor
    return (
        summary.answer_token_accuracy >= floor.heldout_answer_token_accuracy
        and summary.family_balanced_exact_match >= floor.heldout_family_balanced_exact_match
        and summary.nonzero_exact_families >= floor.min_families_nonzero_exact
        and summary.eos_correctness >= floor.eos_correctness
        and summary.invalid_output_rate <= floor.invalid_output_rate_max
    )


def _synthetic_summary(arm_id: str, passes: bool) -> StageCArmSummary:
    if passes:
        family_exact = tuple((family, 1.0) for family in TASK_FAMILIES)
        summary = StageCArmSummary(
            arm_id=arm_id,
            answer_token_accuracy=1.0,
            family_exact=family_exact,
            family_balanced_exact_match=1.0,
            nonzero_exact_families=len(TASK_FAMILIES),
            eos_correctness=1.0,
            invalid_output_rate=0.0,
            passes_floor=True,
        )
    else:
        family_exact = tuple((family, 0.0) for family in TASK_FAMILIES)
        summary = StageCArmSummary(
            arm_id=arm_id,
            answer_token_accuracy=0.0,
            family_exact=family_exact,
            family_balanced_exact_match=0.0,
            nonzero_exact_families=0,
            eos_correctness=0.0,
            invalid_output_rate=1.0,
            passes_floor=False,
        )
    _validate_summary(summary)
    return summary


def _root_payload(evidence: RootEvidence) -> dict[str, object]:
    payload = asdict(evidence)
    payload.pop("artifact_digest", None)
    return payload


def _seal_root_evidence(
    *,
    root: int,
    provenance_valid: bool,
    source_digest: str,
    training_contract_digest: str,
    scoring_contract_digest: str,
    stage_a_selections: Sequence[StageASelectionSeal],
    stage_b_selection_authority_digest: str,
    stage_b_control_pass: bool,
    stage_b_nrs_pass: bool,
    stage_c_manifest_digest: str | None,
    stage_c_commitment_grid_digest: str | None,
    stage_c_scoring_digest: str | None,
    stage_c_control_pass: bool | None,
    stage_c_nrs_pass: bool | None,
    stage_c_summaries: Sequence[StageCArmSummary] | None = None,
) -> RootEvidence:
    if root not in CONTRACT.stage_b.roots:
        raise ValueError("root evidence root mismatch")
    if not isinstance(provenance_valid, bool):
        raise ValueError("provenance_valid must be boolean")
    for label, value in (
        ("source_digest", source_digest),
        ("training_contract_digest", training_contract_digest),
        ("scoring_contract_digest", scoring_contract_digest),
        ("stage_b_selection_authority_digest", stage_b_selection_authority_digest),
    ):
        _require_digest(value, label=label)
    _validate_stage_a_selections(stage_a_selections)
    if not isinstance(stage_b_control_pass, bool) or not isinstance(stage_b_nrs_pass, bool):
        raise ValueError("Stage B pass flags must be boolean")

    c_values = (
        stage_c_manifest_digest,
        stage_c_commitment_grid_digest,
        stage_c_scoring_digest,
        stage_c_control_pass,
        stage_c_nrs_pass,
    )
    c_present = any(value is not None for value in c_values)
    if c_present and not all(value is not None for value in c_values):
        raise ValueError("Stage C evidence must be either complete or absent")
    if c_present:
        _require_digest(stage_c_manifest_digest, label="stage_c_manifest_digest")
        _require_digest(stage_c_commitment_grid_digest, label="stage_c_commitment_grid_digest")
        _require_digest(stage_c_scoring_digest, label="stage_c_scoring_digest")
        if not isinstance(stage_c_control_pass, bool) or not isinstance(stage_c_nrs_pass, bool):
            raise ValueError("Stage C pass flags must be boolean")
        summaries = tuple(stage_c_summaries) if stage_c_summaries is not None else (
            _synthetic_summary("A_FIXED", stage_c_control_pass),
            _synthetic_summary("C_NRS_CORE", stage_c_nrs_pass),
        )
        if len(summaries) != len(PRIMARY_ARMS) or {item.arm_id for item in summaries} != set(PRIMARY_ARMS):
            raise ValueError("Stage C summaries must cover exactly both primary arms")
        for summary in summaries:
            _validate_summary(summary)
        by_arm = {item.arm_id: item for item in summaries}
        if by_arm["A_FIXED"].passes_floor != stage_c_control_pass:
            raise ValueError("Stage C control pass flag/summary mismatch")
        if by_arm["C_NRS_CORE"].passes_floor != stage_c_nrs_pass:
            raise ValueError("Stage C NRS pass flag/summary mismatch")
    else:
        if stage_c_summaries not in (None, (), []):
            raise ValueError("Stage C summaries cannot exist without complete Stage C evidence")
        summaries = ()

    without_digest = {
        "schema": ROOT_EVIDENCE_SCHEMA,
        "root": root,
        "provenance_valid": provenance_valid,
        "source_digest": source_digest,
        "training_contract_digest": training_contract_digest,
        "scoring_contract_digest": scoring_contract_digest,
        "stage_c_floor_digest": _stage_c_floor_digest(),
        "stage_a_selections": tuple(stage_a_selections),
        "stage_b_selection_authority_digest": stage_b_selection_authority_digest,
        "stage_b_control_pass": stage_b_control_pass,
        "stage_b_nrs_pass": stage_b_nrs_pass,
        "stage_c_manifest_digest": stage_c_manifest_digest,
        "stage_c_commitment_grid_digest": stage_c_commitment_grid_digest,
        "stage_c_scoring_digest": stage_c_scoring_digest,
        "stage_c_summaries": summaries,
        "stage_c_control_pass": stage_c_control_pass,
        "stage_c_nrs_pass": stage_c_nrs_pass,
    }
    artifact = RootEvidence(**without_digest, artifact_digest="")
    return RootEvidence(**without_digest, artifact_digest=_digest_payload(_root_payload(artifact)))


def _summarize_stage_c(
    rows: Sequence[challenge.StageCScoredRow],
    *,
    arm_id: str,
) -> StageCArmSummary:
    arm_rows = tuple(row for row in rows if row.arm_id == arm_id)
    if len(arm_rows) != CONTRACT.stage_c.total_heldout_examples:
        raise ValueError("Stage C scored rows must contain exactly 512 rows per primary arm")
    family_exact: list[tuple[str, float]] = []
    for family in TASK_FAMILIES:
        family_rows = tuple(row for row in arm_rows if row.family == family)
        if len(family_rows) != CONTRACT.stage_c.heldout_examples_per_family:
            raise ValueError("Stage C scored family geometry mismatch")
        family_exact.append((family, sum(row.exact for row in family_rows) / len(family_rows)))
    family_exact_tuple = tuple(family_exact)
    summary = StageCArmSummary(
        arm_id=arm_id,
        answer_token_accuracy=sum(row.answer_token_accuracy for row in arm_rows) / len(arm_rows),
        family_exact=family_exact_tuple,
        family_balanced_exact_match=sum(score for _, score in family_exact_tuple) / len(family_exact_tuple),
        nonzero_exact_families=sum(score > 0.0 for _, score in family_exact_tuple),
        eos_correctness=sum(row.eos_correct for row in arm_rows) / len(arm_rows),
        invalid_output_rate=sum(row.invalid_output for row in arm_rows) / len(arm_rows),
        passes_floor=False,
    )
    summary = StageCArmSummary(
        arm_id=summary.arm_id,
        answer_token_accuracy=summary.answer_token_accuracy,
        family_exact=summary.family_exact,
        family_balanced_exact_match=summary.family_balanced_exact_match,
        nonzero_exact_families=summary.nonzero_exact_families,
        eos_correctness=summary.eos_correctness,
        invalid_output_rate=summary.invalid_output_rate,
        passes_floor=_stage_c_summary_passes(summary),
    )
    _validate_summary(summary)
    return summary


def build_root_evidence(
    *,
    root: int,
    source_digest: str,
    training_contract_digest: str,
    scoring_contract_digest: str,
    stage_a_selections: Sequence[StageASelectionSeal],
    stage_b_selection_authority: challenge.StageBSelectionAuthority,
    stage_c_manifest: challenge.StageCManifest,
    stage_c_commitments: Iterable[challenge.StageCPredictionCommitment],
    stage_c_scored_rows: Iterable[challenge.StageCScoredRow],
) -> RootEvidence:
    if root != stage_c_manifest.root:
        raise ValueError("root evidence/Stage C manifest root mismatch")
    _require_digest(source_digest, label="source_digest")
    if stage_c_manifest.source_digest != source_digest:
        raise ValueError("root evidence source digest does not match Stage C manifest")
    rebuilt_authority = challenge.seal_stage_b_selection_authority(stage_b_selection_authority.selections)
    if rebuilt_authority != stage_b_selection_authority:
        raise ValueError("Stage B selection authority does not reproduce")
    if stage_c_manifest.stage_b_selection_digest != stage_b_selection_authority.authority_digest:
        raise ValueError("Stage C manifest is not bound to the supplied Stage B selection authority")

    # Trigger the full answer-blind manifest reproduction checks before consuming truth.
    challenge.prediction_inputs_for_shard(stage_c_manifest, shard_index=0)
    commitments = tuple(stage_c_commitments)
    recomputed_scored = challenge.score_stage_c_commitments(stage_c_manifest, commitments)
    scored = tuple(stage_c_scored_rows)
    if scored != recomputed_scored:
        raise ValueError("Stage C scored rows do not reproduce from committed predictions")
    if any(row.root != root for row in scored):
        raise ValueError("Stage C scored row root mismatch")

    content_position = {content_id: index for index, content_id in enumerate(stage_c_manifest.content_ids)}
    ordered_commitments = tuple(
        sorted(
            commitments,
            key=lambda item: (content_position[item.content_id], PRIMARY_ARMS.index(item.arm_id)),
        )
    )
    commitment_grid_digest = _digest_payload(
        {"commitment_digests": [item.commitment_digest for item in ordered_commitments]}
    )
    ordered_rows = tuple(
        sorted(
            scored,
            key=lambda item: (content_position[item.content_id], PRIMARY_ARMS.index(item.arm_id)),
        )
    )
    scoring_digest = _digest_payload(
        {"rows": [asdict(item) for item in ordered_rows]}
    )
    summaries = tuple(_summarize_stage_c(ordered_rows, arm_id=arm_id) for arm_id in PRIMARY_ARMS)
    summary_by_arm = {item.arm_id: item for item in summaries}

    selections_by_key = {
        (item.arm_id, item.root): item
        for item in stage_b_selection_authority.selections
    }
    try:
        control_selection = selections_by_key[("A_FIXED", root)]
        nrs_selection = selections_by_key[("C_NRS_CORE", root)]
    except KeyError as exc:
        raise ValueError("Stage B selection authority lacks root primary-arm evidence") from exc

    return _seal_root_evidence(
        root=root,
        provenance_valid=True,
        source_digest=source_digest,
        training_contract_digest=training_contract_digest,
        scoring_contract_digest=scoring_contract_digest,
        stage_a_selections=stage_a_selections,
        stage_b_selection_authority_digest=stage_b_selection_authority.authority_digest,
        stage_b_control_pass=control_selection.passes_floor,
        stage_b_nrs_pass=nrs_selection.passes_floor,
        stage_c_manifest_digest=stage_c_manifest.manifest_digest,
        stage_c_commitment_grid_digest=commitment_grid_digest,
        stage_c_scoring_digest=scoring_digest,
        stage_c_control_pass=summary_by_arm["A_FIXED"].passes_floor,
        stage_c_nrs_pass=summary_by_arm["C_NRS_CORE"].passes_floor,
        stage_c_summaries=summaries,
    )


def _root_integrity_valid(root: RootEvidence) -> bool:
    try:
        if root.schema != ROOT_EVIDENCE_SCHEMA:
            return False
        if root.root not in CONTRACT.stage_b.roots:
            return False
        if root.stage_c_floor_digest != _stage_c_floor_digest():
            return False
        _require_digest(root.source_digest, label="source_digest")
        _require_digest(root.training_contract_digest, label="training_contract_digest")
        _require_digest(root.scoring_contract_digest, label="scoring_contract_digest")
        _require_digest(root.stage_b_selection_authority_digest, label="stage_b_selection_authority_digest")
        _validate_stage_a_selections(root.stage_a_selections)
        if not isinstance(root.stage_b_control_pass, bool) or not isinstance(root.stage_b_nrs_pass, bool):
            return False
        c_present = any(
            value is not None
            for value in (
                root.stage_c_manifest_digest,
                root.stage_c_commitment_grid_digest,
                root.stage_c_scoring_digest,
                root.stage_c_control_pass,
                root.stage_c_nrs_pass,
            )
        )
        if c_present:
            if not all(
                value is not None
                for value in (
                    root.stage_c_manifest_digest,
                    root.stage_c_commitment_grid_digest,
                    root.stage_c_scoring_digest,
                    root.stage_c_control_pass,
                    root.stage_c_nrs_pass,
                )
            ):
                return False
            _require_digest(root.stage_c_manifest_digest, label="stage_c_manifest_digest")
            _require_digest(root.stage_c_commitment_grid_digest, label="stage_c_commitment_grid_digest")
            _require_digest(root.stage_c_scoring_digest, label="stage_c_scoring_digest")
            if len(root.stage_c_summaries) != len(PRIMARY_ARMS):
                return False
            if {item.arm_id for item in root.stage_c_summaries} != set(PRIMARY_ARMS):
                return False
            for summary in root.stage_c_summaries:
                _validate_summary(summary)
            by_arm = {item.arm_id: item for item in root.stage_c_summaries}
            if by_arm["A_FIXED"].passes_floor != root.stage_c_control_pass:
                return False
            if by_arm["C_NRS_CORE"].passes_floor != root.stage_c_nrs_pass:
                return False
        elif root.stage_c_summaries:
            return False
        _require_digest(root.artifact_digest, label="artifact_digest")
        if root.artifact_digest != _digest_payload(_root_payload(root)):
            return False
        return root.provenance_valid is True
    except (KeyError, TypeError, ValueError):
        return False


def _decision_payload(evidence: CrossRootDiagnosticEvidence) -> dict[str, object]:
    payload = asdict(evidence)
    payload.pop("evidence_digest", None)
    return payload


def _build_decision(
    *,
    decision: str,
    roots: Sequence[RootEvidence],
    stage_b_control_pass_roots: int,
    stage_b_nrs_pass_roots: int,
    stage_c_control_pass_roots: int | None,
    stage_c_nrs_pass_roots: int | None,
) -> CrossRootDiagnosticEvidence:
    if decision not in DISPOSITIONS:
        raise ValueError("unknown EXP-319 disposition")
    authorizations = dict(AUTHORIZATION_FLAGS)
    without_digest = {
        "schema": CROSS_ROOT_EVIDENCE_SCHEMA,
        "decision": decision,
        "roots": tuple(sorted(root.root for root in roots)),
        "stage_b_control_pass_roots": stage_b_control_pass_roots,
        "stage_b_nrs_pass_roots": stage_b_nrs_pass_roots,
        "stage_c_control_pass_roots": stage_c_control_pass_roots,
        "stage_c_nrs_pass_roots": stage_c_nrs_pass_roots,
        "authorizations": authorizations,
        "exp302_implementation_authorized": False,
        "exp320_implementation_authorized": False,
        "scale_authorized": False,
        "authorized_30m": False,
        "authorized_100m": False,
    }
    artifact = CrossRootDiagnosticEvidence(**without_digest, evidence_digest="")
    return CrossRootDiagnosticEvidence(
        **without_digest,
        evidence_digest=_digest_payload(_decision_payload(artifact)),
    )


def reduce_cross_root(roots: Iterable[RootEvidence]) -> CrossRootDiagnosticEvidence:
    materialized = tuple(roots)
    placeholder_counts = (0, 0)
    if len(materialized) != len(CONTRACT.stage_b.roots) or {root.root for root in materialized} != set(CONTRACT.stage_b.roots):
        return _build_decision(
            decision="INVALID_DIAGNOSTIC",
            roots=materialized,
            stage_b_control_pass_roots=0,
            stage_b_nrs_pass_roots=0,
            stage_c_control_pass_roots=None,
            stage_c_nrs_pass_roots=None,
        )
    if not all(_root_integrity_valid(root) for root in materialized):
        return _build_decision(
            decision="INVALID_DIAGNOSTIC",
            roots=materialized,
            stage_b_control_pass_roots=0,
            stage_b_nrs_pass_roots=0,
            stage_c_control_pass_roots=None,
            stage_c_nrs_pass_roots=None,
        )

    if len({root.source_digest for root in materialized}) != 1:
        return _build_decision(decision="INVALID_DIAGNOSTIC", roots=materialized, stage_b_control_pass_roots=0, stage_b_nrs_pass_roots=0, stage_c_control_pass_roots=None, stage_c_nrs_pass_roots=None)
    if len({root.training_contract_digest for root in materialized}) != 1:
        return _build_decision(decision="INVALID_DIAGNOSTIC", roots=materialized, stage_b_control_pass_roots=0, stage_b_nrs_pass_roots=0, stage_c_control_pass_roots=None, stage_c_nrs_pass_roots=None)
    if len({root.scoring_contract_digest for root in materialized}) != 1:
        return _build_decision(decision="INVALID_DIAGNOSTIC", roots=materialized, stage_b_control_pass_roots=0, stage_b_nrs_pass_roots=0, stage_c_control_pass_roots=None, stage_c_nrs_pass_roots=None)
    if len({root.stage_b_selection_authority_digest for root in materialized}) != 1:
        return _build_decision(decision="INVALID_DIAGNOSTIC", roots=materialized, stage_b_control_pass_roots=0, stage_b_nrs_pass_roots=0, stage_c_control_pass_roots=None, stage_c_nrs_pass_roots=None)
    stage_a_shapes = {
        tuple((item.arm_id, item.passes_floor, item.selected_learning_rate, item.selection_digest) for item in root.stage_a_selections)
        for root in materialized
    }
    if len(stage_a_shapes) != 1:
        return _build_decision(decision="INVALID_DIAGNOSTIC", roots=materialized, stage_b_control_pass_roots=0, stage_b_nrs_pass_roots=0, stage_c_control_pass_roots=None, stage_c_nrs_pass_roots=None)

    stage_a_by_arm = {item.arm_id: item for item in materialized[0].stage_a_selections}
    b_control = sum(root.stage_b_control_pass for root in materialized)
    b_nrs = sum(root.stage_b_nrs_pass for root in materialized)

    if not stage_a_by_arm["A_FIXED"].passes_floor:
        return _build_decision(decision="TRAINING_STACK_NOT_LEARNABLE", roots=materialized, stage_b_control_pass_roots=b_control, stage_b_nrs_pass_roots=b_nrs, stage_c_control_pass_roots=None, stage_c_nrs_pass_roots=None)
    if not stage_a_by_arm["C_NRS_CORE"].passes_floor:
        return _build_decision(decision="NRS_LOCAL_TRAINABILITY_FAIL", roots=materialized, stage_b_control_pass_roots=b_control, stage_b_nrs_pass_roots=b_nrs, stage_c_control_pass_roots=None, stage_c_nrs_pass_roots=None)
    if b_control < CONTRACT.stage_b.cross_root_min_passes:
        return _build_decision(decision="SUPERVISED_FOUNDATION_UNDERTRAINED", roots=materialized, stage_b_control_pass_roots=b_control, stage_b_nrs_pass_roots=b_nrs, stage_c_control_pass_roots=None, stage_c_nrs_pass_roots=None)
    if b_nrs < CONTRACT.stage_b.cross_root_min_passes:
        return _build_decision(decision="NRS_IID_GENERALIZATION_FLOOR_FAIL", roots=materialized, stage_b_control_pass_roots=b_control, stage_b_nrs_pass_roots=b_nrs, stage_c_control_pass_roots=None, stage_c_nrs_pass_roots=None)

    if any(root.stage_c_control_pass is None or root.stage_c_nrs_pass is None for root in materialized):
        return _build_decision(decision="INVALID_DIAGNOSTIC", roots=materialized, stage_b_control_pass_roots=b_control, stage_b_nrs_pass_roots=b_nrs, stage_c_control_pass_roots=None, stage_c_nrs_pass_roots=None)
    c_control = sum(bool(root.stage_c_control_pass) for root in materialized)
    c_nrs = sum(bool(root.stage_c_nrs_pass) for root in materialized)
    if c_control < CONTRACT.stage_c.cross_root_min_passes:
        return _build_decision(decision="SUPERVISED_HELDOUT_FOUNDATION_FAIL", roots=materialized, stage_b_control_pass_roots=b_control, stage_b_nrs_pass_roots=b_nrs, stage_c_control_pass_roots=c_control, stage_c_nrs_pass_roots=c_nrs)
    if c_nrs < CONTRACT.stage_c.cross_root_min_passes:
        return _build_decision(decision="NRS_HELDOUT_GENERALIZATION_FLOOR_FAIL", roots=materialized, stage_b_control_pass_roots=b_control, stage_b_nrs_pass_roots=b_nrs, stage_c_control_pass_roots=c_control, stage_c_nrs_pass_roots=c_nrs)
    return _build_decision(decision="FOUNDATION_READY_FOR_EXP320_DESIGN_ONLY", roots=materialized, stage_b_control_pass_roots=b_control, stage_b_nrs_pass_roots=b_nrs, stage_c_control_pass_roots=c_control, stage_c_nrs_pass_roots=c_nrs)
