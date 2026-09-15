from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

RECOVERY_SCHEMA = "EXP301R-STANDARD-RUNNER-SHARDED-RECOVERY-V1"
PRIOR_FAILED_RUN_ID = 34823251660
FROZEN_IMPLEMENTATION_DIGEST = "89ea87607b75061c0c3d426fdf07ba82cf1c0fb0ad62135c1106ccbaa5e8890c"
FROZEN_ARMS = ("A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE")
FROZEN_EFFORTS = (1, 2, 4, 8, 12, 16)
SHARD_COUNT = 16
WORLDS_PER_ROOT = 512
WORLDS_PER_SHARD = WORLDS_PER_ROOT // SHARD_COUNT
GENERATION_TOKENS = 96

TRIAL_RECEIPT_SCHEMA = "EXP301R-TRIAL-RECEIPT-V1"
SELECTION_RECEIPT_SCHEMA = "EXP301R-SELECTION-RECEIPT-V1"
CHALLENGE_MANIFEST_SCHEMA = "EXP301R-CHALLENGE-MANIFEST-V1"
PREDICTION_SHARD_SCHEMA = "EXP301R-PREDICTION-SHARD-V1"
ROOT_RECOVERY_RECEIPT_SCHEMA = "EXP301R-ROOT-RECOVERY-RECEIPT-V1"
CROSS_ROOT_RECOVERY_ENVELOPE_SCHEMA = "EXP301R-CROSS-ROOT-RECOVERY-ENVELOPE-V1"

_ALLOWED_EXACT_PATHS = frozenset(
    {
        ".github/workflows/exp301r-standard-runner-sharded-recovery.yml",
        "docs/superpowers/specs/2026-09-14-exp301r-standard-runner-sharded-recovery-design.md",
        "docs/superpowers/plans/2026-09-14-exp301r-standard-runner-sharded-recovery.md",
        "scripts/verify_exp301r_recovery.py",
    }
)
_ALLOWED_PREFIXES = (
    "src/nolane_ai/experiments/exp301r_",
    "scripts/exp301r_",
    "tests/test_exp301r_",
)
_HEX = frozenset("0123456789abcdef")


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def canonical_digest(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def write_once_json(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        handle.write(_canonical_bytes(payload).decode("utf-8"))
        handle.write("\n")


def _require_hex64(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in _HEX for ch in value):
        raise ValueError(f"{field} must be 64 lowercase hexadecimal characters")
    return value


def _require_root(root: object) -> int:
    if not isinstance(root, int) or root not in (0, 1, 2, 3):
        raise ValueError("root must be one of (0,1,2,3)")
    return root


def _run_id_text(run_id: int | str) -> str:
    text = str(run_id)
    if not text.isdigit() or int(text) <= 0:
        raise ValueError("run_id must be a positive decimal integer")
    return text


def _require_frozen_digest(value: object) -> str:
    value = _require_hex64(value, field="frozen_implementation_digest")
    if value != FROZEN_IMPLEMENTATION_DIGEST:
        raise ValueError("recovery must bind the frozen EXP-301 implementation digest")
    return value


def _seal(payload: dict[str, object], *, field: str) -> dict[str, object]:
    if field in payload:
        raise ValueError(f"payload already contains {field}")
    return {**payload, field: canonical_digest(payload)}


def _verify_seal(payload: Mapping[str, object], *, field: str) -> None:
    supplied = _require_hex64(payload.get(field), field=field)
    unsigned = dict(payload)
    unsigned.pop(field, None)
    if canonical_digest(unsigned) != supplied:
        raise ValueError(f"{field} mismatch")


def _as_mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise ValueError("value cannot be materialized as an object mapping")


def recovery_beacon(run_id: int | str) -> str:
    return f"github-run-{_run_id_text(run_id)}-exp301r-v1"


def challenge_shard_bounds(shard_index: int) -> tuple[int, int]:
    if not isinstance(shard_index, int) or not 0 <= shard_index < SHARD_COUNT:
        raise ValueError(f"shard_index must be in [0,{SHARD_COUNT - 1}]")
    start = shard_index * WORLDS_PER_SHARD
    return start, start + WORLDS_PER_SHARD


def ensure_exact_shard_cover(shard_indices: Iterable[int]) -> tuple[int, ...]:
    materialized = tuple(shard_indices)
    expected = tuple(range(SHARD_COUNT))
    if tuple(sorted(materialized)) != expected or len(materialized) != SHARD_COUNT:
        raise ValueError(f"recovery requires exactly shard indices {expected}")
    return expected


def _path_allowed(path: str) -> bool:
    if path in _ALLOWED_EXACT_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in _ALLOWED_PREFIXES)


def validate_recovery_changed_paths(paths: Iterable[str]) -> tuple[str, ...]:
    materialized = tuple(sorted(set(paths)))
    for path in materialized:
        if not _path_allowed(path):
            raise ValueError(f"forbidden recovery path: {path}")
    return materialized


def resolve_trial_plan(
    plans: Iterable[object],
    *,
    root: int,
    arm_id: str,
    trial_index: int,
) -> object:
    _require_root(root)
    if arm_id not in FROZEN_ARMS:
        raise ValueError(f"arm_id must be one of {FROZEN_ARMS}")
    if not isinstance(trial_index, int):
        raise ValueError("trial_index must be an integer")
    matches = tuple(
        plan
        for plan in plans
        if getattr(plan, "root", None) == root
        and getattr(plan, "arm_id", None) == arm_id
        and getattr(plan, "trial_index", None) == trial_index
    )
    if len(matches) != 1:
        raise ValueError("coordinate must resolve to exactly one frozen trial")
    return matches[0]


def build_trial_recovery_receipt(
    plan: object,
    result: object,
    *,
    frozen_implementation_digest: str,
    run_id: int | str,
    workflow_digest: str,
) -> dict[str, object]:
    _require_frozen_digest(frozen_implementation_digest)
    workflow_digest = _require_hex64(workflow_digest, field="workflow_digest")
    run_id_text = _run_id_text(run_id)
    root = _require_root(getattr(plan, "root", None))
    arm_id = getattr(plan, "arm_id", None)
    trial_index = getattr(plan, "trial_index", None)
    if arm_id not in FROZEN_ARMS or trial_index not in (0, 1):
        raise ValueError("invalid frozen trial plan coordinate")
    for field in ("root", "arm_id", "learning_rate", "model_init_seed"):
        if getattr(result, field, None) != getattr(plan, field, None):
            raise ValueError(f"trial result {field} does not match frozen plan")
    if getattr(result, "training_steps", None) != 512:
        raise ValueError("trial result must bind exactly 512 training steps")
    checkpoint_digest = _require_hex64(getattr(result, "checkpoint_digest", None), field="checkpoint_digest")
    trial_receipt_digest = _require_hex64(
        getattr(result, "trial_receipt_digest", None), field="trial_receipt_digest"
    )
    payload = {
        "schema": TRIAL_RECEIPT_SCHEMA,
        "recovery_schema": RECOVERY_SCHEMA,
        "frozen_implementation_digest": FROZEN_IMPLEMENTATION_DIGEST,
        "prior_failed_run_id": PRIOR_FAILED_RUN_ID,
        "run_id": run_id_text,
        "workflow_digest": workflow_digest,
        "root": root,
        "arm_id": arm_id,
        "trial_index": trial_index,
        "learning_rate": getattr(plan, "learning_rate"),
        "model_init_seed": getattr(plan, "model_init_seed"),
        "training_steps": 512,
        "checkpoint_digest": checkpoint_digest,
        "trial_receipt_digest": trial_receipt_digest,
    }
    return _seal(payload, field="receipt_digest")


def build_selection_recovery_receipt(
    *,
    root: int,
    selection_manifest_digest: str,
    selected_checkpoint_digests: Iterable[str],
    trial_recovery_receipt_digests: Iterable[str],
    run_id: int | str,
    workflow_digest: str,
) -> dict[str, object]:
    root = _require_root(root)
    selection_manifest_digest = _require_hex64(
        selection_manifest_digest, field="selection_manifest_digest"
    )
    workflow_digest = _require_hex64(workflow_digest, field="workflow_digest")
    selected = tuple(selected_checkpoint_digests)
    trial_receipts = tuple(trial_recovery_receipt_digests)
    if len(selected) != 3:
        raise ValueError("selection recovery requires exactly three selected checkpoints")
    if len(trial_receipts) != 6:
        raise ValueError("selection recovery requires exactly six trial receipts")
    if len(set(selected)) != 3:
        raise ValueError("selected checkpoint digests must be unique")
    if len(set(trial_receipts)) != 6:
        raise ValueError("trial recovery receipt digests must be unique")
    for index, digest in enumerate(selected):
        _require_hex64(digest, field=f"selected_checkpoint_digests[{index}]")
    for index, digest in enumerate(trial_receipts):
        _require_hex64(digest, field=f"trial_recovery_receipt_digests[{index}]")
    payload = {
        "schema": SELECTION_RECEIPT_SCHEMA,
        "recovery_schema": RECOVERY_SCHEMA,
        "frozen_implementation_digest": FROZEN_IMPLEMENTATION_DIGEST,
        "prior_failed_run_id": PRIOR_FAILED_RUN_ID,
        "run_id": _run_id_text(run_id),
        "workflow_digest": workflow_digest,
        "root": root,
        "selection_manifest_digest": selection_manifest_digest,
        "selected_checkpoint_digests": selected,
        "trial_recovery_receipt_digests": trial_receipts,
    }
    return _seal(payload, field="receipt_digest")


def build_challenge_manifest(
    challenge: object,
    *,
    run_id: int | str,
    workflow_digest: str,
    selection_manifest_digest: str,
) -> dict[str, object]:
    root = _require_root(getattr(challenge, "root", None))
    nonce = _require_hex64(getattr(challenge, "challenge_nonce", None), field="challenge_nonce")
    materialization_digest = _require_hex64(
        getattr(challenge, "materialization_digest", None), field="challenge_materialization_digest"
    )
    workflow_digest = _require_hex64(workflow_digest, field="workflow_digest")
    selection_manifest_digest = _require_hex64(
        selection_manifest_digest, field="selection_manifest_digest"
    )
    worlds = tuple(getattr(challenge, "worlds", ()))
    if len(worlds) != WORLDS_PER_ROOT:
        raise ValueError(f"challenge manifest requires exactly {WORLDS_PER_ROOT} worlds")
    content_ids = [getattr(world, "content_id", None) for world in worlds]
    if any(not isinstance(item, str) or not item for item in content_ids):
        raise ValueError("challenge worlds require non-empty content ids")
    if len(set(content_ids)) != WORLDS_PER_ROOT:
        raise ValueError("challenge content ids must be unique")
    run_id_text = _run_id_text(run_id)
    payload = {
        "schema": CHALLENGE_MANIFEST_SCHEMA,
        "recovery_schema": RECOVERY_SCHEMA,
        "frozen_implementation_digest": FROZEN_IMPLEMENTATION_DIGEST,
        "prior_failed_run_id": PRIOR_FAILED_RUN_ID,
        "run_id": run_id_text,
        "workflow_digest": workflow_digest,
        "root": root,
        "beacon": recovery_beacon(run_id_text),
        "challenge_nonce": nonce,
        "challenge_materialization_digest": materialization_digest,
        "selection_manifest_digest": selection_manifest_digest,
        "ordered_content_ids": content_ids,
    }
    return _seal(payload, field="manifest_digest")


def _validate_challenge_manifest(
    manifest: Mapping[str, object],
    *,
    run_id: int | str,
    workflow_digest: str,
    selection_manifest_digest: str,
) -> None:
    if manifest.get("schema") != CHALLENGE_MANIFEST_SCHEMA:
        raise ValueError("challenge manifest schema mismatch")
    _verify_seal(manifest, field="manifest_digest")
    _require_frozen_digest(manifest.get("frozen_implementation_digest"))
    if manifest.get("run_id") != _run_id_text(run_id):
        raise ValueError("challenge manifest run id mismatch")
    if manifest.get("workflow_digest") != _require_hex64(workflow_digest, field="workflow_digest"):
        raise ValueError("challenge manifest workflow digest mismatch")
    if manifest.get("selection_manifest_digest") != _require_hex64(
        selection_manifest_digest, field="selection_manifest_digest"
    ):
        raise ValueError("challenge manifest selection digest mismatch")
    if manifest.get("beacon") != recovery_beacon(run_id):
        raise ValueError("challenge manifest beacon mismatch")
    _require_root(manifest.get("root"))
    _require_hex64(manifest.get("challenge_nonce"), field="challenge_nonce")
    _require_hex64(
        manifest.get("challenge_materialization_digest"), field="challenge_materialization_digest"
    )
    content_ids = tuple(manifest.get("ordered_content_ids", ()))
    if len(content_ids) != WORLDS_PER_ROOT or len(set(content_ids)) != WORLDS_PER_ROOT:
        raise ValueError("challenge manifest must contain exactly 512 unique ordered content ids")


def build_prediction_shard(
    *,
    challenge_manifest: Mapping[str, object],
    shard_index: int,
    commitments: Iterable[object],
    run_id: int | str,
    workflow_digest: str,
    selection_manifest_digest: str,
) -> dict[str, object]:
    _validate_challenge_manifest(
        challenge_manifest,
        run_id=run_id,
        workflow_digest=workflow_digest,
        selection_manifest_digest=selection_manifest_digest,
    )
    start, end = challenge_shard_bounds(shard_index)
    content_ids = tuple(challenge_manifest["ordered_content_ids"])[start:end]
    materialized = tuple(_as_mapping(item) for item in commitments)
    expected_count = WORLDS_PER_SHARD * len(FROZEN_ARMS) * len(FROZEN_EFFORTS)
    if len(materialized) != expected_count:
        raise ValueError(f"prediction shard requires exactly {expected_count} commitments")
    expected_keys = tuple(
        (content_id, arm_id, effort)
        for content_id in content_ids
        for arm_id in FROZEN_ARMS
        for effort in FROZEN_EFFORTS
    )
    actual_keys = []
    root = int(challenge_manifest["root"])
    for item in materialized:
        if "verified_success" in item or "evaluation_rows" in item:
            raise ValueError("prediction shard cannot contain verifier scores")
        if item.get("root") != root:
            raise ValueError("prediction shard commitment root mismatch")
        if item.get("generation_token_count") != GENERATION_TOKENS:
            raise ValueError("prediction shard must preserve the 96-token frozen budget")
        _require_hex64(item.get("prediction_digest"), field="prediction_digest")
        actual_keys.append((item.get("content_id"), item.get("arm_id"), item.get("effort_multiplier")))
    if tuple(actual_keys) != expected_keys:
        raise ValueError("prediction commitments are not in canonical world/arm/effort order")
    payload = {
        "schema": PREDICTION_SHARD_SCHEMA,
        "recovery_schema": RECOVERY_SCHEMA,
        "frozen_implementation_digest": FROZEN_IMPLEMENTATION_DIGEST,
        "prior_failed_run_id": PRIOR_FAILED_RUN_ID,
        "run_id": _run_id_text(run_id),
        "workflow_digest": _require_hex64(workflow_digest, field="workflow_digest"),
        "root": root,
        "beacon": challenge_manifest["beacon"],
        "challenge_materialization_digest": challenge_manifest["challenge_materialization_digest"],
        "selection_manifest_digest": _require_hex64(
            selection_manifest_digest, field="selection_manifest_digest"
        ),
        "challenge_manifest_digest": challenge_manifest["manifest_digest"],
        "shard_index": shard_index,
        "world_start": start,
        "world_end": end,
        "content_ids": content_ids,
        "commitments": materialized,
    }
    return _seal(payload, field="shard_digest")


def _validate_prediction_shard(shard: Mapping[str, object]) -> None:
    if shard.get("schema") != PREDICTION_SHARD_SCHEMA:
        raise ValueError("prediction shard schema mismatch")
    _verify_seal(shard, field="shard_digest")
    _require_frozen_digest(shard.get("frozen_implementation_digest"))
    index = shard.get("shard_index")
    start, end = challenge_shard_bounds(index)  # type: ignore[arg-type]
    if shard.get("world_start") != start or shard.get("world_end") != end:
        raise ValueError("prediction shard bounds mismatch")
    content_ids = tuple(shard.get("content_ids", ()))
    if len(content_ids) != WORLDS_PER_SHARD:
        raise ValueError("prediction shard content id count mismatch")
    materialized = tuple(shard.get("commitments", ()))
    if len(materialized) != WORLDS_PER_SHARD * len(FROZEN_ARMS) * len(FROZEN_EFFORTS):
        raise ValueError("prediction shard commitment count mismatch")


def merge_prediction_shards(shards: Iterable[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    materialized = tuple(shards)
    indices = tuple(int(shard.get("shard_index", -1)) for shard in materialized)
    ensure_exact_shard_cover(indices)
    for shard in materialized:
        _validate_prediction_shard(shard)
    ordered = tuple(sorted(materialized, key=lambda shard: int(shard["shard_index"])))
    anchor = ordered[0]
    shared_fields = (
        "frozen_implementation_digest",
        "run_id",
        "workflow_digest",
        "root",
        "beacon",
        "challenge_materialization_digest",
        "selection_manifest_digest",
        "challenge_manifest_digest",
    )
    for shard in ordered[1:]:
        for field in shared_fields:
            if shard.get(field) != anchor.get(field):
                raise ValueError(f"prediction shard {field} mismatch")
    commitments: list[dict[str, object]] = []
    for shard in ordered:
        commitments.extend(dict(item) for item in shard["commitments"])  # type: ignore[index]
    expected_total = WORLDS_PER_ROOT * len(FROZEN_ARMS) * len(FROZEN_EFFORTS)
    if len(commitments) != expected_total:
        raise ValueError("merged prediction grid is incomplete")
    return tuple(commitments)


def build_root_recovery_receipt(
    *,
    root: int,
    run_id: int | str,
    workflow_digest: str,
    selection_manifest_digest: str,
    challenge_manifest_digest: str,
    root_evidence_artifact_digest: str,
    root_run_identity: str,
    shard_digests: Iterable[str],
) -> dict[str, object]:
    root = _require_root(root)
    workflow_digest = _require_hex64(workflow_digest, field="workflow_digest")
    selection_manifest_digest = _require_hex64(
        selection_manifest_digest, field="selection_manifest_digest"
    )
    challenge_manifest_digest = _require_hex64(
        challenge_manifest_digest, field="challenge_manifest_digest"
    )
    root_evidence_artifact_digest = _require_hex64(
        root_evidence_artifact_digest, field="root_evidence_artifact_digest"
    )
    root_run_identity = _require_hex64(root_run_identity, field="root_run_identity")
    shard_digests = tuple(shard_digests)
    if len(shard_digests) != SHARD_COUNT or len(set(shard_digests)) != SHARD_COUNT:
        raise ValueError("root recovery requires exactly sixteen unique shard digests")
    for index, digest in enumerate(shard_digests):
        _require_hex64(digest, field=f"shard_digests[{index}]")
    run_id_text = _run_id_text(run_id)
    payload = {
        "schema": ROOT_RECOVERY_RECEIPT_SCHEMA,
        "recovery_schema": RECOVERY_SCHEMA,
        "frozen_implementation_digest": FROZEN_IMPLEMENTATION_DIGEST,
        "prior_failed_run_id": PRIOR_FAILED_RUN_ID,
        "run_id": run_id_text,
        "workflow_digest": workflow_digest,
        "root": root,
        "beacon": recovery_beacon(run_id_text),
        "selection_manifest_digest": selection_manifest_digest,
        "challenge_manifest_digest": challenge_manifest_digest,
        "root_evidence_artifact_digest": root_evidence_artifact_digest,
        "root_run_identity": root_run_identity,
        "shard_digests": shard_digests,
    }
    return _seal(payload, field="receipt_digest")


def build_cross_root_recovery_envelope(
    *,
    cross_root_artifact: Mapping[str, object],
    root_recovery_receipts: Iterable[Mapping[str, object]],
    run_id: int | str,
    workflow_digest: str,
) -> dict[str, object]:
    if cross_root_artifact.get("schema") != "EXP301-CROSS-ROOT-SCIENTIFIC-EVIDENCE-V1":
        raise ValueError("cross-root artifact schema mismatch")
    _verify_seal(cross_root_artifact, field="artifact_digest")
    _require_frozen_digest(cross_root_artifact.get("frozen_implementation_digest"))
    if tuple(cross_root_artifact.get("roots", ())) != (0, 1, 2, 3):
        raise ValueError("cross-root artifact must contain exactly roots 0..3")
    if cross_root_artifact.get("scientific_evidence_eligible") is not True:
        raise ValueError("cross-root artifact is not scientific-evidence eligible")
    analysis = cross_root_artifact.get("analysis")
    if not isinstance(analysis, Mapping) or not isinstance(analysis.get("decision"), str):
        raise ValueError("cross-root analysis is missing decision")
    if analysis.get("exp302_implementation_authorized") is not False:
        raise ValueError("EXP-301R cannot authorize EXP-302 implementation")
    if analysis.get("scale_authorized") is not False:
        raise ValueError("EXP-301R cannot authorize scale")

    roots = tuple(root_recovery_receipts)
    if len(roots) != 4 or {item.get("root") for item in roots} != {0, 1, 2, 3}:
        raise ValueError("recovery envelope requires exactly four root receipts")
    ordered = tuple(sorted(roots, key=lambda item: int(item["root"])))
    run_id_text = _run_id_text(run_id)
    workflow_digest = _require_hex64(workflow_digest, field="workflow_digest")
    cross_digests = tuple(cross_root_artifact.get("root_artifact_digests", ()))
    cross_run_ids = tuple(cross_root_artifact.get("root_run_identities", ()))
    if len(cross_digests) != 4 or len(cross_run_ids) != 4:
        raise ValueError("cross-root artifact root bindings are incomplete")
    for index, receipt in enumerate(ordered):
        if receipt.get("schema") != ROOT_RECOVERY_RECEIPT_SCHEMA:
            raise ValueError("root recovery receipt schema mismatch")
        _verify_seal(receipt, field="receipt_digest")
        _require_frozen_digest(receipt.get("frozen_implementation_digest"))
        if receipt.get("run_id") != run_id_text:
            raise ValueError("root recovery receipt run id mismatch")
        if receipt.get("workflow_digest") != workflow_digest:
            raise ValueError("root recovery receipt workflow digest mismatch")
        if receipt.get("beacon") != recovery_beacon(run_id_text):
            raise ValueError("root recovery receipt beacon mismatch")
        if receipt.get("root_evidence_artifact_digest") != cross_digests[index]:
            raise ValueError("root recovery receipt evidence digest mismatch")
        if receipt.get("root_run_identity") != cross_run_ids[index]:
            raise ValueError("root recovery receipt run identity mismatch")

    payload = {
        "schema": CROSS_ROOT_RECOVERY_ENVELOPE_SCHEMA,
        "recovery_schema": RECOVERY_SCHEMA,
        "frozen_implementation_digest": FROZEN_IMPLEMENTATION_DIGEST,
        "prior_failed_run_id": PRIOR_FAILED_RUN_ID,
        "original_monolithic_run_completed": False,
        "run_id": run_id_text,
        "workflow_digest": workflow_digest,
        "beacon": recovery_beacon(run_id_text),
        "root_recovery_receipt_digests": tuple(item["receipt_digest"] for item in ordered),
        "cross_root_artifact_digest": cross_root_artifact["artifact_digest"],
        "decision": analysis["decision"],
        "exp302_implementation_authorized": False,
        "scale_authorized": False,
        "scientific_evidence_eligible": True,
    }
    return _seal(payload, field="envelope_digest")