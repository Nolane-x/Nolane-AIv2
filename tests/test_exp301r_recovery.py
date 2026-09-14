from __future__ import annotations

import json
from pathlib import Path

import pytest

from nolane_ai.experiments.exp301r_recovery import (
    FROZEN_ARMS,
    FROZEN_EFFORTS,
    FROZEN_IMPLEMENTATION_DIGEST,
    PRIOR_FAILED_RUN_ID,
    RECOVERY_SCHEMA,
    SHARD_COUNT,
    WORLDS_PER_ROOT,
    WORLDS_PER_SHARD,
    build_challenge_manifest,
    build_cross_root_recovery_envelope,
    build_prediction_shard,
    build_root_recovery_receipt,
    build_selection_recovery_receipt,
    build_trial_recovery_receipt,
    canonical_digest,
    challenge_shard_bounds,
    ensure_exact_shard_cover,
    merge_prediction_shards,
    recovery_beacon,
    resolve_trial_plan,
    validate_recovery_changed_paths,
    write_once_json,
)


class _Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _hex(ch: str) -> str:
    return ch * 64


def test_recovery_constants_bind_failed_attempt_and_fixed_geometry() -> None:
    assert RECOVERY_SCHEMA == "EXP301R-STANDARD-RUNNER-SHARDED-RECOVERY-V1"
    assert PRIOR_FAILED_RUN_ID == 34823251660
    assert SHARD_COUNT == 16
    assert WORLDS_PER_ROOT == 512
    assert WORLDS_PER_SHARD == 32
    assert FROZEN_ARMS == ("A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE")
    assert FROZEN_EFFORTS == (1, 2, 4, 8, 12, 16)


def test_recovery_beacon_is_stable_for_run_id_and_excludes_attempt() -> None:
    assert recovery_beacon("34899900000") == "github-run-34899900000-exp301r-v1"
    assert recovery_beacon(34899900000) == "github-run-34899900000-exp301r-v1"
    with pytest.raises(ValueError, match="run_id"):
        recovery_beacon("")
    with pytest.raises(ValueError, match="run_id"):
        recovery_beacon("not-a-run")


def test_challenge_shard_bounds_cover_exactly_512_worlds_without_overlap() -> None:
    assert challenge_shard_bounds(0) == (0, 32)
    assert challenge_shard_bounds(15) == (480, 512)
    assert tuple(challenge_shard_bounds(i) for i in range(16)) == tuple(
        (i * 32, (i + 1) * 32) for i in range(16)
    )
    assert ensure_exact_shard_cover(tuple(range(16))) == tuple(range(16))

    with pytest.raises(ValueError, match="shard_index"):
        challenge_shard_bounds(-1)
    with pytest.raises(ValueError, match="shard_index"):
        challenge_shard_bounds(16)
    with pytest.raises(ValueError, match="exactly"):
        ensure_exact_shard_cover(tuple(range(15)))
    with pytest.raises(ValueError, match="exactly"):
        ensure_exact_shard_cover(tuple(range(16)) + (15,))


def test_canonical_digest_is_order_independent_for_object_keys() -> None:
    assert canonical_digest({"b": 2, "a": 1}) == canonical_digest({"a": 1, "b": 2})
    assert len(canonical_digest({"a": 1})) == 64


def test_write_once_json_is_canonical_and_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    payload = {"b": 2, "a": 1}
    write_once_json(target, payload)
    assert target.read_text(encoding="utf-8") == '{"a":1,"b":2}\n'
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError):
        write_once_json(target, payload)


def test_recovery_changed_path_allowlist_rejects_scientific_core_modification() -> None:
    allowed = (
        ".github/workflows/exp301r-standard-runner-sharded-recovery.yml",
        "docs/superpowers/specs/2026-09-14-exp301r-standard-runner-sharded-recovery-design.md",
        "docs/superpowers/plans/2026-09-14-exp301r-standard-runner-sharded-recovery.md",
        "src/nolane_ai/experiments/exp301r_recovery.py",
        "scripts/exp301r_train_trial.py",
        "scripts/verify_exp301r_recovery.py",
        "tests/test_exp301r_recovery.py",
    )
    assert validate_recovery_changed_paths(allowed) == tuple(sorted(allowed))

    forbidden = allowed + ("src/nolane_ai/experiments/exp301_scientific.py",)
    with pytest.raises(ValueError, match="forbidden recovery path"):
        validate_recovery_changed_paths(forbidden)


def test_recovery_changed_path_allowlist_rejects_unrelated_files() -> None:
    with pytest.raises(ValueError, match="forbidden recovery path"):
        validate_recovery_changed_paths(("README.md",))


def test_resolve_trial_plan_selects_only_frozen_coordinate() -> None:
    plans = (
        _Obj(root=2, arm_id="C_NRS_CORE", trial_index=0, learning_rate=1e-4, model_init_seed=10),
        _Obj(root=2, arm_id="C_NRS_CORE", trial_index=1, learning_rate=3e-4, model_init_seed=11),
    )
    selected = resolve_trial_plan(plans, root=2, arm_id="C_NRS_CORE", trial_index=1)
    assert selected is plans[1]
    with pytest.raises(ValueError, match="exactly one frozen trial"):
        resolve_trial_plan(plans, root=2, arm_id="C_NRS_CORE", trial_index=2)


def test_trial_recovery_receipt_binds_frozen_trial_identity() -> None:
    plan = _Obj(root=2, arm_id="C_NRS_CORE", trial_index=1, learning_rate=3e-4, model_init_seed=11)
    result = _Obj(
        root=2,
        arm_id="C_NRS_CORE",
        learning_rate=3e-4,
        model_init_seed=11,
        training_steps=512,
        checkpoint_digest=_hex("a"),
        trial_receipt_digest=_hex("b"),
    )
    receipt = build_trial_recovery_receipt(
        plan,
        result,
        frozen_implementation_digest=FROZEN_IMPLEMENTATION_DIGEST,
        run_id=999,
        workflow_digest=_hex("c"),
    )
    assert receipt["schema"] == "EXP301R-TRIAL-RECEIPT-V1"
    assert receipt["prior_failed_run_id"] == PRIOR_FAILED_RUN_ID
    assert receipt["training_steps"] == 512
    assert receipt["trial_index"] == 1
    digest = receipt["receipt_digest"]
    unsigned = dict(receipt)
    unsigned.pop("receipt_digest")
    assert digest == canonical_digest(unsigned)

    bad = _Obj(**{**result.__dict__, "training_steps": 511})
    with pytest.raises(ValueError, match="512"):
        build_trial_recovery_receipt(
            plan,
            bad,
            frozen_implementation_digest=FROZEN_IMPLEMENTATION_DIGEST,
            run_id=999,
            workflow_digest=_hex("c"),
        )


def test_selection_recovery_receipt_requires_six_trials_and_three_selected_checkpoints() -> None:
    receipt = build_selection_recovery_receipt(
        root=1,
        selection_manifest_digest=_hex("a"),
        selected_checkpoint_digests=(_hex("b"), _hex("c"), _hex("d")),
        trial_recovery_receipt_digests=tuple(f"{i:064x}" for i in range(6)),
        run_id=1000,
        workflow_digest=_hex("e"),
    )
    assert receipt["schema"] == "EXP301R-SELECTION-RECEIPT-V1"
    assert len(receipt["trial_recovery_receipt_digests"]) == 6
    with pytest.raises(ValueError, match="six"):
        build_selection_recovery_receipt(
            root=1,
            selection_manifest_digest=_hex("a"),
            selected_checkpoint_digests=(_hex("b"), _hex("c"), _hex("d")),
            trial_recovery_receipt_digests=(_hex("f"),),
            run_id=1000,
            workflow_digest=_hex("e"),
        )


def test_challenge_manifest_binds_all_512_ordered_world_ids_and_stable_beacon() -> None:
    challenge = _Obj(
        root=1,
        challenge_nonce=_hex("a"),
        materialization_digest=_hex("b"),
        worlds=tuple(_Obj(content_id=f"world-{i:03d}") for i in range(512)),
    )
    manifest = build_challenge_manifest(
        challenge,
        run_id=12345,
        workflow_digest=_hex("c"),
        selection_manifest_digest=_hex("d"),
    )
    assert manifest["schema"] == "EXP301R-CHALLENGE-MANIFEST-V1"
    assert manifest["beacon"] == "github-run-12345-exp301r-v1"
    assert len(manifest["ordered_content_ids"]) == 512
    assert manifest["ordered_content_ids"][0] == "world-000"
    assert manifest["ordered_content_ids"][-1] == "world-511"


def _commitments_for_shard(root: int, content_ids: tuple[str, ...]) -> tuple[dict[str, object], ...]:
    items = []
    counter = 1
    for content_id in content_ids:
        for arm in FROZEN_ARMS:
            for effort in FROZEN_EFFORTS:
                items.append(
                    {
                        "root": root,
                        "content_id": content_id,
                        "arm_id": arm,
                        "effort_multiplier": effort,
                        "generation_token_count": 96,
                        "prediction_digest": f"{counter:064x}",
                    }
                )
                counter += 1
    return tuple(items)


def test_prediction_shard_requires_canonical_32_world_grid_and_never_contains_scores() -> None:
    challenge = _Obj(
        root=0,
        challenge_nonce=_hex("a"),
        materialization_digest=_hex("b"),
        worlds=tuple(_Obj(content_id=f"w{i:03d}") for i in range(512)),
    )
    manifest = build_challenge_manifest(
        challenge,
        run_id=99,
        workflow_digest=_hex("c"),
        selection_manifest_digest=_hex("d"),
    )
    ids = tuple(manifest["ordered_content_ids"][:32])
    shard = build_prediction_shard(
        challenge_manifest=manifest,
        shard_index=0,
        commitments=_commitments_for_shard(0, ids),
        run_id=99,
        workflow_digest=_hex("c"),
        selection_manifest_digest=_hex("d"),
    )
    assert shard["schema"] == "EXP301R-PREDICTION-SHARD-V1"
    assert shard["world_start"] == 0
    assert shard["world_end"] == 32
    assert len(shard["commitments"]) == 576
    text = json.dumps(shard, sort_keys=True)
    assert "verified_success" not in text
    assert "evaluation_rows" not in text

    wrong_order = list(_commitments_for_shard(0, ids))
    wrong_order[0], wrong_order[1] = wrong_order[1], wrong_order[0]
    with pytest.raises(ValueError, match="canonical"):
        build_prediction_shard(
            challenge_manifest=manifest,
            shard_index=0,
            commitments=tuple(wrong_order),
            run_id=99,
            workflow_digest=_hex("c"),
            selection_manifest_digest=_hex("d"),
        )


def test_merge_prediction_shards_rejects_missing_or_duplicate_and_restores_order() -> None:
    challenge = _Obj(
        root=3,
        challenge_nonce=_hex("a"),
        materialization_digest=_hex("b"),
        worlds=tuple(_Obj(content_id=f"w{i:03d}") for i in range(512)),
    )
    manifest = build_challenge_manifest(
        challenge,
        run_id=77,
        workflow_digest=_hex("c"),
        selection_manifest_digest=_hex("d"),
    )
    shards = []
    for index in range(16):
        start, end = challenge_shard_bounds(index)
        ids = tuple(manifest["ordered_content_ids"][start:end])
        shards.append(
            build_prediction_shard(
                challenge_manifest=manifest,
                shard_index=index,
                commitments=_commitments_for_shard(3, ids),
                run_id=77,
                workflow_digest=_hex("c"),
                selection_manifest_digest=_hex("d"),
            )
        )
    merged = merge_prediction_shards(tuple(reversed(shards)))
    assert len(merged) == 512 * 3 * 6
    assert merged[0]["content_id"] == "w000"
    assert merged[-1]["content_id"] == "w511"
    with pytest.raises(ValueError, match="exactly"):
        merge_prediction_shards(tuple(shards[:-1]))
    with pytest.raises(ValueError, match="exactly"):
        merge_prediction_shards(tuple(shards[:-1] + [shards[-2]]))


def test_root_recovery_receipt_binds_complete_shard_set_and_root_evidence() -> None:
    receipt = build_root_recovery_receipt(
        root=2,
        run_id=55,
        workflow_digest=_hex("a"),
        selection_manifest_digest=_hex("b"),
        challenge_manifest_digest=_hex("c"),
        root_evidence_artifact_digest=_hex("d"),
        root_run_identity=_hex("e"),
        shard_digests=tuple(f"{i + 1:064x}" for i in range(16)),
    )
    assert receipt["schema"] == "EXP301R-ROOT-RECOVERY-RECEIPT-V1"
    assert len(receipt["shard_digests"]) == 16


def test_cross_root_recovery_envelope_copies_reducer_decision_without_new_authority() -> None:
    analysis = {
        "decision": "KILL_H_RD_01",
        "exp302_implementation_authorized": False,
        "scale_authorized": False,
    }
    cross = {
        "schema": "EXP301-CROSS-ROOT-SCIENTIFIC-EVIDENCE-V1",
        "frozen_implementation_digest": FROZEN_IMPLEMENTATION_DIGEST,
        "roots": [0, 1, 2, 3],
        "root_artifact_digests": [f"{i + 1:064x}" for i in range(4)],
        "root_run_identities": [f"{i + 10:064x}" for i in range(4)],
        "bootstrap_samples": 10000,
        "analysis": analysis,
        "scientific_evidence_eligible": True,
    }
    cross["artifact_digest"] = canonical_digest(cross)

    roots = []
    for root in range(4):
        roots.append(
            build_root_recovery_receipt(
                root=root,
                run_id=55,
                workflow_digest=_hex("a"),
                selection_manifest_digest=f"{root + 20:064x}",
                challenge_manifest_digest=f"{root + 30:064x}",
                root_evidence_artifact_digest=cross["root_artifact_digests"][root],
                root_run_identity=cross["root_run_identities"][root],
                shard_digests=tuple(f"{root * 16 + i + 100:064x}" for i in range(16)),
            )
        )

    envelope = build_cross_root_recovery_envelope(
        cross_root_artifact=cross,
        root_recovery_receipts=tuple(roots),
        run_id=55,
        workflow_digest=_hex("a"),
    )
    assert envelope["schema"] == "EXP301R-CROSS-ROOT-RECOVERY-ENVELOPE-V1"
    assert envelope["decision"] == "KILL_H_RD_01"
    assert envelope["exp302_implementation_authorized"] is False
    assert envelope["scale_authorized"] is False
    assert envelope["prior_failed_run_id"] == PRIOR_FAILED_RUN_ID
    assert envelope["original_monolithic_run_completed"] is False
