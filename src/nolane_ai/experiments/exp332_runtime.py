from __future__ import annotations
from dataclasses import asdict
import hashlib, json
from pathlib import Path
from typing import Any, Mapping

from .exp323_evidence import expected_reconstruction_payload
from .exp323r_repair import load_locked_reconstruction
from .exp331_contract import AUTHORIZATION_FLAGS, MODES, PAIR_IDS, PAIR_MEMBERS, PARENT_PAIR_PASS, arm_pass, sham_equivalent
from .exp331_runtime import _result, _run_arm, validate_exp330_parent
from .exp332_contract import (
    EXP327_ATTEMPT1_EVIDENCE_DIGEST,
    EXP327_ATTEMPT2_EVIDENCE_DIGEST,
    EXP327_EXECUTION_DIGEST,
    EXP327_SOURCE_SHA,
    EXP327_WORKFLOW_SHA256,
    EXP330_EVIDENCE_DIGEST,
    metric_vector_equal,
    reduce_portable_pair_lattice,
)
from .exp332_identity import Exp332ExecutionIdentity, validate_execution_identity

FINAL_SCHEMA = "EXP332-FINAL-EVIDENCE-V1"

def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()

def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()

def _validate_exp327_witness(payload: Mapping[str, Any], expected_digest: str) -> None:
    if payload.get("schema") != "EXP327-FINAL-EVIDENCE-V1":
        raise ValueError("EXP-332 EXP327 witness schema")
    if payload.get("decision") != "PAIR_MINIMAL_FAILURE_PRESENT" or payload.get("failed_pairs") != ["P02"]:
        raise ValueError("EXP-332 EXP327 witness result")
    if payload.get("evidence_digest") != expected_digest:
        raise ValueError("EXP-332 EXP327 witness digest")
    materialized = dict(payload)
    materialized.pop("evidence_digest", None)
    if _digest(materialized) != expected_digest:
        raise ValueError("EXP-332 EXP327 witness canonical")
    identity = payload.get("execution_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("EXP-332 EXP327 witness identity")
    if (
        identity.get("source_commit_sha") != EXP327_SOURCE_SHA
        or identity.get("workflow_sha256") != EXP327_WORKFLOW_SHA256
        or identity.get("exp327_execution_digest") != EXP327_EXECUTION_DIGEST
    ):
        raise ValueError("EXP-332 EXP327 witness source identity")
    if {pair: bool(payload.get("group_pass", {}).get(pair)) for pair in PAIR_IDS} != PARENT_PAIR_PASS:
        raise ValueError("EXP-332 EXP327 witness pair pass vector")
    groups = payload.get("groups")
    if not isinstance(groups, list):
        raise ValueError("EXP-332 EXP327 witness groups")
    by = {group.get("group_id"): group for group in groups if isinstance(group, Mapping)}
    if any(pair not in by for pair in PAIR_IDS):
        raise ValueError("EXP-332 EXP327 witness pair coverage")
    for pair in PAIR_IDS:
        group = by[pair]
        if group.get("members") != list(PAIR_MEMBERS[pair]) or group.get("invalid_reason") is not None:
            raise ValueError("EXP-332 EXP327 witness pair identity")
        snapshots = [row for row in group.get("snapshots", []) if row.get("exposures_per_world") == 32]
        if len(snapshots) != 1:
            raise ValueError("EXP-332 EXP327 witness final snapshot")
        state = group.get("final_state")
        if not isinstance(state, Mapping) or any(not isinstance(state.get(k), str) or len(state[k]) != 64 for k in ("model_state_digest", "optimizer_state_digest", "rng_state_digest")):
            raise ValueError("EXP-332 EXP327 witness final state")
    for key, expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"EXP-332 EXP327 authorization drift: {key}")

def _anchor(parent: Mapping[str, Any], pair: str) -> dict[str, Any]:
    group = next(group for group in parent["groups"] if group["group_id"] == pair)
    snapshot = next(row for row in group["snapshots"] if row["exposures_per_world"] == 32)
    return {
        "members": group["members"],
        "passed": bool(group["passed"]),
        "world_token_accuracies": snapshot["world_token_accuracies"],
        "world_full_answer_exact": snapshot["world_full_answer_exact"],
        "final_state": group["final_state"],
    }

def _witness_consistency(first: Mapping[str, Any], second: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
    divergence: list[str] = []
    for pair in PAIR_IDS:
        left, right = _anchor(first, pair), _anchor(second, pair)
        behavior_equal = (
            left["members"] == right["members"]
            and left["passed"] is right["passed"]
            and metric_vector_equal(left["world_token_accuracies"], right["world_token_accuracies"])
            and metric_vector_equal(left["world_full_answer_exact"], right["world_full_answer_exact"])
            and left["final_state"]["rng_state_digest"] == right["final_state"]["rng_state_digest"]
        )
        if not behavior_equal:
            return False, ()
        if (
            left["final_state"]["model_state_digest"] != right["final_state"]["model_state_digest"]
            or left["final_state"]["optimizer_state_digest"] != right["final_state"]["optimizer_state_digest"]
        ):
            divergence.append(pair)
    return True, tuple(divergence)

def _control_matches_behavior(record: Mapping[str, Any], first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return (
        record.get("mode") == "CONTROL"
        and record.get("members") == first["members"] == second["members"]
        and metric_vector_equal(record.get("world_token_accuracies"), first["world_token_accuracies"])
        and metric_vector_equal(record.get("world_token_accuracies"), second["world_token_accuracies"])
        and metric_vector_equal(record.get("world_full_answer_exact"), first["world_full_answer_exact"])
        and metric_vector_equal(record.get("world_full_answer_exact"), second["world_full_answer_exact"])
        and record.get("rng_state_digest") == first["final_state"]["rng_state_digest"] == second["final_state"]["rng_state_digest"]
        and record.get("nonfinite_events") == 0
        and arm_pass(_result(record)) is first["passed"] is second["passed"]
    )

def run_court(
    *,
    checkpoint_path: str | Path,
    receipt_path: str | Path,
    selection_lock_path: str | Path,
    exp330_parent_path: str | Path,
    exp327_original_path: str | Path,
    exp327_replay_path: str | Path,
    execution_identity: Exp332ExecutionIdentity,
) -> dict[str, Any]:
    validate_execution_identity(execution_identity)
    p330 = json.loads(Path(exp330_parent_path).read_text(encoding="utf-8"))
    p327a = json.loads(Path(exp327_original_path).read_text(encoding="utf-8"))
    p327b = json.loads(Path(exp327_replay_path).read_text(encoding="utf-8"))
    if not all(isinstance(x, Mapping) for x in (p330, p327a, p327b)):
        raise ValueError("EXP-332 parent payload")
    validate_exp330_parent(p330)
    _validate_exp327_witness(p327a, EXP327_ATTEMPT1_EVIDENCE_DIGEST)
    _validate_exp327_witness(p327b, EXP327_ATTEMPT2_EVIDENCE_DIGEST)
    witness_behavior_equal, divergence_pairs = _witness_consistency(p327a, p327b)
    divergence_confirmed = witness_behavior_equal and divergence_pairs == PAIR_IDS

    authority = load_locked_reconstruction(checkpoint_path, receipt_path, selection_lock_path)
    reconstruction = dict(authority.reconstruction)
    del authority

    arms = [
        _run_arm(checkpoint_path, receipt_path, selection_lock_path, pair_id=pair, mode=mode)
        for pair in PAIR_IDS
        for mode in MODES
    ]
    results = {f"{arm['pair_id']}:{arm['mode']}": _result(arm) for arm in arms}
    anchors_a = {pair: _anchor(p327a, pair) for pair in PAIR_IDS}
    anchors_b = {pair: _anchor(p327b, pair) for pair in PAIR_IDS}
    parent_behavior_reproduced = all(
        _control_matches_behavior(
            next(arm for arm in arms if arm["pair_id"] == pair and arm["mode"] == "CONTROL"),
            anchors_a[pair],
            anchors_b[pair],
        )
        for pair in PAIR_IDS
    )
    invalid = any(arm.get("invalid_reason") is not None for arm in arms)
    decision, passed, regressions = reduce_portable_pair_lattice(
        results,
        parent_behavior_reproduced=parent_behavior_reproduced,
        witness_divergence_confirmed=divergence_confirmed,
        invalid=invalid,
    )
    replay_state_match = {
        pair: (
            next(arm for arm in arms if arm["pair_id"] == pair and arm["mode"] == "CONTROL")["model_state_digest"] == anchors_b[pair]["final_state"]["model_state_digest"]
            and next(arm for arm in arms if arm["pair_id"] == pair and arm["mode"] == "CONTROL")["optimizer_state_digest"] == anchors_b[pair]["final_state"]["optimizer_state_digest"]
        )
        for pair in PAIR_IDS
    }
    payload = {
        "schema": FINAL_SCHEMA,
        "execution_identity": asdict(execution_identity),
        "reconstruction": reconstruction,
        "decision": decision,
        "parent_exp330_evidence_digest": EXP330_EVIDENCE_DIGEST,
        "exp327_attempt1_evidence_digest": EXP327_ATTEMPT1_EVIDENCE_DIGEST,
        "exp327_attempt2_evidence_digest": EXP327_ATTEMPT2_EVIDENCE_DIGEST,
        "witness_behavior_equal": witness_behavior_equal,
        "historical_state_divergence_pairs": list(divergence_pairs),
        "parent_behavior_reproduced": parent_behavior_reproduced,
        "control_matches_attempt2_state": replay_state_match,
        "arms": arms,
        "arm_pass": passed,
        "pair_sham_match": {
            pair: sham_equivalent(results[f"{pair}:CONTROL"], results[f"{pair}:SHAM"])
            for pair in PAIR_IDS
        },
        "project_regressions": list(regressions),
        "project_pass_pairs": [pair for pair in PAIR_IDS if passed.get(f"{pair}:PROJECT", False)],
        "p02_projection_event_count": results["P02:PROJECT"].projection_event_count,
        **AUTHORIZATION_FLAGS,
    }
    payload["evidence_digest"] = _digest(payload)
    validate_final_evidence(payload)
    return payload

def validate_final_evidence(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != FINAL_SCHEMA:
        raise ValueError("EXP-332 final schema")
    raw = payload.get("execution_identity")
    if not isinstance(raw, Mapping):
        raise ValueError("EXP-332 identity missing")
    validate_execution_identity(Exp332ExecutionIdentity(**raw))
    if payload.get("reconstruction") != expected_reconstruction_payload():
        raise ValueError("EXP-332 reconstruction")
    if (
        payload.get("parent_exp330_evidence_digest") != EXP330_EVIDENCE_DIGEST
        or payload.get("exp327_attempt1_evidence_digest") != EXP327_ATTEMPT1_EVIDENCE_DIGEST
        or payload.get("exp327_attempt2_evidence_digest") != EXP327_ATTEMPT2_EVIDENCE_DIGEST
    ):
        raise ValueError("EXP-332 authority")
    if payload.get("witness_behavior_equal") is not True or payload.get("historical_state_divergence_pairs") != list(PAIR_IDS):
        raise ValueError("EXP-332 reproducibility witness")
    arms = payload.get("arms")
    if not isinstance(arms, list) or len(arms) != len(PAIR_IDS) * len(MODES):
        raise ValueError("EXP-332 arm coverage")
    results = {f"{arm.get('pair_id')}:{arm.get('mode')}": _result(arm) for arm in arms}
    invalid = any(arm.get("invalid_reason") is not None for arm in arms)
    decision, passed, regressions = reduce_portable_pair_lattice(
        results,
        parent_behavior_reproduced=payload.get("parent_behavior_reproduced") is True,
        witness_divergence_confirmed=True,
        invalid=invalid,
    )
    if payload.get("decision") != decision or payload.get("arm_pass") != passed or payload.get("project_regressions") != list(regressions):
        raise ValueError("EXP-332 reducer")
    expected_sham = {pair: sham_equivalent(results[f"{pair}:CONTROL"], results[f"{pair}:SHAM"]) for pair in PAIR_IDS}
    if payload.get("pair_sham_match") != expected_sham:
        raise ValueError("EXP-332 sham integrity")
    if payload.get("p02_projection_event_count") != results["P02:PROJECT"].projection_event_count:
        raise ValueError("EXP-332 projection count")
    for key, expected in AUTHORIZATION_FLAGS.items():
        if payload.get(key) is not expected:
            raise ValueError(f"EXP-332 authorization drift: {key}")
    materialized = dict(payload)
    claimed = materialized.pop("evidence_digest", None)
    if not isinstance(claimed, str) or _digest(materialized) != claimed:
        raise ValueError("EXP-332 evidence digest")
