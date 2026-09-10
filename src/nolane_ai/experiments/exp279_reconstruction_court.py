from __future__ import annotations

from copy import deepcopy
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256
from .exp279_confirmatory_authorization import validate_exp279_gate_a_seal
from .exp279_routing_worlds import STRATA


SCHEMA = "NLM-EXP-279-CONFIRMATORY-RECONSTRUCTION-AUTH-V1"
STATUS = "RECONSTRUCTION_AUTHORIZED_NOT_EXECUTED"
EXPERIMENT_ID = "EXP-279"
_FORBIDDEN_PREFREEZE_TERMS = (
    "beacon_receipt",
    "challenge_seed",
    "challenge_batch",
    "challenge_entropy",
)


def _reconstruction_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("reconstruction_digest", None)
    return canonical_sha256(clean)


def _binding_digest(payload: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            "seal_digest": payload.get("seal_digest"),
            "seal_snapshot_digest": payload.get("seal_snapshot_digest"),
            "protocol_digest": payload.get("protocol_digest"),
            "source_tree_digest": payload.get("source_tree_digest"),
            "freeze_commit_sha": payload.get("freeze_commit_sha"),
            "freeze_commit_timestamp_utc": payload.get("freeze_commit_timestamp_utc"),
            "checkpoint_seal_created_at_utc": payload.get(
                "checkpoint_seal_created_at_utc"
            ),
            "confirmatory_n": payload.get("confirmatory_n"),
            "reserved_replicate_ids": payload.get("reserved_replicate_ids"),
            "stratum_schedule": payload.get("stratum_schedule"),
            "route_threshold": payload.get("route_threshold"),
            "checkpoint_scientific_identity_digest": payload.get(
                "checkpoint_scientific_identity_digest"
            ),
            "checkpoint_receipt_digest": payload.get(
                "checkpoint_receipt_digest"
            ),
            "checkpoint_replay_contract_digest": payload.get(
                "checkpoint_replay_contract_digest"
            ),
            "checkpoint_final_state_digest": payload.get(
                "checkpoint_final_state_digest"
            ),
            "checkpoint_file_sha256": payload.get("checkpoint_file_sha256"),
            "accounted_flops_contract": payload.get("accounted_flops_contract"),
            "model_geometry": payload.get("model_geometry"),
            "information_separation": payload.get("information_separation"),
            "reconstruction_code_digest": payload.get(
                "reconstruction_code_digest"
            ),
        }
    )


def _valid_digest(value: Any, *, size: int = 64) -> bool:
    return isinstance(value, str) and len(value) == size


def _prefreeze_errors(payload: dict[str, Any]) -> list[str]:
    rendered = repr(payload).lower()
    return [
        f"EXP-279 reconstruction contains forbidden pre-freeze material: {term}"
        for term in _FORBIDDEN_PREFREEZE_TERMS
        if term in rendered
    ]


def _sealed_authorization(seal: dict[str, Any]) -> dict[str, Any]:
    errors = validate_exp279_gate_a_seal(seal)
    if errors:
        raise ValueError("invalid EXP-279 Gate A seal: " + "; ".join(errors))
    snapshot = seal.get("authorization_snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("EXP-279 Gate A seal authorization snapshot missing")
    return snapshot


def _validate_accounted_flops_contract(
    contract: Any,
    *,
    ceiling: Any,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(contract, dict) or set(contract) != {
        "propagation_only",
        "branch_only",
        "hybrid",
    }:
        return ["EXP-279 reconstruction accounted FLOP contract surface invalid"]
    valid_ceiling = (
        isinstance(ceiling, int)
        and not isinstance(ceiling, bool)
        and ceiling > 0
    )
    for arm_id in ("propagation_only", "branch_only"):
        item = contract.get(arm_id) or {}
        value = item.get("accounted_flops_per_episode")
        if (
            item.get("mode") != "fixed"
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            or (valid_ceiling and value > ceiling)
        ):
            errors.append(
                f"EXP-279 reconstruction {arm_id} fixed FLOP contract invalid"
            )
    hybrid = contract.get("hybrid") or {}
    stop_cost = hybrid.get("stop_accounted_flops_per_episode")
    branch_cost = hybrid.get("branch_accounted_flops_per_episode")
    if (
        hybrid.get("mode") != "route_dependent"
        or not isinstance(stop_cost, int)
        or isinstance(stop_cost, bool)
        or not isinstance(branch_cost, int)
        or isinstance(branch_cost, bool)
        or stop_cost <= 0
        or branch_cost < stop_cost
        or (valid_ceiling and branch_cost > ceiling)
    ):
        errors.append(
            "EXP-279 reconstruction hybrid route-dependent FLOP contract invalid"
        )
    return errors


def build_exp279_reconstruction_authorization(
    *,
    seal: dict[str, Any],
    reconstruction_code_digest: str,
) -> dict[str, Any]:
    sealed = _sealed_authorization(seal)
    if not _valid_digest(reconstruction_code_digest):
        raise ValueError(
            "EXP-279 reconstruction code digest must be a 64-character digest"
        )
    machinery = sealed.get("machinery_digests") or {}
    if machinery.get("reconstruction_code_digest") != reconstruction_code_digest:
        raise ValueError(
            "EXP-279 reconstruction code digest does not match Gate A machinery freeze"
        )

    checkpoint = sealed.get("checkpoint") or {}
    resource = sealed.get("resource_court") or {}
    accounted_flops = deepcopy(resource.get("accounted_flops_contract") or {})
    flop_errors = _validate_accounted_flops_contract(
        accounted_flops,
        ceiling=resource.get("declared_max_accounted_flops_per_episode"),
    )
    if flop_errors:
        raise ValueError("; ".join(flop_errors))
    if sealed.get("stratum_schedule") != list(STRATA):
        raise ValueError("EXP-279 reconstruction Gate A stratum schedule drift")

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "status": STATUS,
        "confirmatory_ready": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "seed_materialization_status": "NOT_EXECUTED",
        "decision_rule_executed": False,
        "seal_digest": seal.get("seal_digest"),
        "seal_snapshot": deepcopy(seal),
        "seal_snapshot_digest": canonical_sha256(seal),
        "protocol_digest": sealed.get("protocol_digest"),
        "source_tree_digest": sealed.get("source_tree_digest"),
        "freeze_commit_sha": sealed.get("freeze_commit_sha"),
        "freeze_commit_timestamp_utc": sealed.get("freeze_commit_timestamp_utc"),
        "checkpoint_seal_created_at_utc": sealed.get(
            "checkpoint_seal_created_at_utc"
        ),
        "confirmatory_n": sealed.get("confirmatory_n"),
        "reserved_replicate_ids": deepcopy(
            sealed.get("reserved_replicate_ids") or []
        ),
        "stratum_schedule": deepcopy(sealed.get("stratum_schedule") or []),
        "route_threshold": sealed.get("route_threshold"),
        "checkpoint_scientific_identity_digest": checkpoint.get(
            "scientific_identity_digest"
        ),
        "checkpoint_receipt_digest": checkpoint.get("receipt_digest"),
        "checkpoint_replay_contract_digest": checkpoint.get(
            "replay_contract_digest"
        ),
        "checkpoint_final_state_digest": checkpoint.get("final_state_digest"),
        "checkpoint_file_sha256": checkpoint.get("checkpoint_file_sha256"),
        "accounted_flops_contract": accounted_flops,
        "model_geometry": deepcopy(sealed.get("model_geometry") or {}),
        "information_separation": deepcopy(
            sealed.get("information_separation") or {}
        ),
        "reconstruction_code_digest": reconstruction_code_digest,
        "binding_digest": "",
        "reconstruction_digest": "",
    }
    payload["binding_digest"] = _binding_digest(payload)
    payload["reconstruction_digest"] = _reconstruction_digest(payload)
    errors = validate_exp279_reconstruction_authorization(payload)
    if errors:
        raise RuntimeError(
            "invalid EXP-279 reconstruction authorization: " + "; ".join(errors)
        )
    return payload


def validate_exp279_reconstruction_authorization(
    payload: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-279 reconstruction authorization identity")
    if payload.get("evidence_level") != "EV-E2" or payload.get("decision") != "UNVERIFIED":
        errors.append("EXP-279 reconstruction authorization cannot promote evidence")
    if payload.get("status") != STATUS:
        errors.append("EXP-279 reconstruction authorization status drift")
    for flag in (
        "confirmatory_ready",
        "confirmatory_data_consumed",
        "challenge_materialized",
        "decision_rule_executed",
    ):
        if payload.get(flag) is not False:
            errors.append(
                f"EXP-279 reconstruction authorization forbidden flag enabled: {flag}"
            )
    if payload.get("seed_materialization_status") != "NOT_EXECUTED":
        errors.append("EXP-279 reconstruction authorization cannot materialize seeds")
    errors.extend(_prefreeze_errors(payload))

    seal = payload.get("seal_snapshot")
    if not isinstance(seal, dict):
        errors.append("EXP-279 reconstruction seal snapshot missing")
        return errors
    seal_errors = validate_exp279_gate_a_seal(seal)
    if seal_errors:
        errors.append(
            "invalid EXP-279 reconstruction seal snapshot: "
            + "; ".join(seal_errors)
        )
        return errors
    if payload.get("seal_digest") != seal.get("seal_digest"):
        errors.append("EXP-279 reconstruction seal digest mismatch")
    if payload.get("seal_snapshot_digest") != canonical_sha256(seal):
        errors.append("EXP-279 reconstruction seal snapshot digest mismatch")

    sealed = seal.get("authorization_snapshot") or {}
    checkpoint = sealed.get("checkpoint") or {}
    resource = sealed.get("resource_court") or {}
    expected = {
        "protocol_digest": sealed.get("protocol_digest"),
        "source_tree_digest": sealed.get("source_tree_digest"),
        "freeze_commit_sha": sealed.get("freeze_commit_sha"),
        "freeze_commit_timestamp_utc": sealed.get("freeze_commit_timestamp_utc"),
        "checkpoint_seal_created_at_utc": sealed.get(
            "checkpoint_seal_created_at_utc"
        ),
        "confirmatory_n": sealed.get("confirmatory_n"),
        "reserved_replicate_ids": sealed.get("reserved_replicate_ids"),
        "stratum_schedule": sealed.get("stratum_schedule"),
        "route_threshold": sealed.get("route_threshold"),
        "checkpoint_scientific_identity_digest": checkpoint.get(
            "scientific_identity_digest"
        ),
        "checkpoint_receipt_digest": checkpoint.get("receipt_digest"),
        "checkpoint_replay_contract_digest": checkpoint.get(
            "replay_contract_digest"
        ),
        "checkpoint_final_state_digest": checkpoint.get("final_state_digest"),
        "checkpoint_file_sha256": checkpoint.get("checkpoint_file_sha256"),
        "accounted_flops_contract": resource.get("accounted_flops_contract"),
        "model_geometry": sealed.get("model_geometry"),
        "information_separation": sealed.get("information_separation"),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            label = "FLOP cost" if field == "accounted_flops_contract" else field
            errors.append(
                f"EXP-279 reconstruction Gate A lineage mismatch: {label}"
            )

    if payload.get("stratum_schedule") != list(STRATA):
        errors.append("EXP-279 reconstruction stratum schedule drift")
    try:
        route_threshold = float(payload.get("route_threshold"))
        sealed_threshold = float(sealed.get("route_threshold"))
    except (TypeError, ValueError):
        route_threshold = -1.0
        sealed_threshold = -2.0
    if (
        not 0.0 <= route_threshold <= 1.0
        or abs(route_threshold - sealed_threshold) > 1e-12
    ):
        errors.append("EXP-279 reconstruction route threshold drift")

    n = payload.get("confirmatory_n")
    reserved = payload.get("reserved_replicate_ids") or []
    if not isinstance(n, int) or isinstance(n, bool) or not 32 <= n <= 128:
        errors.append("EXP-279 reconstruction confirmatory n invalid")
    elif len(reserved) != n or len(set(reserved)) != n:
        errors.append("EXP-279 reconstruction reserved replicate mismatch")
    elif reserved != list(range(reserved[0], reserved[0] + len(reserved))):
        errors.append("EXP-279 reconstruction reserved replicates are not contiguous")

    errors.extend(
        _validate_accounted_flops_contract(
            payload.get("accounted_flops_contract"),
            ceiling=resource.get("declared_max_accounted_flops_per_episode"),
        )
    )

    machinery = sealed.get("machinery_digests") or {}
    if payload.get("reconstruction_code_digest") != machinery.get(
        "reconstruction_code_digest"
    ):
        errors.append("EXP-279 reconstruction machinery code digest mismatch")
    if not _valid_digest(payload.get("reconstruction_code_digest")):
        errors.append("EXP-279 reconstruction code digest invalid")

    for field in (
        "checkpoint_scientific_identity_digest",
        "checkpoint_receipt_digest",
        "checkpoint_replay_contract_digest",
        "checkpoint_final_state_digest",
        "checkpoint_file_sha256",
    ):
        if not _valid_digest(payload.get(field)):
            errors.append(f"EXP-279 reconstruction checkpoint binding invalid: {field}")

    if payload.get("binding_digest") != _binding_digest(payload):
        errors.append("EXP-279 reconstruction semantic binding digest mismatch")
    if payload.get("reconstruction_digest") != _reconstruction_digest(payload):
        errors.append("EXP-279 reconstruction authorization digest mismatch")
    return errors


def _row_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("row_digest", None)
    return canonical_sha256(clean)


def _arm_input_receipt() -> dict[str, bool]:
    return {
        "same_surface_events": True,
        "same_variable_states": True,
        "propagation_only_received_incidence": True,
        "branch_only_received_incidence": False,
        "hybrid_received_incidence": True,
        "evaluator_targets_withheld_from_arms": True,
    }


def _analytical_cost_receipt(accounted_flops: float) -> dict[str, Any]:
    return {
        "accounting_semantics": (
            "analytical scalar arithmetic FLOPs for frozen neural geometry; "
            "not hardware-profiler FLOPs"
        ),
        "accounted_flops_per_episode": float(accounted_flops),
        "hardware_profiler_flops_claimed": False,
    }


def _arm_metrics(output: Any, targets: Any, *, accounted_flops: float) -> dict[str, Any]:
    from .exp279_paired_runner import PRIMARY_METRIC

    flops = float(accounted_flops)
    if flops <= 0.0:
        raise ValueError("EXP-279 reconstruction accounted FLOPs must be positive")
    predictions = output.decision_logits.argmax(dim=-1)
    exact_per_episode = (predictions == targets).all(dim=-1)
    solution_rate = float(exact_per_episode.to(dtype=output.decision_logits.dtype).mean().item())
    accuracy = float((predictions == targets).to(dtype=output.decision_logits.dtype).mean().item())
    return {
        "predictions": predictions.detach().cpu().tolist(),
        "verified_solution_rate": solution_rate,
        "verified_decision_accuracy": accuracy,
        "mean_verifier_confidence": float(output.verifier_confidence.mean().item()),
        "accounted_flops_per_episode": flops,
        PRIMARY_METRIC: solution_rate / flops,
        "analytical_cost_receipt": _analytical_cost_receipt(flops),
    }


def _validate_checkpoint_against_seal(
    *,
    reconstruction_authorization: dict[str, Any],
    seal: dict[str, Any],
    checkpoint_path: Any,
    checkpoint_receipt: dict[str, Any],
) -> None:
    from pathlib import Path

    from nolane_ai.protocol.identity import file_sha256
    from .exp279_checkpoint import validate_exp279_checkpoint_receipt

    receipt_errors = validate_exp279_checkpoint_receipt(checkpoint_receipt)
    if receipt_errors:
        raise ValueError(
            "invalid EXP-279 checkpoint receipt: " + "; ".join(receipt_errors)
        )
    sealed = _sealed_authorization(seal)
    sealed_checkpoint = sealed.get("checkpoint") or {}
    fields = (
        "receipt_digest",
        "scientific_identity_digest",
        "replay_contract_digest",
        "final_state_digest",
        "final_state",
        "checkpoint_file_sha256",
        "state_policy",
    )
    for field in fields:
        if checkpoint_receipt.get(field) != sealed_checkpoint.get(field):
            raise ValueError(f"EXP-279 checkpoint Gate A binding mismatch: {field}")

    reconstruction_fields = {
        "scientific_identity_digest": "checkpoint_scientific_identity_digest",
        "receipt_digest": "checkpoint_receipt_digest",
        "replay_contract_digest": "checkpoint_replay_contract_digest",
        "final_state_digest": "checkpoint_final_state_digest",
        "checkpoint_file_sha256": "checkpoint_file_sha256",
    }
    for receipt_field, reconstruction_field in reconstruction_fields.items():
        if checkpoint_receipt.get(receipt_field) != reconstruction_authorization.get(
            reconstruction_field
        ):
            raise ValueError(
                f"EXP-279 checkpoint reconstruction binding mismatch: {receipt_field}"
            )

    path = Path(checkpoint_path)
    if not path.is_file():
        raise ValueError("EXP-279 checkpoint file is missing")
    if file_sha256(path) != checkpoint_receipt.get("checkpoint_file_sha256"):
        raise ValueError("EXP-279 checkpoint file SHA256 mismatch")


def _replicate_ordinal_and_stratum(
    reconstruction_authorization: dict[str, Any],
    replicate: int,
) -> tuple[int, str]:
    reserved = list(reconstruction_authorization.get("reserved_replicate_ids") or [])
    if (
        not isinstance(replicate, int)
        or isinstance(replicate, bool)
        or replicate not in reserved
    ):
        raise ValueError(
            "EXP-279 reconstruction replicate is outside frozen reserved lineage"
        )
    schedule = list(reconstruction_authorization.get("stratum_schedule") or [])
    if schedule != list(STRATA):
        raise ValueError("EXP-279 reconstruction stratum schedule drift")
    ordinal = reserved.index(replicate)
    return ordinal, schedule[ordinal % len(schedule)]


def reconstruct_exp279_expected_row(
    *,
    reconstruction_authorization: dict[str, Any],
    seal: dict[str, Any],
    beacon_receipt: dict[str, Any],
    checkpoint_path: Any,
    checkpoint_receipt: dict[str, Any],
    replicate: int,
) -> dict[str, Any]:
    import torch

    from .exp279_beacon import (
        CHALLENGE_STREAM,
        derive_exp279_challenge_seed,
        validate_exp279_beacon_receipt,
    )
    from .exp279_challenge_worlds import build_exp279_challenge_batch
    from .exp279_checkpoint import load_exp279_trained_checkpoint
    from .exp279_paired_runner import PRIMARY_METRIC, _functional_state_digest

    reconstruction_errors = validate_exp279_reconstruction_authorization(
        reconstruction_authorization
    )
    if reconstruction_errors:
        raise ValueError(
            "invalid EXP-279 reconstruction authorization: "
            + "; ".join(reconstruction_errors)
        )
    seal_errors = validate_exp279_gate_a_seal(seal)
    if seal_errors:
        raise ValueError("invalid EXP-279 Gate A seal: " + "; ".join(seal_errors))
    if reconstruction_authorization.get("seal_digest") != seal.get("seal_digest"):
        raise ValueError("EXP-279 reconstruction/seal binding mismatch")
    if reconstruction_authorization.get("seal_snapshot_digest") != canonical_sha256(seal):
        raise ValueError("EXP-279 reconstruction seal snapshot mismatch")

    ordinal, stratum = _replicate_ordinal_and_stratum(
        reconstruction_authorization,
        replicate,
    )

    beacon_errors = validate_exp279_beacon_receipt(
        beacon_receipt,
        freeze_commit_timestamp_utc=reconstruction_authorization.get(
            "freeze_commit_timestamp_utc"
        ),
        checkpoint_seal_created_at_utc=reconstruction_authorization.get(
            "checkpoint_seal_created_at_utc"
        ),
    )
    if beacon_errors:
        raise ValueError("invalid EXP-279 beacon receipt: " + "; ".join(beacon_errors))

    _validate_checkpoint_against_seal(
        reconstruction_authorization=reconstruction_authorization,
        seal=seal,
        checkpoint_path=checkpoint_path,
        checkpoint_receipt=checkpoint_receipt,
    )
    propagation, branch, hybrid = load_exp279_trained_checkpoint(
        checkpoint_path=checkpoint_path,
        receipt=checkpoint_receipt,
    )
    propagation.eval()
    branch.eval()
    hybrid.eval()
    before = {
        "propagation_only": _functional_state_digest(propagation),
        "branch_only": _functional_state_digest(branch),
        "hybrid": _functional_state_digest(hybrid),
    }

    challenge_seed = derive_exp279_challenge_seed(
        protocol_digest=str(reconstruction_authorization.get("protocol_digest") or ""),
        beacon_receipt=beacon_receipt,
        stream=CHALLENGE_STREAM,
        replicate=replicate,
        stratum=stratum,
        freeze_commit_timestamp_utc=reconstruction_authorization.get(
            "freeze_commit_timestamp_utc"
        ),
        checkpoint_seal_created_at_utc=reconstruction_authorization.get(
            "checkpoint_seal_created_at_utc"
        ),
    )
    world = (reconstruction_authorization.get("model_geometry") or {}).get(
        "world"
    ) or {}
    batch = build_exp279_challenge_batch(
        challenge_seed=challenge_seed,
        replicate=replicate,
        stratum=stratum,
        batch_size=int(world.get("batch_size", 0)),
        timesteps=int(world.get("timesteps", 0)),
        variables=int(world.get("variables", 0)),
        constraints=int(world.get("constraints", 0)),
        d_model=int(world.get("d_model", 0)),
        noise_std=float(world.get("noise_std", -1.0)),
        device="cpu",
    )

    with torch.no_grad():
        propagation_output = propagation(
            batch.surface_events,
            batch.variable_states,
            batch.incidence,
        )
        branch_output = branch(batch.surface_events, batch.variable_states)
        hybrid_output = hybrid(
            batch.surface_events,
            batch.variable_states,
            batch.incidence,
        )

    after = {
        "propagation_only": _functional_state_digest(propagation),
        "branch_only": _functional_state_digest(branch),
        "hybrid": _functional_state_digest(hybrid),
    }
    if before != after:
        raise RuntimeError("EXP-279 reconstruction detected a forbidden parameter write")

    costs = reconstruction_authorization.get("accounted_flops_contract") or {}
    prop_cost = float(
        (costs.get("propagation_only") or {}).get("accounted_flops_per_episode", 0)
    )
    branch_cost = float(
        (costs.get("branch_only") or {}).get("accounted_flops_per_episode", 0)
    )
    hybrid_contract = costs.get("hybrid") or {}
    stop_cost = float(hybrid_contract.get("stop_accounted_flops_per_episode", 0))
    full_branch_cost = float(
        hybrid_contract.get("branch_accounted_flops_per_episode", 0)
    )
    route_mask = [
        bool(item)
        for item in hybrid_output.branch_route_mask.detach().cpu().tolist()
    ]
    batch_size = len(route_mask)
    if batch_size <= 0:
        raise RuntimeError("EXP-279 reconstruction hybrid route mask is empty")
    routed = sum(route_mask)
    route_fraction = routed / batch_size
    hybrid_cost = stop_cost + route_fraction * (full_branch_cost - stop_cost)
    route_receipt = {
        "strategy": "propagation_then_branch_on_residual_uncertainty",
        "threshold": float(reconstruction_authorization.get("route_threshold")),
        "batch_size": batch_size,
        "branch_route_mask": route_mask,
        "routed_episodes": routed,
        "route_fraction": route_fraction,
        "mean_residual_uncertainty": float(
            hybrid_output.residual_uncertainty.mean().item()
        ),
        "stop_accounted_flops_per_episode": stop_cost,
        "branch_accounted_flops_per_episode": full_branch_cost,
        "charged_accounted_flops_per_episode": hybrid_cost,
    }

    propagation_metrics = _arm_metrics(
        propagation_output,
        batch.targets,
        accounted_flops=prop_cost,
    )
    branch_metrics = _arm_metrics(
        branch_output,
        batch.targets,
        accounted_flops=branch_cost,
    )
    hybrid_metrics = _arm_metrics(
        hybrid_output,
        batch.targets,
        accounted_flops=hybrid_cost,
    )

    hybrid_utility = float(hybrid_metrics[PRIMARY_METRIC])
    prop_utility = float(propagation_metrics[PRIMARY_METRIC])
    branch_utility = float(branch_metrics[PRIMARY_METRIC])
    paired = {
        "world_pairing_closed": True,
        "hybrid_minus_propagation_solution_rate": float(
            hybrid_metrics["verified_solution_rate"]
        )
        - float(propagation_metrics["verified_solution_rate"]),
        "hybrid_minus_branch_solution_rate": float(
            hybrid_metrics["verified_solution_rate"]
        )
        - float(branch_metrics["verified_solution_rate"]),
        "hybrid_minus_propagation_utility": hybrid_utility - prop_utility,
        "hybrid_minus_branch_utility": hybrid_utility - branch_utility,
        "hybrid_vs_propagation_relative_gain": (
            None if prop_utility <= 0.0 else hybrid_utility / prop_utility - 1.0
        ),
        "hybrid_vs_branch_relative_gain": (
            None if branch_utility <= 0.0 else hybrid_utility / branch_utility - 1.0
        ),
        "simple_denominators_positive": {
            "propagation_only": prop_utility > 0.0,
            "branch_only": branch_utility > 0.0,
        },
    }

    row: dict[str, Any] = {
        "schema": "NLM-EXP-279-CONFIRMATORY-RAW-ROW-V1",
        "experiment_id": EXPERIMENT_ID,
        "replicate": replicate,
        "replicate_ordinal": ordinal,
        "stratum": stratum,
        "challenge_seed": challenge_seed,
        "challenge_digest": batch.digest,
        "checkpoint_scientific_identity_digest": checkpoint_receipt.get(
            "scientific_identity_digest"
        ),
        "checkpoint_functional_state_before": before,
        "checkpoint_functional_state_after": after,
        "information_separation": deepcopy(
            reconstruction_authorization.get("information_separation") or {}
        ),
        "arm_input_receipt": _arm_input_receipt(),
        "hybrid_route_receipt": route_receipt,
        "propagation_only": propagation_metrics,
        "branch_only": branch_metrics,
        "hybrid": hybrid_metrics,
        "paired": paired,
        "row_digest": "",
    }
    row["row_digest"] = _row_digest(row)
    return row


def _float_equal(left: Any, right: Any) -> bool:
    import math

    if (
        not isinstance(left, (int, float))
        or isinstance(left, bool)
        or not isinstance(right, (int, float))
        or isinstance(right, bool)
    ):
        return False
    left_f = float(left)
    right_f = float(right)
    if not math.isfinite(left_f) or not math.isfinite(right_f):
        return left_f == right_f
    return math.isclose(left_f, right_f, rel_tol=0.0, abs_tol=1e-15)


def validate_exp279_raw_row_against_reconstruction(
    *,
    raw_row: dict[str, Any],
    reconstruction_authorization: dict[str, Any],
    seal: dict[str, Any],
    beacon_receipt: dict[str, Any],
    checkpoint_path: Any,
    checkpoint_receipt: dict[str, Any],
) -> list[str]:
    from .exp279_paired_runner import PRIMARY_METRIC

    errors: list[str] = []
    if not isinstance(raw_row, dict):
        return ["EXP-279 raw row must be an object"]
    if raw_row.get("row_digest") != _row_digest(raw_row):
        errors.append("EXP-279 raw row digest mismatch")

    replicate = raw_row.get("replicate")
    try:
        expected_ordinal, expected_stratum = _replicate_ordinal_and_stratum(
            reconstruction_authorization,
            replicate,
        )
    except ValueError as exc:
        errors.append(str(exc))
        return errors
    if raw_row.get("replicate_ordinal") != expected_ordinal:
        errors.append("EXP-279 raw row replicate lineage ordinal mismatch")
    if raw_row.get("stratum") != expected_stratum:
        errors.append("EXP-279 raw row stratum mismatch")

    try:
        expected = reconstruct_exp279_expected_row(
            reconstruction_authorization=reconstruction_authorization,
            seal=seal,
            beacon_receipt=beacon_receipt,
            checkpoint_path=checkpoint_path,
            checkpoint_receipt=checkpoint_receipt,
            replicate=replicate,
        )
    except (ValueError, RuntimeError) as exc:
        errors.append(str(exc))
        return errors

    if raw_row.get("schema") != expected.get("schema"):
        errors.append("EXP-279 raw row schema mismatch")
    if raw_row.get("experiment_id") != EXPERIMENT_ID:
        errors.append("EXP-279 raw row experiment identity mismatch")
    if raw_row.get("challenge_seed") != expected.get("challenge_seed"):
        errors.append("EXP-279 raw row challenge seed mismatch")
    if raw_row.get("challenge_digest") != expected.get("challenge_digest"):
        errors.append("EXP-279 raw row challenge digest mismatch")
    if raw_row.get("checkpoint_scientific_identity_digest") != expected.get(
        "checkpoint_scientific_identity_digest"
    ):
        errors.append("EXP-279 raw row checkpoint identity mismatch")
    if (
        raw_row.get("checkpoint_functional_state_before")
        != expected.get("checkpoint_functional_state_before")
        or raw_row.get("checkpoint_functional_state_after")
        != expected.get("checkpoint_functional_state_after")
    ):
        errors.append("EXP-279 raw row checkpoint functional-state mismatch")
    if raw_row.get("information_separation") != expected.get("information_separation"):
        errors.append("EXP-279 raw row information-separation mismatch")
    if raw_row.get("arm_input_receipt") != expected.get("arm_input_receipt"):
        errors.append("EXP-279 raw row arm-input information receipt mismatch")

    for arm_id in ("propagation_only", "branch_only", "hybrid"):
        actual = raw_row.get(arm_id) or {}
        wanted = expected.get(arm_id) or {}
        if actual.get("predictions") != wanted.get("predictions"):
            errors.append(f"EXP-279 {arm_id} prediction mismatch")
        for field in (
            "verified_solution_rate",
            "verified_decision_accuracy",
            "mean_verifier_confidence",
        ):
            if not _float_equal(actual.get(field), wanted.get(field)):
                label = "solution" if field == "verified_solution_rate" else "metric"
                errors.append(f"EXP-279 {arm_id} {label} mismatch: {field}")
        if not _float_equal(
            actual.get("accounted_flops_per_episode"),
            wanted.get("accounted_flops_per_episode"),
        ):
            errors.append(f"EXP-279 {arm_id} accounted FLOP mismatch")
        if not _float_equal(actual.get(PRIMARY_METRIC), wanted.get(PRIMARY_METRIC)):
            errors.append(f"EXP-279 {arm_id} utility mismatch")
        if actual.get("analytical_cost_receipt") != wanted.get(
            "analytical_cost_receipt"
        ):
            errors.append(f"EXP-279 {arm_id} analytical FLOP receipt mismatch")

    actual_route = raw_row.get("hybrid_route_receipt") or {}
    expected_route = expected.get("hybrid_route_receipt") or {}
    for field in (
        "strategy",
        "batch_size",
        "branch_route_mask",
        "routed_episodes",
    ):
        if actual_route.get(field) != expected_route.get(field):
            errors.append(f"EXP-279 hybrid route receipt mismatch: {field}")
    for field in (
        "threshold",
        "route_fraction",
        "mean_residual_uncertainty",
        "stop_accounted_flops_per_episode",
        "branch_accounted_flops_per_episode",
        "charged_accounted_flops_per_episode",
    ):
        if not _float_equal(actual_route.get(field), expected_route.get(field)):
            errors.append(f"EXP-279 hybrid route FLOP/metric mismatch: {field}")

    actual_paired = raw_row.get("paired") or {}
    expected_paired = expected.get("paired") or {}
    if actual_paired.get("world_pairing_closed") is not True:
        errors.append("EXP-279 raw paired world lineage is open")
    for field in (
        "hybrid_minus_propagation_solution_rate",
        "hybrid_minus_branch_solution_rate",
        "hybrid_minus_propagation_utility",
        "hybrid_minus_branch_utility",
    ):
        if not _float_equal(actual_paired.get(field), expected_paired.get(field)):
            errors.append(f"EXP-279 raw paired metric mismatch: {field}")
    for field in (
        "hybrid_vs_propagation_relative_gain",
        "hybrid_vs_branch_relative_gain",
    ):
        actual_value = actual_paired.get(field)
        expected_value = expected_paired.get(field)
        if actual_value is None or expected_value is None:
            if actual_value != expected_value:
                errors.append(f"EXP-279 raw paired relative gain mismatch: {field}")
        elif not _float_equal(actual_value, expected_value):
            errors.append(f"EXP-279 raw paired relative gain mismatch: {field}")
    if actual_paired.get("simple_denominators_positive") != expected_paired.get(
        "simple_denominators_positive"
    ):
        errors.append("EXP-279 raw paired denominator status mismatch")
    return errors
