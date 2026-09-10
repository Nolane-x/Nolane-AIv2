from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any

import torch

from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.protocol.identity import file_sha256
from .exp279_beacon import (
    CHALLENGE_STREAM,
    derive_exp279_challenge_seed,
    validate_exp279_beacon_receipt,
)
from .exp279_challenge_worlds import build_exp279_challenge_batch
from .exp279_checkpoint import (
    load_exp279_trained_checkpoint,
    validate_exp279_checkpoint_receipt,
)
from .exp279_confirmatory_authorization import validate_exp279_gate_a_seal
from .exp279_paired_runner import PRIMARY_METRIC, _functional_state_digest
from .exp279_reconstruction_court import (
    _row_digest,
    validate_exp279_raw_row_against_reconstruction,
    validate_exp279_reconstruction_authorization,
)
from .exp279_routing_worlds import STRATA


SCHEMA = "NLM-EXP-279-CONFIRMATORY-CHALLENGE-RAW-V1"
EXPERIMENT_ID = "EXP-279"
SCIENTIFIC_STATUS = "CONFIRMATORY_CHALLENGE_EXECUTED_UNANALYZED"
TEST_ONLY_STATUS = "TEST_ONLY_CHALLENGE_EXECUTED_UNANALYZED"
INVALID_STATUS = "INVALID_RUN"
COST_SEMANTICS = (
    "analytical scalar arithmetic FLOPs for frozen neural geometry; "
    "not hardware-profiler FLOPs"
)


def _artifact_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("artifact_digest", None)
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
        "accounting_semantics": COST_SEMANTICS,
        "accounted_flops_per_episode": float(accounted_flops),
        "hardware_profiler_flops_claimed": False,
    }


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _invalid_artifact(
    *,
    reconstruction_authorization: dict[str, Any],
    seal: dict[str, Any],
    beacon_receipt: dict[str, Any],
    current_source_tree_digest: str,
    executor_code_digest: str,
    integrity_errors: list[str],
    challenge_materialized: bool = False,
) -> dict[str, Any]:
    test_only = beacon_receipt.get("test_only") is True
    scientific = bool(
        beacon_receipt.get("scientific_evidence_eligible") is True and not test_only
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "evidence_level": "EV-E2",
        "decision": INVALID_STATUS,
        "status": INVALID_STATUS,
        "test_only": test_only,
        "scientific_evidence_eligible": False,
        "confirmatory_data_consumed": bool(challenge_materialized and scientific),
        "synthetic_challenge_data_consumed": bool(challenge_materialized and test_only),
        "challenge_materialized": bool(challenge_materialized),
        "seed_materialization_status": (
            "EXECUTION_FAILED_AFTER_MATERIALIZATION"
            if challenge_materialized
            else "NOT_EXECUTED"
        ),
        "decision_rule_executed": False,
        "confirmatory_n": reconstruction_authorization.get("confirmatory_n"),
        "reserved_replicate_ids": deepcopy(
            reconstruction_authorization.get("reserved_replicate_ids") or []
        ),
        "challenge_stream": CHALLENGE_STREAM,
        "per_replicate": [],
        "integrity_errors": list(integrity_errors),
        "current_source_tree_digest": current_source_tree_digest,
        "executor_code_digest": executor_code_digest,
        "seal_digest": seal.get("seal_digest"),
        "reconstruction_digest": reconstruction_authorization.get(
            "reconstruction_digest"
        ),
        "beacon_receipt_digest": beacon_receipt.get("receipt_digest"),
        "artifact_digest": "",
    }
    payload["artifact_digest"] = _artifact_digest(payload)
    return payload


def _preflight_integrity_errors(
    *,
    reconstruction_authorization: dict[str, Any],
    seal: dict[str, Any],
    beacon_receipt: dict[str, Any],
    checkpoint_path: str | Path,
    checkpoint_receipt: dict[str, Any],
    current_source_tree_digest: str,
    executor_code_digest: str,
) -> list[str]:
    errors: list[str] = []

    reconstruction_errors = validate_exp279_reconstruction_authorization(
        reconstruction_authorization
    )
    if reconstruction_errors:
        errors.append(
            "EXP-279 reconstruction authorization invalid: "
            + "; ".join(reconstruction_errors)
        )

    seal_errors = validate_exp279_gate_a_seal(seal)
    if seal_errors:
        errors.append("EXP-279 Gate A seal invalid: " + "; ".join(seal_errors))

    if reconstruction_authorization.get("seal_digest") != seal.get("seal_digest"):
        errors.append("EXP-279 reconstruction/seal lineage mismatch")
    if reconstruction_authorization.get("seal_snapshot_digest") != canonical_sha256(
        seal
    ):
        errors.append("EXP-279 reconstruction seal snapshot mismatch")

    frozen_source = reconstruction_authorization.get("source_tree_digest")
    if current_source_tree_digest != frozen_source:
        errors.append(
            "EXP-279 current source tree digest does not match frozen source tree"
        )

    sealed = seal.get("authorization_snapshot") or {}
    machinery = sealed.get("machinery_digests") or {}
    if executor_code_digest != machinery.get("executor_code_digest"):
        errors.append(
            "EXP-279 executor code digest does not match frozen executor machinery"
        )

    checkpoint_errors = validate_exp279_checkpoint_receipt(checkpoint_receipt)
    if checkpoint_errors:
        errors.append(
            "EXP-279 checkpoint receipt invalid: " + "; ".join(checkpoint_errors)
        )

    sealed_checkpoint = sealed.get("checkpoint") or {}
    for field in (
        "receipt_digest",
        "scientific_identity_digest",
        "replay_contract_digest",
        "final_state_digest",
        "final_state",
        "checkpoint_file_sha256",
        "state_policy",
    ):
        if checkpoint_receipt.get(field) != sealed_checkpoint.get(field):
            errors.append(f"EXP-279 checkpoint Gate A binding mismatch: {field}")

    reconstruction_checkpoint_fields = {
        "scientific_identity_digest": "checkpoint_scientific_identity_digest",
        "receipt_digest": "checkpoint_receipt_digest",
        "replay_contract_digest": "checkpoint_replay_contract_digest",
        "final_state_digest": "checkpoint_final_state_digest",
        "checkpoint_file_sha256": "checkpoint_file_sha256",
    }
    for receipt_field, reconstruction_field in reconstruction_checkpoint_fields.items():
        if checkpoint_receipt.get(receipt_field) != reconstruction_authorization.get(
            reconstruction_field
        ):
            errors.append(
                "EXP-279 checkpoint reconstruction binding mismatch: "
                f"{receipt_field}"
            )

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        errors.append("EXP-279 checkpoint file is missing")
    elif checkpoint_receipt.get("checkpoint_file_sha256") != file_sha256(
        checkpoint_path
    ):
        errors.append("EXP-279 checkpoint file SHA256 differs from frozen checkpoint")

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
        errors.append("EXP-279 beacon receipt invalid: " + "; ".join(beacon_errors))

    reserved = reconstruction_authorization.get("reserved_replicate_ids") or []
    confirmatory_n = reconstruction_authorization.get("confirmatory_n")
    if (
        not isinstance(confirmatory_n, int)
        or isinstance(confirmatory_n, bool)
        or not 32 <= confirmatory_n <= 128
        or not isinstance(reserved, list)
        or len(reserved) != confirmatory_n
        or len(set(reserved)) != confirmatory_n
        or (
            reserved
            and reserved != list(range(reserved[0], reserved[0] + len(reserved)))
        )
    ):
        errors.append("EXP-279 frozen confirmatory replicate lineage invalid")

    schedule = reconstruction_authorization.get("stratum_schedule")
    if schedule != list(STRATA):
        errors.append("EXP-279 frozen confirmatory stratum schedule invalid")

    costs = reconstruction_authorization.get("accounted_flops_contract")
    if not isinstance(costs, dict) or set(costs) != {
        "propagation_only",
        "branch_only",
        "hybrid",
    }:
        errors.append("EXP-279 analytical FLOP contract missing")
    else:
        for arm_id in ("propagation_only", "branch_only"):
            item = costs.get(arm_id) or {}
            value = item.get("accounted_flops_per_episode")
            if (
                item.get("mode") != "fixed"
                or not _finite_number(value)
                or float(value) <= 0.0
            ):
                errors.append(f"EXP-279 {arm_id} analytical FLOP contract invalid")
        hybrid = costs.get("hybrid") or {}
        stop = hybrid.get("stop_accounted_flops_per_episode")
        branch = hybrid.get("branch_accounted_flops_per_episode")
        if (
            hybrid.get("mode") != "route_dependent"
            or not _finite_number(stop)
            or not _finite_number(branch)
            or float(stop) <= 0.0
            or float(branch) < float(stop)
        ):
            errors.append("EXP-279 hybrid route-dependent FLOP contract invalid")

    return errors


def _arm_metrics(
    output: Any,
    targets: torch.Tensor,
    *,
    accounted_flops: float,
) -> dict[str, Any]:
    flops = float(accounted_flops)
    if not math.isfinite(flops) or flops <= 0.0:
        raise ValueError("EXP-279 executor accounted FLOPs must be finite and positive")
    predictions = output.decision_logits.argmax(dim=-1)
    exact_per_episode = (predictions == targets).all(dim=-1)
    solution_rate = float(exact_per_episode.to(torch.float32).mean().item())
    accuracy = float((predictions == targets).to(torch.float32).mean().item())
    return {
        "predictions": predictions.detach().cpu().tolist(),
        "verified_solution_rate": solution_rate,
        "verified_decision_accuracy": accuracy,
        "mean_verifier_confidence": float(output.verifier_confidence.mean().item()),
        "accounted_flops_per_episode": flops,
        PRIMARY_METRIC: solution_rate / flops,
        "analytical_cost_receipt": _analytical_cost_receipt(flops),
    }


def _execute_rows_independently(
    *,
    reconstruction_authorization: dict[str, Any],
    beacon_receipt: dict[str, Any],
    checkpoint_path: str | Path,
    checkpoint_receipt: dict[str, Any],
) -> list[dict[str, Any]]:
    propagation, branch, hybrid = load_exp279_trained_checkpoint(
        checkpoint_path=checkpoint_path,
        receipt=checkpoint_receipt,
    )
    propagation.eval()
    branch.eval()
    hybrid.eval()

    reserved = list(reconstruction_authorization["reserved_replicate_ids"])
    schedule = list(reconstruction_authorization["stratum_schedule"])
    world = (reconstruction_authorization.get("model_geometry") or {}).get(
        "world"
    ) or {}
    costs = reconstruction_authorization["accounted_flops_contract"]
    prop_cost = float(costs["propagation_only"]["accounted_flops_per_episode"])
    branch_cost = float(costs["branch_only"]["accounted_flops_per_episode"])
    hybrid_contract = costs["hybrid"]
    stop_cost = float(hybrid_contract["stop_accounted_flops_per_episode"])
    full_branch_cost = float(
        hybrid_contract["branch_accounted_flops_per_episode"]
    )

    rows: list[dict[str, Any]] = []
    for ordinal, replicate in enumerate(reserved):
        stratum = schedule[ordinal % len(schedule)]
        before = {
            "propagation_only": _functional_state_digest(propagation),
            "branch_only": _functional_state_digest(branch),
            "hybrid": _functional_state_digest(hybrid),
        }
        challenge_seed = derive_exp279_challenge_seed(
            protocol_digest=str(
                reconstruction_authorization.get("protocol_digest") or ""
            ),
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
            raise RuntimeError(
                "EXP-279 executor detected forbidden parameter drift during "
                "confirmatory inference"
            )

        route_mask = [
            bool(item)
            for item in hybrid_output.branch_route_mask.detach().cpu().tolist()
        ]
        batch_size = len(route_mask)
        if batch_size <= 0:
            raise RuntimeError("EXP-279 executor hybrid route mask is empty")
        routed = sum(route_mask)
        route_fraction = routed / batch_size
        hybrid_cost = stop_cost + route_fraction * (full_branch_cost - stop_cost)
        route_receipt = {
            "strategy": "propagation_then_branch_on_residual_uncertainty",
            "threshold": float(reconstruction_authorization["route_threshold"]),
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
                None
                if prop_utility <= 0.0
                else hybrid_utility / prop_utility - 1.0
            ),
            "hybrid_vs_branch_relative_gain": (
                None
                if branch_utility <= 0.0
                else hybrid_utility / branch_utility - 1.0
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
        rows.append(row)

    return rows


def execute_exp279_confirmatory_challenge(
    *,
    reconstruction_authorization: dict[str, Any],
    seal: dict[str, Any],
    beacon_receipt: dict[str, Any],
    checkpoint_path: str | Path,
    checkpoint_receipt: dict[str, Any],
    current_source_tree_digest: str,
    executor_code_digest: str,
) -> dict[str, Any]:
    errors = _preflight_integrity_errors(
        reconstruction_authorization=reconstruction_authorization,
        seal=seal,
        beacon_receipt=beacon_receipt,
        checkpoint_path=checkpoint_path,
        checkpoint_receipt=checkpoint_receipt,
        current_source_tree_digest=current_source_tree_digest,
        executor_code_digest=executor_code_digest,
    )
    if errors:
        return _invalid_artifact(
            reconstruction_authorization=reconstruction_authorization,
            seal=seal,
            beacon_receipt=beacon_receipt,
            current_source_tree_digest=current_source_tree_digest,
            executor_code_digest=executor_code_digest,
            integrity_errors=errors,
        )

    try:
        rows = _execute_rows_independently(
            reconstruction_authorization=reconstruction_authorization,
            beacon_receipt=beacon_receipt,
            checkpoint_path=checkpoint_path,
            checkpoint_receipt=checkpoint_receipt,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        return _invalid_artifact(
            reconstruction_authorization=reconstruction_authorization,
            seal=seal,
            beacon_receipt=beacon_receipt,
            current_source_tree_digest=current_source_tree_digest,
            executor_code_digest=executor_code_digest,
            integrity_errors=[
                "EXP-279 confirmatory executor integrity failure after challenge "
                f"materialization: {exc}"
            ],
            challenge_materialized=True,
        )

    test_only = beacon_receipt.get("test_only") is True
    scientific_eligible = bool(
        beacon_receipt.get("scientific_evidence_eligible") is True and not test_only
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "status": TEST_ONLY_STATUS if test_only else SCIENTIFIC_STATUS,
        "test_only": test_only,
        "scientific_evidence_eligible": scientific_eligible,
        "confirmatory_data_consumed": bool(scientific_eligible),
        "synthetic_challenge_data_consumed": bool(test_only),
        "challenge_materialized": True,
        "seed_materialization_status": (
            "TEST_ONLY_EXECUTED" if test_only else "EXECUTED"
        ),
        "decision_rule_executed": False,
        "confirmatory_n": len(rows),
        "reserved_replicate_ids": deepcopy(
            reconstruction_authorization.get("reserved_replicate_ids") or []
        ),
        "challenge_stream": CHALLENGE_STREAM,
        "per_replicate": rows,
        "integrity_errors": [],
        "current_source_tree_digest": current_source_tree_digest,
        "executor_code_digest": executor_code_digest,
        "seal": deepcopy(seal),
        "reconstruction_authorization": deepcopy(reconstruction_authorization),
        "beacon_receipt": deepcopy(beacon_receipt),
        "lineage": {
            "protocol_digest": reconstruction_authorization.get("protocol_digest"),
            "source_tree_digest": reconstruction_authorization.get(
                "source_tree_digest"
            ),
            "seal_digest": seal.get("seal_digest"),
            "reconstruction_digest": reconstruction_authorization.get(
                "reconstruction_digest"
            ),
            "checkpoint_scientific_identity_digest": checkpoint_receipt.get(
                "scientific_identity_digest"
            ),
            "checkpoint_receipt_digest": checkpoint_receipt.get("receipt_digest"),
            "beacon_receipt_digest": beacon_receipt.get("receipt_digest"),
            "executor_code_digest": executor_code_digest,
        },
        "artifact_digest": "",
    }
    payload["artifact_digest"] = _artifact_digest(payload)

    validation_errors = validate_exp279_confirmatory_raw(
        payload,
        checkpoint_path=checkpoint_path,
        checkpoint_receipt=checkpoint_receipt,
    )
    if validation_errors:
        return _invalid_artifact(
            reconstruction_authorization=reconstruction_authorization,
            seal=seal,
            beacon_receipt=beacon_receipt,
            current_source_tree_digest=current_source_tree_digest,
            executor_code_digest=executor_code_digest,
            integrity_errors=validation_errors,
            challenge_materialized=True,
        )
    return payload


def validate_exp279_confirmatory_raw(
    payload: dict[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
    checkpoint_receipt: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["EXP-279 raw artifact must be an object"]
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        errors.append("invalid EXP-279 confirmatory raw identity")
    if payload.get("artifact_digest") != _artifact_digest(payload):
        errors.append("EXP-279 confirmatory raw artifact digest mismatch")

    status = payload.get("status")
    decision = payload.get("decision")
    if status == INVALID_STATUS or decision == INVALID_STATUS:
        if status != INVALID_STATUS or decision != INVALID_STATUS:
            errors.append("EXP-279 invalid raw status/decision mismatch")
        if payload.get("evidence_level") != "EV-E2":
            errors.append("EXP-279 invalid raw evidence level drift")
        if payload.get("scientific_evidence_eligible") is not False:
            errors.append("EXP-279 invalid run cannot be scientific evidence")
        if payload.get("decision_rule_executed") is not False:
            errors.append("EXP-279 invalid run cannot execute decision rule")
        if payload.get("per_replicate") != []:
            errors.append("EXP-279 invalid run cannot publish partial rows")
        if not payload.get("integrity_errors"):
            errors.append("EXP-279 invalid run must record integrity errors")
        materialized = payload.get("challenge_materialized")
        if materialized is False:
            if payload.get("seed_materialization_status") != "NOT_EXECUTED":
                errors.append("EXP-279 preflight invalid seed status drift")
            if payload.get("confirmatory_data_consumed") is not False:
                errors.append("EXP-279 preflight invalid cannot consume confirmatory data")
            if payload.get("synthetic_challenge_data_consumed") is not False:
                errors.append("EXP-279 preflight invalid cannot consume synthetic data")
        elif materialized is True:
            if (
                payload.get("seed_materialization_status")
                != "EXECUTION_FAILED_AFTER_MATERIALIZATION"
            ):
                errors.append("EXP-279 post-materialization invalid seed status drift")
        else:
            errors.append("EXP-279 invalid run materialization status missing")
        return errors

    if payload.get("evidence_level") != "EV-E2" or decision != "UNVERIFIED":
        errors.append("EXP-279 raw executor cannot promote scientific evidence")
    if payload.get("decision_rule_executed") is not False:
        errors.append("EXP-279 raw executor cannot execute frozen decision rule")
    if payload.get("challenge_materialized") is not True:
        errors.append("EXP-279 valid raw executor must materialize challenge")
    if payload.get("challenge_stream") != CHALLENGE_STREAM:
        errors.append("EXP-279 raw challenge stream drift")
    if payload.get("integrity_errors") != []:
        errors.append("EXP-279 valid raw artifact cannot contain integrity errors")

    test_only = payload.get("test_only")
    scientific = payload.get("scientific_evidence_eligible")
    if test_only is True:
        if status != TEST_ONLY_STATUS:
            errors.append("EXP-279 TEST-ONLY raw status drift")
        if scientific is not False:
            errors.append("EXP-279 TEST-ONLY raw cannot be scientific evidence")
        if payload.get("confirmatory_data_consumed") is not False:
            errors.append("EXP-279 TEST-ONLY raw cannot consume confirmatory data")
        if payload.get("synthetic_challenge_data_consumed") is not True:
            errors.append("EXP-279 TEST-ONLY raw synthetic-data flag missing")
        if payload.get("seed_materialization_status") != "TEST_ONLY_EXECUTED":
            errors.append("EXP-279 TEST-ONLY seed materialization status drift")
    elif test_only is False:
        if status != SCIENTIFIC_STATUS:
            errors.append("EXP-279 scientific raw status drift")
        if scientific is not True:
            errors.append("EXP-279 scientific raw must be evidence-eligible")
        if payload.get("confirmatory_data_consumed") is not True:
            errors.append("EXP-279 scientific raw must record confirmatory consumption")
        if payload.get("synthetic_challenge_data_consumed") is not False:
            errors.append("EXP-279 scientific raw cannot mark synthetic consumption")
        if payload.get("seed_materialization_status") != "EXECUTED":
            errors.append("EXP-279 scientific seed materialization status drift")
    else:
        errors.append("EXP-279 raw test_only classification missing")

    reconstruction = payload.get("reconstruction_authorization")
    seal = payload.get("seal")
    beacon = payload.get("beacon_receipt")
    if (
        not isinstance(reconstruction, dict)
        or not isinstance(seal, dict)
        or not isinstance(beacon, dict)
    ):
        errors.append("EXP-279 raw reconstruction/seal/beacon lineage missing")
        return errors

    reconstruction_errors = validate_exp279_reconstruction_authorization(
        reconstruction
    )
    if reconstruction_errors:
        errors.append(
            "EXP-279 raw reconstruction authorization invalid: "
            + "; ".join(reconstruction_errors)
        )
    seal_errors = validate_exp279_gate_a_seal(seal)
    if seal_errors:
        errors.append("EXP-279 raw Gate A seal invalid: " + "; ".join(seal_errors))
    beacon_errors = validate_exp279_beacon_receipt(
        beacon,
        freeze_commit_timestamp_utc=reconstruction.get(
            "freeze_commit_timestamp_utc"
        ),
        checkpoint_seal_created_at_utc=reconstruction.get(
            "checkpoint_seal_created_at_utc"
        ),
    )
    if beacon_errors:
        errors.append("EXP-279 raw beacon invalid: " + "; ".join(beacon_errors))

    if payload.get("current_source_tree_digest") != reconstruction.get(
        "source_tree_digest"
    ):
        errors.append("EXP-279 raw current/frozen source tree mismatch")
    sealed = seal.get("authorization_snapshot") or {}
    if payload.get("executor_code_digest") != (
        sealed.get("machinery_digests") or {}
    ).get("executor_code_digest"):
        errors.append("EXP-279 raw executor machinery digest mismatch")

    reserved = list(reconstruction.get("reserved_replicate_ids") or [])
    schedule = list(reconstruction.get("stratum_schedule") or [])
    rows = payload.get("per_replicate")
    if payload.get("confirmatory_n") != len(reserved):
        errors.append("EXP-279 raw confirmatory n mismatch")
    if payload.get("reserved_replicate_ids") != reserved:
        errors.append("EXP-279 raw reserved replicate lineage mismatch")
    if not isinstance(rows, list) or len(rows) != len(reserved):
        errors.append("EXP-279 raw confirmatory row count mismatch")
        return errors
    if [row.get("replicate") for row in rows] != reserved:
        errors.append("EXP-279 raw replicate lineage order mismatch")
        return errors
    expected_strata = [schedule[index % len(schedule)] for index in range(len(rows))]
    if [row.get("stratum") for row in rows] != expected_strata:
        errors.append("EXP-279 raw stratum schedule mismatch")
        return errors

    if checkpoint_receipt is None:
        errors.append("EXP-279 raw validator requires checkpoint receipt")
        return errors
    lineage = payload.get("lineage") or {}
    expected_lineage = {
        "protocol_digest": reconstruction.get("protocol_digest"),
        "source_tree_digest": reconstruction.get("source_tree_digest"),
        "seal_digest": seal.get("seal_digest"),
        "reconstruction_digest": reconstruction.get("reconstruction_digest"),
        "checkpoint_scientific_identity_digest": checkpoint_receipt.get(
            "scientific_identity_digest"
        ),
        "checkpoint_receipt_digest": checkpoint_receipt.get("receipt_digest"),
        "beacon_receipt_digest": beacon.get("receipt_digest"),
        "executor_code_digest": payload.get("executor_code_digest"),
    }
    if lineage != expected_lineage:
        errors.append("EXP-279 raw lineage binding mismatch")

    if checkpoint_path is None:
        errors.append(
            "EXP-279 raw validator requires checkpoint bytes for independent reconstruction"
        )
        return errors

    expected_input_receipt = _arm_input_receipt()
    for row in rows:
        if row.get("arm_input_receipt") != expected_input_receipt:
            errors.append("EXP-279 raw arm input separation receipt mismatch")
            break
        route = row.get("hybrid_route_receipt") or {}
        if route.get("strategy") != "propagation_then_branch_on_residual_uncertainty":
            errors.append("EXP-279 raw hybrid route strategy mismatch")
            break
        if not _finite_number(route.get("route_fraction")) or not 0.0 <= float(
            route.get("route_fraction")
        ) <= 1.0:
            errors.append("EXP-279 raw hybrid route fraction invalid")
            break
        for arm_id in ("propagation_only", "branch_only", "hybrid"):
            metrics = row.get(arm_id) or {}
            if not _finite_number(metrics.get("verified_solution_rate")):
                errors.append(f"EXP-279 raw {arm_id} solution rate is non-finite")
                break
            if not _finite_number(metrics.get(PRIMARY_METRIC)):
                errors.append(f"EXP-279 raw {arm_id} primary utility is non-finite")
                break
            if not _finite_number(metrics.get("accounted_flops_per_episode")) or float(
                metrics.get("accounted_flops_per_episode")
            ) <= 0.0:
                errors.append(f"EXP-279 raw {arm_id} accounted FLOP invalid")
                break
        if errors:
            break

        row_errors = validate_exp279_raw_row_against_reconstruction(
            raw_row=row,
            reconstruction_authorization=reconstruction,
            seal=seal,
            beacon_receipt=beacon,
            checkpoint_path=checkpoint_path,
            checkpoint_receipt=checkpoint_receipt,
        )
        if row_errors:
            errors.extend(row_errors)
            break

    return errors
