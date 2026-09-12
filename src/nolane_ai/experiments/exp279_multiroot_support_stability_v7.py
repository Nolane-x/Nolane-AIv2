from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from nolane_ai.protocol.evidence import canonical_sha256

from .exp279_branch_rescue_oracle import _branch_outcome_counts
from .exp279_routing_worlds import STRATA

SCHEMA_SHARD = "NLM-EXP-279-MULTIROOT-SUPPORT-SHARD-V7"
SCHEMA_BUDGET = "NLM-EXP-279-MULTIROOT-SUPPORT-BUDGET-V7"
ROOT_PREFIX = "20260912-exp279-multiroot-support-v7-dev"
TRAIN_BUDGETS = (60, 120)
CANONICAL_INDICES = (0, 1, 2, 3)
PROBE_INDICES = (0, 1, 2, 3)
PROBE_REPLICATES = 198
PROBE_FOLDS = 3
WILSON_Z = 1.6448536269514722
PROTOCOL_ID = "NLM-REASONING-STAGE-A-CONFIRMATORY-V1"
FROZEN_PROTOCOL_DIGEST = "c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440"

FROZEN_WORLD_GEOMETRY = {
    "batch_size": 8,
    "timesteps": 4,
    "variables": 6,
    "constraints": 3,
    "d_model": 64,
    "noise_std": 0.05,
}
FROZEN_ARM_GEOMETRY = {"hidden_size": 48, "target_parameters": 500_000}
FROZEN_OPTIMIZER = {"lr": 0.002, "weight_decay": 0.0}
FROZEN_ROUTE_THRESHOLD = 0.5

_COUNT_KEYS = ("branch_rescues", "branch_harms", "both_success", "both_failure")
_FALSE_BOUNDARY_KEYS = (
    "scientific_evidence_eligible",
    "evaluation_rng_stream_used",
    "evaluation_targets_used",
    "raw_examples_exported",
    "raw_model_outputs_exported",
    "selector_trained",
    "cdd_distiller_trained",
    "branch_preview_selector_trained",
    "threshold_tuned",
    "fresh_evaluation_lineage_may_be_reserved",
    "fresh_evaluation_lineage_consumed",
    "confirmatory_data_consumed",
    "challenge_materialized",
    "promotion_claimed",
)


def _require_budget(train_replicates: int) -> int:
    value = int(train_replicates)
    if value not in TRAIN_BUDGETS:
        raise ValueError(f"V7 train budget must be one of {TRAIN_BUDGETS}, got {train_replicates!r}")
    return value


def _require_index(value: int, allowed: tuple[int, ...], *, label: str) -> int:
    index = int(value)
    if index not in allowed:
        raise ValueError(f"V7 {label} must be one of {allowed}, got {value!r}")
    return index


def canonical_root(train_replicates: int, canonical_index: int) -> str:
    budget = _require_budget(train_replicates)
    index = _require_index(canonical_index, CANONICAL_INDICES, label="canonical index")
    return f"{ROOT_PREFIX}::train::{budget}::canonical::{index}"


def probe_root(train_replicates: int, canonical_index: int, probe_index: int) -> str:
    budget = _require_budget(train_replicates)
    canonical = _require_index(canonical_index, CANONICAL_INDICES, label="canonical index")
    probe = _require_index(probe_index, PROBE_INDICES, label="probe index")
    return f"{ROOT_PREFIX}::train::{budget}::probe::{canonical}::{probe}"


def expected_root_map(train_replicates: int) -> dict[str, list[str]]:
    budget = _require_budget(train_replicates)
    canonical_roots = [canonical_root(budget, c) for c in CANONICAL_INDICES]
    probe_roots = [
        probe_root(budget, c, p)
        for c in CANONICAL_INDICES
        for p in PROBE_INDICES
    ]
    if len(set(canonical_roots + probe_roots)) != 20:
        raise RuntimeError("V7 frozen root map is not unique")
    return {"canonical_roots": canonical_roots, "probe_roots": probe_roots}


def diagnostic_fold(replicate: int) -> int:
    replicate = int(replicate)
    if replicate < 0:
        raise ValueError("V7 diagnostic replicate must be non-negative")
    return (replicate // len(STRATA)) % PROBE_FOLDS


def wilson_lower_bound(rescues: int, episodes: int) -> float:
    x = int(rescues)
    n = int(episodes)
    if n <= 0 or x < 0 or x > n:
        raise ValueError("V7 Wilson inputs require episodes > 0 and 0 <= rescues <= episodes")
    if x == 0:
        return 0.0
    p = x / n
    z2 = WILSON_Z * WILSON_Z
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    radius = (
        WILSON_Z
        * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
        / denominator
    )
    return max(0.0, center - radius)


def _normalized_partition(counts: Mapping[str, Any]) -> dict[str, int]:
    try:
        episodes = int(counts["episodes"])
        normalized = {key: int(counts[key]) for key in _COUNT_KEYS}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("V7 support partition is missing integer sufficient statistics") from exc
    if episodes <= 0 or any(value < 0 for value in normalized.values()):
        raise ValueError("V7 support partition counts must be non-negative with episodes > 0")
    if sum(normalized.values()) != episodes:
        raise ValueError("V7 support partition did not close")
    return {"episodes": episodes, **normalized}


def summarize_probe_support(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(folds) != PROBE_FOLDS:
        raise ValueError(f"V7 probe support requires exactly {PROBE_FOLDS} diagnostic folds")
    normalized_folds: list[dict[str, Any]] = []
    for fold_index, counts in enumerate(folds):
        partition = _normalized_partition(counts)
        rescues = partition["branch_rescues"]
        nonrescue = partition["episodes"] - rescues
        normalized_folds.append(
            {
                "fold": fold_index,
                **partition,
                "nonrescue_count": nonrescue,
                "rescue_prevalence": rescues / partition["episodes"],
                "fold_rescue_supported": rescues > 0,
                "fold_nonrescue_supported": nonrescue > 0,
            }
        )
    aggregate = {
        key: sum(int(fold[key]) for fold in normalized_folds)
        for key in ("episodes", *_COUNT_KEYS)
    }
    if sum(aggregate[key] for key in _COUNT_KEYS) != aggregate["episodes"]:
        raise RuntimeError("V7 aggregate probe partition did not close")
    rescues = aggregate["branch_rescues"]
    nonrescue = aggregate["episodes"] - rescues
    return {
        **aggregate,
        "nonrescue_count": nonrescue,
        "rescue_prevalence": rescues / aggregate["episodes"],
        "probe_rescue_supported": rescues > 0,
        "probe_nonrescue_supported": nonrescue > 0,
        "fold_support_closed": all(
            bool(fold["fold_rescue_supported"] and fold["fold_nonrescue_supported"])
            for fold in normalized_folds
        ),
        "wilson_lower_bound": wilson_lower_bound(rescues, aggregate["episodes"]),
        "folds": normalized_folds,
    }


def summarize_canonical_support(probes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(probes) != len(PROBE_INDICES):
        raise ValueError("V7 canonical support requires exactly four probe roots")
    episodes = sum(int(probe["episodes"]) for probe in probes)
    rescues = sum(int(probe["branch_rescues"]) for probe in probes)
    harms = sum(int(probe["branch_harms"]) for probe in probes)
    both_success = sum(int(probe["both_success"]) for probe in probes)
    both_failure = sum(int(probe["both_failure"]) for probe in probes)
    if rescues + harms + both_success + both_failure != episodes:
        raise ValueError("V7 canonical support partition did not close")
    nonrescue = episodes - rescues
    return {
        "canonical_total_episodes": episodes,
        "canonical_total_rescues": rescues,
        "canonical_total_harms": harms,
        "canonical_total_both_success": both_success,
        "canonical_total_both_failure": both_failure,
        "canonical_total_nonrescue": nonrescue,
        "canonical_has_any_rescue": rescues > 0,
        "canonical_all_probe_roots_supported": all(
            bool(probe["probe_rescue_supported"] and probe["probe_nonrescue_supported"])
            for probe in probes
        ),
        "canonical_all_folds_supported": all(bool(probe["fold_support_closed"]) for probe in probes),
        "pooled_rescue_prevalence": rescues / episodes,
        "pooled_wilson_lower_bound": wilson_lower_bound(rescues, episodes),
    }


def classify_budget_support(canonical_supports: Sequence[Mapping[str, Any]]) -> str:
    if len(canonical_supports) != len(CANONICAL_INDICES):
        raise ValueError("V7 budget classification requires exactly four canonical roots")
    if any(int(item["canonical_total_rescues"]) == 0 for item in canonical_supports):
        return "MODEL_ROOT_SUPPORT_COLLAPSE"
    if any(
        not bool(item["canonical_all_probe_roots_supported"])
        or not bool(item["canonical_all_folds_supported"])
        for item in canonical_supports
    ):
        return "PROBE_SUPPORT_INTERMITTENT"
    return "SUPPORT_RECURRENT"


def _artifact_digest(receipt: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in receipt.items() if key != "artifact_digest"}
    return canonical_sha256(payload)


def _train_canonical_hybrid(
    *,
    root_seed: str,
    d_model: int,
    hidden_size: int,
    target_parameters: int,
    route_threshold: float,
    train_replicates: int,
    batch_size: int,
    timesteps: int,
    variables: int,
    constraints: int,
    noise_std: float,
    lr: float,
    weight_decay: float,
) -> tuple[Any, str, str, str | None]:
    from nolane_ai.training.optimizer import build_functional_optimizer

    from .exp279_paired_runner import _build_seeded_triplet, _functional_state_digest, _train_step
    from .exp279_routing_worlds import Exp279RoutingGenerator
    from .matched_routing_arms import audit_matched_exp279_arm_triplet

    propagation, branch, hybrid, model_init_seed = _build_seeded_triplet(
        root_seed=root_seed,
        d_model=d_model,
        hidden_size=hidden_size,
        target_parameters=target_parameters,
        route_threshold=route_threshold,
    )
    pair_audit = audit_matched_exp279_arm_triplet(
        propagation,
        branch,
        hybrid,
        timesteps=timesteps,
        variables=variables,
        constraints=constraints,
        max_accounted_flops_per_episode=None,
    )
    required = (
        "parameter_match",
        "functional_parameter_match",
        "active_functional_parameter_match",
        "optimizer_visible_parameter_match",
        "reclaimed_parameter_assignment_closed",
        "compute_budget_closed",
    )
    if not all(pair_audit.get(key) is True for key in required):
        raise RuntimeError("V7 matched resource contract did not close")

    generator = Exp279RoutingGenerator(root_seed=root_seed)
    optimizer = build_functional_optimizer(hybrid, lr=lr, weight_decay=weight_decay)
    for replicate in range(train_replicates):
        batch = generator.make_batch(
            replicate=replicate,
            batch_size=batch_size,
            timesteps=timesteps,
            variables=variables,
            constraints=constraints,
            d_model=d_model,
            noise_std=noise_std,
            rng_stream="augmentation",
            stratum=STRATA[replicate % len(STRATA)],
        )
        _train_step(
            hybrid,
            optimizer,
            arm_id="hybrid",
            surface_events=batch.surface_events,
            variable_states=batch.variable_states,
            incidence=batch.incidence,
            targets=batch.targets,
        )
    final_digest = _functional_state_digest(hybrid)
    hybrid.eval()
    for parameter in hybrid.parameters():
        parameter.requires_grad_(False)
    return hybrid, model_init_seed, final_digest, pair_audit.get("pair_audit_digest")


def run_exp279_multiroot_support_stability_v7_shard(
    *,
    train_replicates: int,
    canonical_index: int,
    d_model: int,
    hidden_size: int,
    target_parameters: int,
    route_threshold: float,
    probe_replicates: int,
    batch_size: int,
    timesteps: int,
    variables: int,
    constraints: int,
    noise_std: float,
    lr: float,
    weight_decay: float,
    protocol_digest: str,
    code_digest: str,
) -> dict[str, Any]:
    import torch

    from .exp279_paired_runner import (
        _hybrid_forced_branch_decision_logits,
        _hybrid_stop_decision_logits,
    )
    from .exp279_routing_worlds import Exp279RoutingGenerator

    budget = _require_budget(train_replicates)
    canonical = _require_index(canonical_index, CANONICAL_INDICES, label="canonical index")
    if not protocol_digest or not code_digest:
        raise ValueError("V7 protocol_digest and code_digest are required")
    if min(d_model, hidden_size, target_parameters, probe_replicates, batch_size, timesteps, variables, constraints) <= 0:
        raise ValueError("V7 dimensions and counts must be positive")
    if probe_replicates < PROBE_FOLDS * len(STRATA) or probe_replicates % PROBE_FOLDS != 0:
        raise ValueError("V7 probe_replicates must cover all diagnostic folds and be divisible by three")
    if constraints > variables or timesteps < constraints:
        raise ValueError("V7 world geometry is invalid")
    if not 0.0 <= route_threshold <= 1.0:
        raise ValueError("V7 route threshold must be within [0,1]")
    if noise_std < 0.0 or lr <= 0.0 or weight_decay < 0.0:
        raise ValueError("V7 noise/optimizer configuration is invalid")

    canonical_seed = canonical_root(budget, canonical)
    hybrid, model_init_seed, final_digest, pair_audit_digest = _train_canonical_hybrid(
        root_seed=canonical_seed,
        d_model=d_model,
        hidden_size=hidden_size,
        target_parameters=target_parameters,
        route_threshold=route_threshold,
        train_replicates=budget,
        batch_size=batch_size,
        timesteps=timesteps,
        variables=variables,
        constraints=constraints,
        noise_std=noise_std,
        lr=lr,
        weight_decay=weight_decay,
    )

    probe_receipts: list[dict[str, Any]] = []
    with torch.no_grad():
        for p in PROBE_INDICES:
            root = probe_root(budget, canonical, p)
            generator = Exp279RoutingGenerator(root_seed=root)
            fold_counts = [
                {"episodes": 0, "branch_rescues": 0, "branch_harms": 0, "both_success": 0, "both_failure": 0}
                for _ in range(PROBE_FOLDS)
            ]
            for replicate in range(probe_replicates):
                batch = generator.make_batch(
                    replicate=replicate,
                    batch_size=batch_size,
                    timesteps=timesteps,
                    variables=variables,
                    constraints=constraints,
                    d_model=d_model,
                    noise_std=noise_std,
                    rng_stream="augmentation",
                    stratum=STRATA[replicate % len(STRATA)],
                )
                stop_logits = _hybrid_stop_decision_logits(
                    hybrid,
                    surface_events=batch.surface_events,
                    variable_states=batch.variable_states,
                    incidence=batch.incidence,
                )
                branch_logits = _hybrid_forced_branch_decision_logits(
                    hybrid,
                    surface_events=batch.surface_events,
                    variable_states=batch.variable_states,
                    incidence=batch.incidence,
                )
                counts = _branch_outcome_counts(
                    stop_logits=stop_logits,
                    branch_logits=branch_logits,
                    targets=batch.targets,
                )
                fold = fold_counts[diagnostic_fold(replicate)]
                fold["episodes"] += int(counts["episodes"])
                for key in _COUNT_KEYS:
                    fold[key] += int(counts[key])
            summary = summarize_probe_support(fold_counts)
            probe_receipts.append({"probe_index": p, "probe_root": root, **summary})

    canonical_support = summarize_canonical_support(probe_receipts)
    fragment = {
        "canonical_root": canonical_seed,
        "probe_roots": [item["probe_root"] for item in probe_receipts],
    }
    receipt: dict[str, Any] = {
        "schema": SCHEMA_SHARD,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "scientific_evidence_eligible": False,
        "protocol_id": PROTOCOL_ID,
        "protocol_digest": protocol_digest,
        "code_digest": code_digest,
        "root_prefix": ROOT_PREFIX,
        "train_replicates": budget,
        "canonical_index": canonical,
        "canonical_root": canonical_seed,
        "model_init_seed": model_init_seed,
        "training_rng_stream": "augmentation",
        "probe_rng_stream": "augmentation",
        "evaluation_rng_stream_used": False,
        "evaluation_targets_used": False,
        "raw_examples_exported": False,
        "raw_model_outputs_exported": False,
        "selector_trained": False,
        "cdd_distiller_trained": False,
        "branch_preview_selector_trained": False,
        "threshold_tuned": False,
        "route_config": {"canonical_threshold": float(route_threshold), "threshold_tuned": False},
        "world_geometry": {
            "batch_size": int(batch_size),
            "timesteps": int(timesteps),
            "variables": int(variables),
            "constraints": int(constraints),
            "d_model": int(d_model),
            "noise_std": float(noise_std),
        },
        "arm_geometry": {
            "hidden_size": int(hidden_size),
            "target_parameters": int(target_parameters),
        },
        "optimizer": {"lr": float(lr), "weight_decay": float(weight_decay)},
        "probe_config": {"replicates": int(probe_replicates), "folds": PROBE_FOLDS, "probe_roots": 4},
        "final_canonical_digest": final_digest,
        "resource_pair_audit_digest": pair_audit_digest,
        "root_map_fragment_digest": canonical_sha256(fragment),
        "probe_receipts": probe_receipts,
        "canonical_support": canonical_support,
        "fresh_evaluation_lineage_may_be_reserved": False,
        "fresh_evaluation_lineage_consumed": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
    }
    receipt["artifact_digest"] = _artifact_digest(receipt)
    return receipt


def _expect_false_boundaries(receipt: Mapping[str, Any]) -> None:
    for key in _FALSE_BOUNDARY_KEYS:
        if receipt.get(key) is not False:
            raise ValueError(f"V7 boundary mismatch for {key}")


def validate_v7_shard(
    receipt: Mapping[str, Any],
    *,
    require_frozen_scientific: bool = False,
) -> None:
    if receipt.get("schema") != SCHEMA_SHARD:
        raise ValueError("V7 shard schema mismatch")
    if receipt.get("evidence_level") != "EV-E2" or receipt.get("decision") != "UNVERIFIED":
        raise ValueError("V7 shard evidence state mismatch")
    if receipt.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("V7 shard protocol id mismatch")
    if receipt.get("root_prefix") != ROOT_PREFIX:
        raise ValueError("V7 shard root prefix mismatch")
    _expect_false_boundaries(receipt)
    if receipt.get("training_rng_stream") != "augmentation" or receipt.get("probe_rng_stream") != "augmentation":
        raise ValueError("V7 shard RNG stream mismatch")

    budget = _require_budget(int(receipt.get("train_replicates", -1)))
    canonical = _require_index(int(receipt.get("canonical_index", -1)), CANONICAL_INDICES, label="canonical index")
    if receipt.get("canonical_root") != canonical_root(budget, canonical):
        raise ValueError("V7 shard canonical root mismatch")
    probes = receipt.get("probe_receipts")
    if not isinstance(probes, list) or len(probes) != 4:
        raise ValueError("V7 shard must contain exactly four probe receipts")
    expected_probe_roots = [probe_root(budget, canonical, p) for p in PROBE_INDICES]
    actual_probe_roots = [str(item.get("probe_root")) for item in probes]
    if actual_probe_roots != expected_probe_roots or len(set(actual_probe_roots)) != 4:
        raise ValueError("V7 shard probe roots mismatch")
    for p, item in enumerate(probes):
        if int(item.get("probe_index", -1)) != p:
            raise ValueError("V7 shard probe index mismatch")
        folds = item.get("folds")
        if not isinstance(folds, list) or len(folds) != PROBE_FOLDS:
            raise ValueError("V7 shard probe fold count mismatch")
        rebuilt = summarize_probe_support(folds)
        for key in (
            "episodes",
            "branch_rescues",
            "branch_harms",
            "both_success",
            "both_failure",
            "nonrescue_count",
            "probe_rescue_supported",
            "probe_nonrescue_supported",
            "fold_support_closed",
        ):
            if item.get(key) != rebuilt.get(key):
                raise ValueError(f"V7 shard probe sufficient statistic mismatch: {key}")
        if not math.isclose(float(item.get("wilson_lower_bound", -1.0)), float(rebuilt["wilson_lower_bound"]), rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("V7 shard probe Wilson mismatch")
    rebuilt_canonical = summarize_canonical_support(probes)
    supplied_canonical = receipt.get("canonical_support")
    if not isinstance(supplied_canonical, Mapping):
        raise ValueError("V7 shard canonical support missing")
    for key, value in rebuilt_canonical.items():
        supplied = supplied_canonical.get(key)
        if isinstance(value, float):
            if not math.isclose(float(supplied), value, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(f"V7 shard canonical support mismatch: {key}")
        elif supplied != value:
            raise ValueError(f"V7 shard canonical support mismatch: {key}")

    fragment = {"canonical_root": receipt["canonical_root"], "probe_roots": actual_probe_roots}
    if receipt.get("root_map_fragment_digest") != canonical_sha256(fragment):
        raise ValueError("V7 shard root map digest mismatch")
    if receipt.get("artifact_digest") != _artifact_digest(receipt):
        raise ValueError("V7 shard artifact digest mismatch")

    if require_frozen_scientific:
        if receipt.get("protocol_digest") != FROZEN_PROTOCOL_DIGEST:
            raise ValueError("V7 shard frozen protocol digest mismatch")
        if receipt.get("world_geometry") != FROZEN_WORLD_GEOMETRY:
            raise ValueError("V7 shard frozen world geometry mismatch")
        if receipt.get("arm_geometry") != FROZEN_ARM_GEOMETRY:
            raise ValueError("V7 shard frozen arm geometry mismatch")
        if receipt.get("optimizer") != FROZEN_OPTIMIZER:
            raise ValueError("V7 shard frozen optimizer mismatch")
        route = receipt.get("route_config")
        if not isinstance(route, Mapping) or float(route.get("canonical_threshold", -1.0)) != FROZEN_ROUTE_THRESHOLD:
            raise ValueError("V7 shard frozen route threshold mismatch")
        if receipt.get("probe_config") != {"replicates": PROBE_REPLICATES, "folds": PROBE_FOLDS, "probe_roots": 4}:
            raise ValueError("V7 shard frozen probe configuration mismatch")
        expected_episodes = PROBE_REPLICATES * FROZEN_WORLD_GEOMETRY["batch_size"]
        if any(int(item["episodes"]) != expected_episodes for item in probes):
            raise ValueError("V7 shard frozen per-probe episode count mismatch")
        if int(supplied_canonical["canonical_total_episodes"]) != expected_episodes * 4:
            raise ValueError("V7 shard frozen canonical episode count mismatch")


def aggregate_exp279_multiroot_support_stability_v7_budget(
    shards: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(shards) != len(CANONICAL_INDICES):
        raise ValueError("V7 budget reducer requires exactly four canonical shards")
    for shard in shards:
        validate_v7_shard(shard, require_frozen_scientific=True)
    ordered = sorted(shards, key=lambda item: int(item["canonical_index"]))
    indices = [int(item["canonical_index"]) for item in ordered]
    if indices != list(CANONICAL_INDICES):
        raise ValueError("V7 budget reducer canonical index set mismatch")
    budgets = {int(item["train_replicates"]) for item in ordered}
    if len(budgets) != 1:
        raise ValueError("V7 budget reducer cannot mix train budgets")
    budget = budgets.pop()
    code_digests = {str(item["code_digest"]) for item in ordered}
    if len(code_digests) != 1:
        raise ValueError("V7 budget reducer code provenance mismatch")

    root_map = expected_root_map(budget)
    canonical_roots = [str(item["canonical_root"]) for item in ordered]
    probe_roots = [str(probe["probe_root"]) for item in ordered for probe in item["probe_receipts"]]
    if canonical_roots != root_map["canonical_roots"] or probe_roots != root_map["probe_roots"]:
        raise ValueError("V7 budget reducer root map mismatch")
    if len(set(canonical_roots + probe_roots)) != 20:
        raise ValueError("V7 budget reducer root map is not unique")

    canonical_records = [
        {
            "canonical_index": int(item["canonical_index"]),
            "canonical_root": str(item["canonical_root"]),
            "final_canonical_digest": str(item["final_canonical_digest"]),
            "artifact_digest": str(item["artifact_digest"]),
            **dict(item["canonical_support"]),
        }
        for item in ordered
    ]
    pair_records = [
        {
            "canonical_index": int(item["canonical_index"]),
            "probe_index": int(probe["probe_index"]),
            "canonical_root": str(item["canonical_root"]),
            "probe_root": str(probe["probe_root"]),
            **{key: probe[key] for key in (
                "episodes", "branch_rescues", "branch_harms", "both_success", "both_failure",
                "nonrescue_count", "rescue_prevalence", "probe_rescue_supported",
                "probe_nonrescue_supported", "fold_support_closed", "wilson_lower_bound",
            )},
        }
        for item in ordered
        for probe in item["probe_receipts"]
    ]
    fold_records = [
        {
            "canonical_index": int(item["canonical_index"]),
            "probe_index": int(probe["probe_index"]),
            "canonical_root": str(item["canonical_root"]),
            "probe_root": str(probe["probe_root"]),
            **dict(fold),
        }
        for item in ordered
        for probe in item["probe_receipts"]
        for fold in probe["folds"]
    ]
    if len(pair_records) != 16 or len(fold_records) != 48:
        raise RuntimeError("V7 budget reducer record cardinality mismatch")

    classification = classify_budget_support(canonical_records)
    prevalence_floor = min(float(item["pooled_wilson_lower_bound"]) for item in canonical_records)
    receipt: dict[str, Any] = {
        "schema": SCHEMA_BUDGET,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "scientific_evidence_eligible": False,
        "protocol_id": PROTOCOL_ID,
        "protocol_digest": FROZEN_PROTOCOL_DIGEST,
        "code_digest": next(iter(code_digests)),
        "root_prefix": ROOT_PREFIX,
        "train_replicates": budget,
        "route_config": {"canonical_threshold": FROZEN_ROUTE_THRESHOLD, "threshold_tuned": False},
        "world_geometry": dict(FROZEN_WORLD_GEOMETRY),
        "arm_geometry": dict(FROZEN_ARM_GEOMETRY),
        "optimizer": dict(FROZEN_OPTIMIZER),
        "probe_config": {"replicates": PROBE_REPLICATES, "folds": PROBE_FOLDS, "probe_roots_per_canonical": 4},
        "training_rng_stream": "augmentation",
        "probe_rng_stream": "augmentation",
        "evaluation_rng_stream_used": False,
        "evaluation_targets_used": False,
        "raw_examples_exported": False,
        "raw_model_outputs_exported": False,
        "selector_trained": False,
        "cdd_distiller_trained": False,
        "branch_preview_selector_trained": False,
        "threshold_tuned": False,
        "root_map": root_map,
        "root_map_digest": canonical_sha256(root_map),
        "canonical_records": canonical_records,
        "pair_records": pair_records,
        "fold_records": fold_records,
        "canonical_prevalence_floor": prevalence_floor,
        "train_cell_classification": classification,
        "total_episodes": sum(int(item["canonical_total_episodes"]) for item in canonical_records),
        "fresh_evaluation_lineage_may_be_reserved": False,
        "fresh_evaluation_lineage_consumed": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
    }
    receipt["artifact_digest"] = _artifact_digest(receipt)
    return receipt
