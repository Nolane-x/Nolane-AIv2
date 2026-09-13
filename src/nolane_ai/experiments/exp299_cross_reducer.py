from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from nolane_ai.protocol.evidence import canonical_sha256

SCHEMA = "NLM-EXP-299-NATIVE-FIDELITY-CROSS-V1"
ROOT_ESTABLISHED = "NATIVE_FIDELITY_SURVIVES_SCAFFOLD_REMOVAL"
RECURRENT = "NATIVE_FIDELITY_SURVIVES_SCAFFOLD_REMOVAL_RECURRENT"
NOT_RECURRENT = "NATIVE_FIDELITY_NOT_ESTABLISHED_RECURRENT"
AUTHORIZED_SCOPE = "DESIGN_EXP300_INTEGRATED_COURT_ONLY"

FALSE_FLAGS = (
    "scientific_evidence_eligible",
    "confirmatory_data_consumed",
    "challenge_materialized",
    "promotion_claimed",
    "unrestricted_semantic_authority_claimed",
    "open_language_understanding_claimed",
    "causal_discovery_claimed",
    "general_code_reasoning_claimed",
    "durable_lifelong_internalization_claimed",
    "exp300_execution_authorized",
)
IDENTITY_FIELDS = (
    "repository_head",
    "protocol_digest",
    "geometry_digest",
    "code_digest",
)
CONFIG_IDENTITY_FIELDS = (
    "fit_replicates",
    "fit_start_replicate",
    "eval_replicates",
    "eval_start_replicate",
    "d_model",
    "hidden_size",
    "target_parameters",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "gradient_clip_norm",
    "authority_threshold",
    "max_exact_probes",
    "domains",
    "candidates_per_replicate",
    "native_ba_floor",
    "gain_mesi",
    "wrong_authority_ceiling",
    "faithful_rejection_ceiling",
    "fit_passes",
)


def _artifact_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("artifact_digest", None)
    return canonical_sha256(clean)


def _validate_hex(name: str, value: Any, length: int) -> str | None:
    if not isinstance(value, str) or len(value) != length:
        return f"EXP-299 cross {name} identity invalid"
    try:
        bytes.fromhex(value)
    except ValueError:
        return f"EXP-299 cross {name} identity invalid"
    return None


def _normalize_roots(root_receipts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    # Local import is deliberate: unit tests can isolate reducer semantics while
    # the production path still validates every sealed root before reduction.
    from nolane_ai.experiments.exp299_native_runner import validate_exp299_root

    roots = list(root_receipts)
    if len(roots) != 4:
        raise ValueError("EXP-299 cross reducer requires exactly four root receipts")

    root_errors: list[str] = []
    for index, root in enumerate(roots):
        errors = validate_exp299_root(root)
        if errors:
            root_errors.append(f"root[{index}]: {'; '.join(errors)}")
    if root_errors:
        raise ValueError("EXP-299 invalid root receipt: " + " | ".join(root_errors))

    indices = sorted(int(root.get("canonical_index", -1)) for root in roots)
    if indices != [0, 1, 2, 3]:
        raise ValueError("EXP-299 cross reducer requires canonical indices [0,1,2,3]")

    ordered = sorted(roots, key=lambda root: int(root["canonical_index"]))
    baseline = ordered[0]
    baseline_config = baseline.get("config") or {}
    baseline_encoder = baseline.get("structural_encoder") or {}
    for root in ordered[1:]:
        if any(root.get(field) != baseline.get(field) for field in IDENTITY_FIELDS):
            raise ValueError("EXP-299 root identity drift detected")
        config = root.get("config") or {}
        if any(config.get(field) != baseline_config.get(field) for field in CONFIG_IDENTITY_FIELDS):
            raise ValueError("EXP-299 root identity drift detected")
        encoder = root.get("structural_encoder") or {}
        if encoder.get("contract_digest") != baseline_encoder.get("contract_digest"):
            raise ValueError("EXP-299 root identity drift detected")

    digests = [root.get("artifact_digest") for root in ordered]
    if len(set(digests)) != 4:
        raise ValueError("EXP-299 cross reducer requires four distinct root artifacts")
    return ordered


def reduce_exp299_roots(root_receipts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    roots = _normalize_roots(root_receipts)
    root_decisions = [str(root["decision"]) for root in roots]
    established_count = sum(decision == ROOT_ESTABLISHED for decision in root_decisions)
    recurrent = established_count == 4
    baseline = roots[0]
    baseline_config = baseline.get("config") or {}
    baseline_encoder = baseline.get("structural_encoder") or {}

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": "EXP-299",
        "evidence_level": "EV-E2",
        "canonical_indices": [0, 1, 2, 3],
        "repository_head": baseline["repository_head"],
        "protocol_digest": baseline["protocol_digest"],
        "geometry_digest": baseline["geometry_digest"],
        "code_digest": baseline["code_digest"],
        "structural_encoder_contract_digest": baseline_encoder.get("contract_digest"),
        "config_identity_digest": canonical_sha256(
            {field: baseline_config.get(field) for field in CONFIG_IDENTITY_FIELDS}
        ),
        "source_root_artifact_digests": [root["artifact_digest"] for root in roots],
        "source_root_decisions": root_decisions,
        "established_root_count": established_count,
        "required_established_root_count": 4,
        "decision": RECURRENT if recurrent else NOT_RECURRENT,
        "successor_design_authorized": recurrent,
        "authorization_scope": AUTHORIZED_SCOPE if recurrent else "NONE",
        "scientific_evidence_eligible": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
        "unrestricted_semantic_authority_claimed": False,
        "open_language_understanding_claimed": False,
        "causal_discovery_claimed": False,
        "general_code_reasoning_claimed": False,
        "durable_lifelong_internalization_claimed": False,
        "exp300_execution_authorized": False,
    }
    payload["artifact_digest"] = _artifact_digest(payload)
    return payload


def validate_exp299_cross(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("EXP-299 cross schema mismatch")
    if payload.get("experiment_id") != "EXP-299" or payload.get("evidence_level") != "EV-E2":
        errors.append("EXP-299 cross epistemic identity mismatch")
    if payload.get("artifact_digest") != _artifact_digest(payload):
        errors.append("EXP-299 cross artifact digest mismatch")
    if payload.get("canonical_indices") != [0, 1, 2, 3]:
        errors.append("EXP-299 cross canonical indices mismatch")

    digests = payload.get("source_root_artifact_digests")
    if not isinstance(digests, list) or len(digests) != 4 or len(set(digests)) != 4:
        errors.append("EXP-299 cross source root digests invalid")
    else:
        for digest in digests:
            error = _validate_hex("source_root_artifact_digest", digest, 64)
            if error:
                errors.append(error)
                break

    for name, length in (
        ("repository_head", 40),
        ("protocol_digest", 64),
        ("geometry_digest", 64),
        ("code_digest", 64),
        ("structural_encoder_contract_digest", 64),
        ("config_identity_digest", 64),
    ):
        error = _validate_hex(name, payload.get(name), length)
        if error:
            errors.append(error)

    decisions = payload.get("source_root_decisions")
    if not isinstance(decisions, list) or len(decisions) != 4:
        errors.append("EXP-299 cross source root decisions invalid")
        decisions = []
    else:
        valid = {ROOT_ESTABLISHED, "NATIVE_FIDELITY_NOT_ESTABLISHED"}
        if any(decision not in valid for decision in decisions):
            errors.append("EXP-299 cross source root decision value invalid")

    established = sum(decision == ROOT_ESTABLISHED for decision in decisions)
    if payload.get("established_root_count") != established:
        errors.append("EXP-299 cross established-root count mismatch")
    if payload.get("required_established_root_count") != 4:
        errors.append("EXP-299 cross required-root count mismatch")

    recurrent = len(decisions) == 4 and established == 4
    expected_decision = RECURRENT if recurrent else NOT_RECURRENT
    expected_scope = AUTHORIZED_SCOPE if recurrent else "NONE"
    if payload.get("decision") != expected_decision:
        errors.append("EXP-299 cross decision mismatch")
    if payload.get("successor_design_authorized") is not recurrent:
        errors.append("EXP-299 cross authorization mismatch")
    if payload.get("authorization_scope") != expected_scope:
        errors.append("EXP-299 cross authorization scope mismatch")

    for flag in FALSE_FLAGS:
        if payload.get(flag) is not False:
            errors.append(f"EXP-299 forbidden cross flag enabled: {flag}")
    return errors
