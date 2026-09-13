from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from nolane_ai.protocol.evidence import canonical_sha256

SCHEMA = "NLM-EXP-298-CROSS-DOMAIN-FIDELITY-CROSS-V1"
ROOT_ESTABLISHED = "CROSS_DOMAIN_FIDELITY_TRANSFER_ESTABLISHED"
RECURRENT = "CROSS_DOMAIN_FIDELITY_TRANSFER_RECURRENT"
NOT_RECURRENT = "CROSS_DOMAIN_FIDELITY_TRANSFER_NOT_RECURRENT"
AUTHORIZED_SCOPE = "DESIGN_EXP299_SCAFFOLD_REMOVAL_COURT_ONLY"

FALSE_FLAGS = (
    "scientific_evidence_eligible",
    "confirmatory_data_consumed",
    "challenge_materialized",
    "promotion_claimed",
    "unrestricted_semantic_authority_claimed",
    "open_language_understanding_claimed",
    "causal_discovery_claimed",
    "general_code_reasoning_claimed",
    "exp290_authority_inherited",
    "exp291_296_authority_inherited",
    "exp300_authorized",
)
IDENTITY_FIELDS = (
    "repository_head",
    "protocol_digest",
    "geometry_digest",
    "code_digest",
)


def _artifact_digest(payload: dict[str, Any]) -> str:
    clean = deepcopy(payload)
    clean.pop("artifact_digest", None)
    return canonical_sha256(clean)


def _validate_hex(name: str, value: Any, length: int) -> str | None:
    if not isinstance(value, str) or len(value) != length:
        return f"EXP-298 cross {name} identity invalid"
    try:
        bytes.fromhex(value)
    except ValueError:
        return f"EXP-298 cross {name} identity invalid"
    return None


def _normalize_roots(root_receipts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    from nolane_ai.experiments.exp298_paired_runner import validate_exp298_root

    roots = list(root_receipts)
    if len(roots) != 4:
        raise ValueError("EXP-298 cross reducer requires exactly four root receipts")

    root_errors: list[str] = []
    for index, root in enumerate(roots):
        errors = validate_exp298_root(root)
        if errors:
            root_errors.append(f"root[{index}]: {'; '.join(errors)}")
    if root_errors:
        raise ValueError("EXP-298 invalid root receipt: " + " | ".join(root_errors))

    indices = sorted(int(root.get("canonical_index", -1)) for root in roots)
    if indices != [0, 1, 2, 3]:
        raise ValueError("EXP-298 cross reducer requires canonical indices [0,1,2,3]")

    ordered = sorted(roots, key=lambda root: int(root["canonical_index"]))
    baseline = ordered[0]
    for root in ordered[1:]:
        if any(root.get(field) != baseline.get(field) for field in IDENTITY_FIELDS):
            raise ValueError("EXP-298 root identity drift detected")
        base_config = baseline.get("config") or {}
        config = root.get("config") or {}
        for field in (
            "eval_replicates",
            "eval_start_replicate",
            "d_model",
            "hidden_size",
            "target_parameters",
            "max_exact_probes",
            "domains",
            "candidates_per_replicate",
        ):
            if config.get(field) != base_config.get(field):
                raise ValueError("EXP-298 root identity drift detected")
    return ordered


def reduce_exp298_roots(root_receipts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    roots = _normalize_roots(root_receipts)
    root_decisions = [str(root["decision"]) for root in roots]
    recurrent = all(decision == ROOT_ESTABLISHED for decision in root_decisions)
    baseline = roots[0]

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": "EXP-298",
        "evidence_level": "EV-E2",
        "canonical_indices": [0, 1, 2, 3],
        "repository_head": baseline["repository_head"],
        "protocol_digest": baseline["protocol_digest"],
        "geometry_digest": baseline["geometry_digest"],
        "code_digest": baseline["code_digest"],
        "source_root_artifact_digests": [root["artifact_digest"] for root in roots],
        "source_root_decisions": root_decisions,
        "established_root_count": sum(decision == ROOT_ESTABLISHED for decision in root_decisions),
        "required_established_root_count": 4,
        "decision": RECURRENT if recurrent else NOT_RECURRENT,
        "successor_design_authorized": recurrent,
        "authorization_scope": AUTHORIZED_SCOPE if recurrent else "NONE",
        "exp300_authorized": False,
        "scientific_evidence_eligible": False,
        "confirmatory_data_consumed": False,
        "challenge_materialized": False,
        "promotion_claimed": False,
        "unrestricted_semantic_authority_claimed": False,
        "open_language_understanding_claimed": False,
        "causal_discovery_claimed": False,
        "general_code_reasoning_claimed": False,
        "exp290_authority_inherited": False,
        "exp291_296_authority_inherited": False,
    }
    payload["artifact_digest"] = _artifact_digest(payload)
    return payload


def validate_exp298_cross(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append("EXP-298 cross schema mismatch")
    if payload.get("experiment_id") != "EXP-298" or payload.get("evidence_level") != "EV-E2":
        errors.append("EXP-298 cross epistemic identity mismatch")
    if payload.get("artifact_digest") != _artifact_digest(payload):
        errors.append("EXP-298 cross artifact digest mismatch")
    if payload.get("canonical_indices") != [0, 1, 2, 3]:
        errors.append("EXP-298 cross canonical indices mismatch")

    digests = payload.get("source_root_artifact_digests")
    if not isinstance(digests, list) or len(digests) != 4 or len(set(digests)) != 4:
        errors.append("EXP-298 cross source root digests invalid")
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
    ):
        error = _validate_hex(name, payload.get(name), length)
        if error:
            errors.append(error)

    decisions = payload.get("source_root_decisions")
    if not isinstance(decisions, list) or len(decisions) != 4:
        errors.append("EXP-298 cross source root decisions invalid")
        decisions = []
    established = sum(decision == ROOT_ESTABLISHED for decision in decisions)
    if payload.get("established_root_count") != established:
        errors.append("EXP-298 cross established-root count mismatch")
    if payload.get("required_established_root_count") != 4:
        errors.append("EXP-298 cross required-root count mismatch")

    recurrent = len(decisions) == 4 and established == 4
    expected_decision = RECURRENT if recurrent else NOT_RECURRENT
    expected_scope = AUTHORIZED_SCOPE if recurrent else "NONE"
    if payload.get("decision") != expected_decision:
        errors.append("EXP-298 cross decision mismatch")
    if payload.get("successor_design_authorized") is not recurrent:
        errors.append("EXP-298 cross authorization mismatch")
    if payload.get("authorization_scope") != expected_scope:
        errors.append("EXP-298 cross authorization scope mismatch")

    for flag in FALSE_FLAGS:
        if payload.get(flag) is not False:
            errors.append(f"EXP-298 forbidden cross flag enabled: {flag}")
    return errors
