from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .seeds import STREAM_NAMES

EXPECTED_FIRST_GATES = ("EXP-277", "EXP-279", "EXP-282", "EXP-286", "EXP-289", "EXP-297")
EXPECTED_RESULT_STATES = (
    "PROMOTE_TO_NEXT_STAGE",
    "HOLD_UNSTABLE",
    "PRACTICALLY_EQUIVALENT_USE_SIMPLER_RIVAL",
    "KILL_SUBSYSTEM",
    "INVALID_RUN",
    "UNVERIFIED",
)


class ProtocolValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    experiment_id: str
    hypothesis_id: str
    question: str
    arms: tuple[dict[str, Any], ...]
    primary_endpoint: dict[str, Any]
    protected_endpoints: tuple[dict[str, Any], ...]
    mesi: dict[str, Any]
    sample_size_plan: dict[str, Any]
    analysis_method: str
    multiplicity_family: str
    resource_match: dict[str, Any]
    failure_policy: dict[str, Any]
    challenge_generator: dict[str, Any]
    decision_rule: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ProtocolSpec:
    protocol_id: str
    status: str
    source_revision: str
    result_states: tuple[str, ...]
    rng_streams: tuple[str, ...]
    experiments: tuple[ExperimentSpec, ...]

    @property
    def experiment_ids(self) -> tuple[str, ...]:
        return tuple(exp.experiment_id for exp in self.experiments)


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def load_and_validate_protocol(path: str | Path) -> ProtocolSpec:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    errors: list[str] = []
    experiments_raw = raw.get("experiments", [])
    _require(raw.get("protocol_id") == "NLM-REASONING-STAGE-A-CONFIRMATORY-V1", "bad protocol_id", errors)
    _require(raw.get("status") == "FROZEN_V1", "protocol must be FROZEN_V1", errors)
    _require(tuple(raw.get("result_states", [])) == EXPECTED_RESULT_STATES, "result-state vocabulary drift", errors)
    _require(tuple(raw.get("rng", {}).get("streams", [])) == STREAM_NAMES, "RNG stream namespace drift", errors)
    _require(tuple(item.get("experiment_id") for item in experiments_raw) == EXPECTED_FIRST_GATES, "first-gate set/order drift", errors)

    experiments: list[ExperimentSpec] = []
    required = (
        "experiment_id", "hypothesis_id", "question", "arms", "primary_endpoint",
        "protected_endpoints", "mesi", "sample_size_plan", "analysis_method",
        "multiplicity_family", "resource_match", "failure_policy", "challenge_generator",
        "decision_rule",
    )
    for item in experiments_raw:
        for key in required:
            _require(key in item, f"{item.get('experiment_id', '?')}: missing {key}", errors)
        if any(key not in item for key in required):
            continue
        _require(item["experiment_id"].startswith("EXP-"), f"{item['experiment_id']}: noncanonical ID", errors)
        _require(len(item["arms"]) >= 2, f"{item['experiment_id']}: need >=2 arms", errors)
        _require(float(item["mesi"].get("value", 0)) > 0, f"{item['experiment_id']}: MESI must be numeric and >0", errors)
        _require(float(item["sample_size_plan"].get("power_target", 0)) >= 0.8, f"{item['experiment_id']}: power target too low", errors)
        experiments.append(ExperimentSpec(
            experiment_id=item["experiment_id"], hypothesis_id=item["hypothesis_id"], question=item["question"],
            arms=tuple(item["arms"]), primary_endpoint=item["primary_endpoint"],
            protected_endpoints=tuple(item["protected_endpoints"]), mesi=item["mesi"],
            sample_size_plan=item["sample_size_plan"], analysis_method=item["analysis_method"],
            multiplicity_family=item["multiplicity_family"], resource_match=item["resource_match"],
            failure_policy=item["failure_policy"], challenge_generator=item["challenge_generator"],
            decision_rule=item["decision_rule"],
        ))

    if errors:
        raise ProtocolValidationError("; ".join(errors))
    return ProtocolSpec(
        protocol_id=raw["protocol_id"], status=raw["status"], source_revision=raw["source_revision"],
        result_states=tuple(raw["result_states"]), rng_streams=tuple(raw["rng"]["streams"]),
        experiments=tuple(experiments),
    )
