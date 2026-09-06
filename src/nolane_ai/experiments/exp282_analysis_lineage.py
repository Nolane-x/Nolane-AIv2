from __future__ import annotations

import math
from statistics import fmean, stdev
from typing import Any

from nolane_ai.protocol.evidence import canonical_sha256


def validate_exp282_prep_pilot_lineage(
    prep_artifact: dict[str, Any],
    paired_execution_artifact: dict[str, Any],
) -> list[str]:
    """Cross-check prep pilot claims against the exact paired development artifact they cite."""
    errors: list[str] = []
    pilot = prep_artifact.get("pilot_summary") or {}
    training = paired_execution_artifact.get("training") or {}
    evaluation = paired_execution_artifact.get("evaluation") or {}
    rows = evaluation.get("per_replicate") or []

    eval_count = int(evaluation.get("replicates", 0) or 0)
    replicate_ids = [int(row.get("replicate", -1)) for row in rows]
    effects: list[float] = []
    for row in rows:
        recurrent = float((row.get("recurrent_hidden") or {}).get("grounded_decision_accuracy", math.nan))
        explicit = float((row.get("explicit_belief") or {}).get("grounded_decision_accuracy", math.nan))
        declared = row.get("explicit_minus_recurrent_accuracy")
        if not isinstance(declared, (int, float)) or not math.isfinite(float(declared)):
            errors.append("paired execution contains non-finite pilot accuracy contrast")
            continue
        expected = explicit - recurrent
        if not math.isfinite(expected) or not math.isclose(float(declared), expected, rel_tol=0.0, abs_tol=1e-12):
            errors.append("paired execution pilot accuracy contrast mismatch")
            continue
        effects.append(float(declared))

    train_start = int(training.get("start_replicate", -1))
    train_count = int(training.get("replicates", 0) or 0)
    training_ids = list(range(train_start, train_start + train_count)) if train_start >= 0 and train_count > 0 else []

    if int(pilot.get("n", 0) or 0) != eval_count or eval_count != len(rows):
        errors.append("prep pilot n does not match paired execution")
    if pilot.get("replicate_ids") != replicate_ids:
        errors.append("prep pilot replicate IDs do not match paired execution")
    if pilot.get("training_replicate_ids") != training_ids:
        errors.append("prep pilot training IDs do not match paired execution")
    if replicate_ids:
        if pilot.get("replicate_start") != min(replicate_ids) or pilot.get("replicate_end") != max(replicate_ids):
            errors.append("prep pilot replicate range does not match paired execution")

    if len(effects) == len(rows) and effects:
        expected_mean = fmean(effects)
        expected_sd = stdev(effects) if len(effects) >= 2 else 0.0
        expected_digest = canonical_sha256({"replicate_ids": replicate_ids, "effects": effects})
        mean = pilot.get("mean_accuracy_gain")
        sd = pilot.get("paired_sd")
        if not isinstance(mean, (int, float)) or not math.isfinite(float(mean)) or not math.isclose(
            float(mean), expected_mean, rel_tol=0.0, abs_tol=1e-12
        ):
            errors.append("prep pilot mean accuracy gain does not match paired execution")
        if not isinstance(sd, (int, float)) or not math.isfinite(float(sd)) or not math.isclose(
            float(sd), expected_sd, rel_tol=0.0, abs_tol=1e-12
        ):
            errors.append("prep pilot paired SD does not match paired execution")
        if pilot.get("paired_effect_digest") != expected_digest:
            errors.append("prep pilot effect digest does not match paired execution")

    return errors
