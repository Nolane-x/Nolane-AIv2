from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from . import exp319_training_base as _base
from .exp319_training_base import *  # noqa: F401,F403
from .exp319_worlds import verify_world_answer


@dataclass(frozen=True, slots=True)
class SnapshotMetrics:
    step: int
    answer_only_loss: float
    answer_token_accuracy: float
    exact_match: float
    family_balanced_exact_match: float
    family_exact: tuple[tuple[str, float], ...]
    eos_correctness: float
    invalid_output_rate: float
    nonzero_exact_families: int
    gradient_norm_preclip: float
    parameter_update_norm_ratio: float
    nonfinite_events: int
    train_exact_match: float = 0.0

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["family_exact"] = [list(item) for item in self.family_exact]
        return payload


def measure_training_exact_match(compiled: object, *, stage: str, root: int) -> float:
    worlds = training_worlds(stage=stage, root=root)
    if not worlds:
        raise RuntimeError("training exact-match evaluation found no worlds")
    exact = 0
    for world in worlds:
        candidate, _, _ = _diagnostic_generate(compiled, prompt=world.model_input)
        exact += int(verify_world_answer(world, candidate))
    return exact / len(worlds)


_ORIGINAL_EVALUATE_SNAPSHOT = _base.evaluate_snapshot


def evaluate_snapshot(
    compiled: object,
    *,
    stage: str,
    root: int,
    step: int,
    gradient_norm_preclip: float,
    update_norm_ratio: float,
    nonfinite_events: int,
) -> SnapshotMetrics:
    snapshot = _ORIGINAL_EVALUATE_SNAPSHOT(
        compiled,
        stage=stage,
        root=root,
        step=step,
        gradient_norm_preclip=gradient_norm_preclip,
        update_norm_ratio=update_norm_ratio,
        nonfinite_events=nonfinite_events,
    )
    train_exact = (
        measure_training_exact_match(compiled, stage=stage, root=root)
        if stage == "B_TRAIN"
        else 0.0
    )
    return replace(snapshot, train_exact_match=train_exact)


# The immutable training implementation lives in exp319_training_base. Patch its
# runtime symbols deliberately so inherited functions such as run_training_chunk
# construct the extended snapshot schema and call the Stage-B measurement above.
_base.SnapshotMetrics = SnapshotMetrics
_base.measure_training_exact_match = measure_training_exact_match
_base.evaluate_snapshot = evaluate_snapshot
