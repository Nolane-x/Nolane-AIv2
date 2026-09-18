from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScientificTrialResult:
    """Torch-free evidence record shared by training and ceremony layers."""

    arm_id: str
    root: int
    learning_rate: float
    model_init_seed: int
    training_steps: int
    development_family_balanced_score: float
    development_family_scores: tuple[tuple[str, float], ...]
    checkpoint_digest: str
    trial_receipt_digest: str
