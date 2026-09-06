from __future__ import annotations

import math


class ExplicitBeliefState:
    def __init__(self, *, prior: tuple[float, float] = (0.5, 0.5), observation_accuracy: float = 0.8) -> None:
        if len(prior) != 2 or any(p < 0 for p in prior) or sum(prior) <= 0:
            raise ValueError("prior must contain two non-negative masses")
        if not 0.5 < observation_accuracy < 1.0:
            raise ValueError("observation_accuracy must be in (0.5, 1)")
        total = sum(prior)
        self.probabilities = [prior[0] / total, prior[1] / total]
        self.observation_accuracy = observation_accuracy

    def observe(self, observation: int) -> None:
        if observation not in (0, 1):
            raise ValueError("binary observation required")
        a = self.observation_accuracy
        likelihood = [a if observation == state else 1.0 - a for state in (0, 1)]
        posterior = [self.probabilities[i] * likelihood[i] for i in (0, 1)]
        z = sum(posterior)
        self.probabilities = [posterior[0] / z, posterior[1] / z]

    def predict(self) -> int:
        return int(self.probabilities[1] >= self.probabilities[0])

    @property
    def probability_one(self) -> float:
        return self.probabilities[1]


class RecurrentEvidenceState:
    """A deliberately representation-agnostic scalar recurrent baseline for Stage-A plumbing.

    This is not a claim that a scalar state is the strongest learned recurrent rival. EXP-282
    remains EV-E2 until the matched neural recurrent baseline is trained and frozen.
    """

    def __init__(self, *, decay: float = 0.75) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0,1)")
        self.decay = decay
        self.hidden = 0.0

    def observe(self, observation: int) -> None:
        if observation not in (0, 1):
            raise ValueError("binary observation required")
        evidence = 1.0 if observation == 1 else -1.0
        self.hidden = math.tanh(self.decay * self.hidden + evidence)

    def predict(self) -> int:
        return int(self.hidden >= 0.0)

    @property
    def probability_one(self) -> float:
        return 1.0 / (1.0 + math.exp(-2.0 * self.hidden))
