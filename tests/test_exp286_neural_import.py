from __future__ import annotations

import pytest

pytest.importorskip("torch")


def test_exp286_neural_modules_import() -> None:
    from nolane_ai.experiments.exp286_paired_runner import run_exp286_paired_development
    from nolane_ai.experiments.matched_conflict_arms import build_matched_exp286_arm_pair

    assert callable(run_exp286_paired_development)
    assert callable(build_matched_exp286_arm_pair)
