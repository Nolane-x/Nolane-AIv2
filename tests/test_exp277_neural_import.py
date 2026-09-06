from __future__ import annotations

import pytest

pytest.importorskip("torch")


def test_exp277_neural_modules_import() -> None:
    from nolane_ai.experiments.exp277_paired_runner import run_exp277_paired_development
    from nolane_ai.experiments.matched_cbrf_arms import build_matched_exp277_arm_pair

    assert callable(run_exp277_paired_development)
    assert callable(build_matched_exp277_arm_pair)
