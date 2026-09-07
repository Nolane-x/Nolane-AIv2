from __future__ import annotations

import pytest

pytest.importorskip("torch")


def test_exp289_neural_modules_import() -> None:
    from nolane_ai.experiments.exp289_paired_runner import run_exp289_paired_development
    from nolane_ai.experiments.matched_nogood_arms import build_matched_exp289_arm_pair

    assert callable(run_exp289_paired_development)
    assert callable(build_matched_exp289_arm_pair)
