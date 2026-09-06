import pytest

pytest.importorskip("torch")


def test_exp279_neural_modules_import() -> None:
    from nolane_ai.experiments.exp279_paired_runner import run_exp279_paired_development
    from nolane_ai.experiments.matched_routing_arms import build_matched_exp279_arm_triplet

    assert callable(run_exp279_paired_development)
    assert callable(build_matched_exp279_arm_triplet)
