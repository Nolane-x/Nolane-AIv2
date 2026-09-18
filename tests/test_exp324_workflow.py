from pathlib import Path


WORKFLOW=Path(".github/workflows/exp324-family-isolated-learnability.yml")


def text()->str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_workflow_is_manual_fixed_four_family_geometry() -> None:
    t=text()
    assert "workflow_dispatch:" in t
    assert "inputs:" not in t
    for family in (
        "algorithmic-sequence-transform",
        "generator-heldout-abstract-transformation",
        "iterative-grid-and-maze",
        "language-sequence-control",
    ):
        assert family in t
    assert "fail-fast: false" in t


def test_workflow_binds_recovered_parent_and_reconstruction() -> None:
    t=text()
    for token in (
        "35356470634",
        "10552276255",
        "7155d72ab05b57f8b66284f83ed41e63057824a3288adcd27e2d5429f3e0a19c",
        "35345351869",
        "10547681681",
        "c0862235302243e6d9ef689ea6ef7214431ce44f41575aea12954524ad1ac621",
        "4aa03459b5266a3455bcfbc8cb070d9ceaeb0e7b8390944e7d483b0953e567c5",
    ):
        assert token in t


def test_workflow_has_no_scale_or_user_selected_scientific_controls() -> None:
    t=text().lower()
    for forbidden in (
        "--learning-rate",
        "--model-size",
        "--checkpoint-step",
        "target_parameters: 30000000",
        "target_parameters: 100000000",
        "c_nrs_core",
    ):
        assert forbidden not in t


def test_reducer_waits_for_all_four_family_artifacts() -> None:
    t=text()
    assert "needs: family" in t
    assert "exp324-algorithmic-sequence-transform-" in t
    assert "exp324-generator-heldout-abstract-transformation-" in t
    assert "exp324-iterative-grid-and-maze-" in t
    assert "exp324-language-sequence-control-" in t
    assert "exp324-family-isolated-final-" in t
