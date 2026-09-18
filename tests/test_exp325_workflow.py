from pathlib import Path

WORKFLOW=Path(".github/workflows/exp325-single-world-memorization.yml")


def text(): return WORKFLOW.read_text(encoding="utf-8")


def test_workflow_is_manual_fixed_four_family_jobs() -> None:
    t=text()
    assert "workflow_dispatch:" in t
    assert "inputs:" not in t
    for family in (
        "iterative-grid-and-maze",
        "algorithmic-sequence-transform",
        "generator-heldout-abstract-transformation",
        "language-sequence-control",
    ):
        assert family in t
    assert "Run eight independent world clones for family" in t


def test_workflow_binds_parent_and_reconstruction_artifacts() -> None:
    t=text()
    for token in (
        "35362450626","10554724648",
        "70fae1e9d8b54df17f2a46740630cbb1861662a5ec3d2df756193e3cd6991965",
        "35345351869","10547681681",
        "c0862235302243e6d9ef689ea6ef7214431ce44f41575aea12954524ad1ac621",
        "4aa03459b5266a3455bcfbc8cb070d9ceaeb0e7b8390944e7d483b0953e567c5",
        "9d8504d75f72f34ed65514f9d09ec7d26de584b9efd6a87e95fc4b2a2870e18d",
    ):
        assert token in t


def test_workflow_exposes_no_scientific_inputs_or_scale() -> None:
    t=text().lower()
    for token in ("--learning-rate","--model-size","--world-index","30000000","100000000"):
        assert token not in t
