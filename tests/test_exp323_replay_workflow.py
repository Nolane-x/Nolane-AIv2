from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(".github/workflows/v017-exp323-replay-authority.yml")


def _text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_replay_authority_court_is_pr_only_and_zero_post2048() -> None:
    text = _text()
    assert "pull_request:" in text
    assert "workflow_dispatch:" not in text
    assert "post_2048_optimizer_steps" in text
    assert '"post_2048_optimizer_steps": 0' in text
    assert "Replay only through exact step 2048" in text


def test_replay_court_binds_authoritative_artifacts_and_digests() -> None:
    text = _text()
    for token in (
        "35339004168",
        "10544985252",
        "18eb23aa09086aea89bf9ab96e07c52270069b75e8dc276f7610148b46b2e4cc",
        "35311529822",
        "10534351390",
        "ee07280fee8379f39ccea16d38f1ff31482975cb39780a75a0592916e8d154c3",
        "4d3b848d193e6e465473dab7346edaafa3ee0ffc870eac600c6c46952ca880fc",
        "9b32588f0bc63881aa974df54829c4cda1d9e4913aa0fcc40cff5d5f2cddeb79",
        "e293380e9776eb9e373bd0c3d3ab6a8aa4d0ab3b6764dc582bc859efc12fca07",
    ):
        assert token in text


def test_replay_court_cannot_select_new_scientific_controls() -> None:
    text = _text().lower()
    for token in (
        "--arm",
        "--learning-rate",
        "--model-size",
        "--checkpoint-step",
        "30000000",
        "100000000",
    ):
        assert token not in text
