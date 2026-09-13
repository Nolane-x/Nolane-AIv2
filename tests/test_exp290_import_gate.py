from __future__ import annotations

import importlib
import importlib.util


def test_exp290_geometry_module_exists() -> None:
    assert importlib.util.find_spec("nolane_ai.experiments.exp290_transfer_geometry") is not None


def test_exp290_world_module_exists() -> None:
    assert importlib.util.find_spec("nolane_ai.experiments.exp290_transfer_worlds") is not None


def test_exp290_correspondence_module_exists() -> None:
    assert importlib.util.find_spec("nolane_ai.experiments.exp290_correspondence") is not None


def test_exp290_runner_module_exists() -> None:
    assert importlib.util.find_spec("nolane_ai.experiments.exp290_learned_clause_transfer") is not None


def test_exp290_root_runner_entrypoint_exists() -> None:
    runner = importlib.import_module("nolane_ai.experiments.exp290_learned_clause_transfer")
    assert hasattr(runner, "_run_exp290_root")
