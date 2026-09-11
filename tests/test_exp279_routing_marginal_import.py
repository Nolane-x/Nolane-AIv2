from __future__ import annotations

import importlib


def test_exp279_routing_marginal_module_exists() -> None:
    module = importlib.import_module("nolane_ai.experiments.exp279_routing_marginal")
    assert hasattr(module, "run_exp279_routing_marginal_development")
