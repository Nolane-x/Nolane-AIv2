from __future__ import annotations

import importlib


def test_exp279_branch_rescue_oracle_module_exists() -> None:
    module = importlib.import_module("nolane_ai.experiments.exp279_branch_rescue_oracle")
    assert hasattr(module, "run_exp279_branch_rescue_oracle_development")
