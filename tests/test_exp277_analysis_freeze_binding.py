from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

_support_path = Path(__file__).with_name("test_exp277_confirmatory_authorization.py")
_support_spec = importlib.util.spec_from_file_location("exp277_authorization_test_support", _support_path)
assert _support_spec is not None and _support_spec.loader is not None
_support = importlib.util.module_from_spec(_support_spec)
_support_spec.loader.exec_module(_support)


def test_exp277_gate_a_explicitly_binds_analysis_code_identity_before_beacon() -> None:
    from nolane_ai.experiments.exp277_confirmatory_authorization import (
        _authorization_digest,
        validate_exp277_gate_a_authorization,
    )

    _, prep, _, authorization, seal = _support._build()
    expected = prep["lineage"]["analysis_code_digest"]
    assert expected == "d" * 64
    assert authorization["analysis_code_digest"] == expected
    assert seal["authorization_snapshot"]["analysis_code_digest"] == expected
    assert validate_exp277_gate_a_authorization(authorization) == []

    tampered = deepcopy(authorization)
    tampered["analysis_code_digest"] = "e" * 64
    # Re-hash only the outer content digest. The independent semantic binding
    # must still expose analysis-code drift.
    tampered["authorization_digest"] = _authorization_digest(tampered)
    errors = validate_exp277_gate_a_authorization(tampered)
    assert any("analysis" in error.lower() for error in errors)
    assert any("binding" in error.lower() for error in errors)
