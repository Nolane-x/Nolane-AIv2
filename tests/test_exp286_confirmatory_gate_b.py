from __future__ import annotations

import pytest

pytest.importorskip("torch")


def _executor_api():
    try:
        from nolane_ai.experiments.exp286_confirmatory_executor import (
            execute_exp286_confirmatory_challenge,
            validate_exp286_confirmatory_raw,
        )
    except ModuleNotFoundError:
        pytest.fail("EXP-286 raw confirmatory executor is missing")
    return execute_exp286_confirmatory_challenge, validate_exp286_confirmatory_raw


def test_exp286_gate_b_executor_surface_exists() -> None:
    execute, validate = _executor_api()
    assert callable(execute)
    assert callable(validate)
