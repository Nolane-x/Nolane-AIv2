import pytest

from tests.test_exp277_confirmatory_prep import _execution, _protocol_exp277, _registry


def test_exp277_confirmatory_prep_requires_32_development_replicates() -> None:
    from nolane_ai.experiments.exp277_confirmatory_prep import build_exp277_confirmatory_prep

    with pytest.raises(ValueError, match="at least 32 paired pilot replicates"):
        build_exp277_confirmatory_prep(
            experiment=_protocol_exp277(),
            execution_artifact=_execution([1e-6] * 31),
            arm_registry=_registry(),
            analysis_code_digest="d" * 64,
        )
