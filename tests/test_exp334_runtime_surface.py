from pathlib import Path

def test_runtime_contains_simultaneous_subspace_projection():
    text=Path("src/nolane_ai/experiments/exp334_runtime.py").read_text()
    assert "torch.linalg.pinv" in text
    assert "TARGET_NORM_SQUARED_FLOOR" in text
    assert "rng_after_source" in text
    assert "parent_higher_order_reproduced" in text
    assert "validate_final_evidence(payload)" in text

def test_no_scale_or_persistent_state_authorization():
    text=Path("src/nolane_ai/experiments/exp334_contract.py").read_text()
    for phrase in [
        '"exp302_implementation_authorized": False',
        '"exp320_implementation_authorized": False',
        '"scale_authorized": False',
        '"authorized_30m": False',
        '"authorized_100m": False',
    ]:
        assert phrase in text
