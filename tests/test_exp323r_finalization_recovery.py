from pathlib import Path


WORKFLOW = Path(".github/workflows/exp323r-reducer-finalization-recovery.yml")


def test_recovery_is_artifact_only_and_forbids_scientific_execution() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    for token in (
        "35354410137",
        "217477a911aabbb764741e76cca38e54013e7da6",
        "471d211e15b4840d06536e9b2edbc404958612b2",
        "10551422072",
        "10551321837",
        "c71ff48acfe7de3e3b6a96b7a967e391a7746948385e707efaaf1b3598b12093",
        "eda5e2fb46ecbe6b7e1bc6c8b6a3f014bfe4c72df31fad9e96d121163b516708",
        "4cb2296f8c7a40727f5087ed6da5b01b4379f84111668a107c5b23ad25caa0d1",
        "a05cc80410e3c7281122bfee7cde343262690a2c04cf8085188d75914ff9c451",
        "34326f59ceb05030403bad187c09b29f5b103e711551b451c60680858efe2946",
        "e91840bc531f74bf7899e0ddb587c55330df4b60610e9ba29e28879e76d88ac6",
        "70ebb1bda4c91f457aae1007df0d939f2f2a144257dfebf5481ee6dc55b10cd9",
        "frozen/scripts/exp323r_reduce.py",
    ):
        assert token in text

    for forbidden in (
        "exp323r_run_arm.py",
        "exp323r_materialize_reconstruction.py",
        "verify_exp323_replay.py",
        "_train_one_step",
        "optimizer.step",
    ):
        assert forbidden not in text


def test_recovery_installs_model_dependency_only_for_frozen_reducer_import() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "python -m pip install -e './frozen[dev,model]'" in text
    assert "Reduce immutable arm evidence with unchanged frozen reducer" in text
