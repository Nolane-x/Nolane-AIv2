from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAMES = (
    "verify_exp319_contract.py",
    "exp319_stage_a_chunk.py",
    "exp319_stage_a_select.py",
    "exp319_stage_b_chunk.py",
    "exp319_stage_b_select.py",
    "exp319_materialize_heldout.py",
    "exp319_predict_heldout_shard.py",
    "exp319_seal_root.py",
    "exp319_finalize.py",
    "verify_exp319_freeze.py",
)

EXPECTED_OPTIONS = {
    "verify_exp319_contract.py": {"--protocol", "--digest-file"},
    "exp319_stage_a_chunk.py": {
        "--arm", "--root", "--learning-rate", "--chunk-index", "--run-identity",
        "--source-commit-sha", "--parent-checkpoint", "--parent-receipt",
        "--checkpoint-output", "--receipt-output", "--summary-output", "--device",
    },
    "exp319_stage_b_chunk.py": {
        "--arm", "--root", "--learning-rate", "--chunk-index", "--run-identity",
        "--source-commit-sha", "--parent-checkpoint", "--parent-receipt",
        "--checkpoint-output", "--receipt-output", "--summary-output", "--device",
    },
    "exp319_stage_a_select.py": {"--summaries", "--output"},
    "exp319_stage_b_select.py": {"--arm", "--root", "--summaries", "--output"},
    "exp319_materialize_heldout.py": {
        "--run-identity", "--source-digest", "--selection-authority", "--root", "--output",
    },
    "exp319_predict_heldout_shard.py": {
        "--manifest", "--root", "--shard-index", "--a-checkpoint", "--a-receipt",
        "--nrs-checkpoint", "--nrs-receipt", "--output", "--device",
    },
    "exp319_seal_root.py": {
        "--manifest", "--selection-authority", "--stage-a-selections", "--shards",
        "--root", "--source-digest", "--training-contract-digest",
        "--scoring-contract-digest", "--output",
    },
    "exp319_finalize.py": {"--root-evidence", "--output"},
    "verify_exp319_freeze.py": {
        "--repo-root", "--source-commit-sha", "--marker-json", "--marker-sha256",
    },
}

FORBIDDEN_OPTIONS = {
    "--threshold",
    "--effort",
    "--epochs",
    "--optimizer-steps",
    "--budget",
    "--world-count",
    "--examples",
    "--tokenizer",
    "--score-threshold",
    "--roots",
    "--batch-size",
    "--weight-decay",
    "--gradient-clip",
    "--max-generation-tokens",
    "--shard-count",
    "--worlds-per-shard",
}


def _load_script(name: str):
    path = ROOT / "scripts" / name
    assert path.exists(), f"missing EXP-319 CLI: {name}"
    spec = importlib.util.spec_from_file_location(f"exp319_cli_{name[:-3]}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _option_map(parser) -> dict[str, object]:
    options: dict[str, object] = {}
    for action in parser._actions:
        for option in action.option_strings:
            if option != "--help" and option.startswith("--"):
                options[option] = action
    return options


def test_all_exp319_cli_entrypoints_exist_and_expose_build_parser() -> None:
    for name in SCRIPT_NAMES:
        module = _load_script(name)
        assert callable(getattr(module, "build_parser", None)), name
        assert callable(getattr(module, "main", None)), name


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_cli_surfaces_are_exactly_preregistered_identity_and_io_coordinates(name: str) -> None:
    module = _load_script(name)
    options = _option_map(module.build_parser())
    assert set(options) == EXPECTED_OPTIONS[name]
    assert not (set(options) & FORBIDDEN_OPTIONS)


@pytest.mark.parametrize(
    "name",
    ("exp319_stage_a_chunk.py", "exp319_stage_b_chunk.py", "exp319_predict_heldout_shard.py"),
)
def test_execution_clis_are_cpu_only(name: str) -> None:
    module = _load_script(name)
    device_action = _option_map(module.build_parser())["--device"]
    assert tuple(device_action.choices) == ("cpu",)
    assert device_action.default == "cpu"


def test_chunk_index_is_coordinate_not_budget_override() -> None:
    for name in ("exp319_stage_a_chunk.py", "exp319_stage_b_chunk.py"):
        module = _load_script(name)
        parser = module.build_parser()
        options = _option_map(parser)
        assert "--chunk-index" in options
        assert "--optimizer-steps" not in options
        assert "--epochs" not in options
        assert "--budget" not in options
