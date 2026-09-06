from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from nolane_ai.model.config import NLMConfig
from nolane_ai.model.nlm import NolaneLivingModel
from nolane_ai.training.curriculum import StageACurriculum
from nolane_ai.training.pilot import (
    StageAPilotTrainer,
    build_pilot_config_digest,
    build_seeded_pilot_model,
    functional_model_state_digest,
    pilot_model_init_seed,
    load_pilot_checkpoint,
    save_pilot_checkpoint,
    validate_pilot_manifest,
)



def test_seeded_pilot_model_initialization_is_reproducible_and_domain_separated():
    config = NLMConfig.stage_a_pilot_tiny_for_tests()
    a = build_seeded_pilot_model(config, root_seed="seed-a", replicate=0)
    b = build_seeded_pilot_model(config, root_seed="seed-a", replicate=0)
    c = build_seeded_pilot_model(config, root_seed="seed-b", replicate=0)
    a_state = [p.detach().clone() for n, p in a.named_parameters() if "capacity_reserve" not in n]
    b_state = [p.detach().clone() for n, p in b.named_parameters() if "capacity_reserve" not in n]
    c_state = [p.detach().clone() for n, p in c.named_parameters() if "capacity_reserve" not in n]
    assert pilot_model_init_seed("seed-a", 0) == pilot_model_init_seed("seed-a", 0)
    assert pilot_model_init_seed("seed-a", 0) != pilot_model_init_seed("seed-b", 0)
    assert all(torch.equal(x, y) for x, y in zip(a_state, b_state, strict=True))
    assert any(not torch.equal(x, z) for x, z in zip(a_state, c_state, strict=True))



def test_functional_model_digest_does_not_require_numpy(monkeypatch):
    config = NLMConfig.stage_a_pilot_tiny_for_tests()
    model = build_seeded_pilot_model(config, root_seed="digest-no-numpy")

    def _forbid_numpy(self):
        raise AssertionError("model digest must not depend on Tensor.numpy")

    monkeypatch.setattr(torch.Tensor, "numpy", _forbid_numpy, raising=True)
    digest = functional_model_state_digest(model)
    assert len(digest) == 64


def test_pilot_trainer_runs_bounded_steps_with_finite_telemetry_and_reserve_immutability():
    config = NLMConfig.stage_a_pilot_tiny_for_tests()
    curriculum = StageACurriculum(root_seed="trainer")
    init_seed = pilot_model_init_seed(curriculum.root_seed)
    model = build_seeded_pilot_model(config, root_seed=curriculum.root_seed)
    trainer = StageAPilotTrainer(
        model,
        curriculum=curriculum,
        lr=2e-3,
        weight_decay=0.0,
        protocol_digest="protocol",
        code_digest="code",
        model_init_seed=init_seed,
    )
    reserve_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if "capacity_reserve" in name
    }
    telemetry = trainer.train_steps(
        steps=3,
        start_replicate=0,
        batch_size=4,
        variables=4,
        constraints=3,
    )
    assert len(telemetry) == 3
    assert [item.step for item in telemetry] == [1, 2, 3]
    assert all(item.total_loss > 0 and item.gradient_norm >= 0 for item in telemetry)
    assert all(item.curriculum_digest for item in telemetry)
    for name, parameter in model.named_parameters():
        if "capacity_reserve" in name:
            assert parameter.grad is None
            assert torch.equal(parameter.detach(), reserve_before[name])


def test_config_digest_is_deterministic_and_distinguishes_configs():
    a = build_pilot_config_digest(NLMConfig.stage_a_pilot_16m())
    b = build_pilot_config_digest(NLMConfig.stage_a_pilot_16m())
    tiny = build_pilot_config_digest(NLMConfig.stage_a_pilot_tiny_for_tests())
    assert a == b
    assert a != tiny


def test_checkpoint_manifest_binds_tensor_file_and_provenance(tmp_path: Path):
    config = NLMConfig.stage_a_pilot_tiny_for_tests()
    curriculum = StageACurriculum(root_seed="checkpoint")
    init_seed = pilot_model_init_seed(curriculum.root_seed)
    model = build_seeded_pilot_model(config, root_seed=curriculum.root_seed)
    trainer = StageAPilotTrainer(
        model,
        curriculum=curriculum,
        lr=1e-3,
        protocol_digest="p" * 64,
        code_digest="c" * 64,
        model_init_seed=init_seed,
    )
    telemetry = trainer.train_steps(
        steps=1,
        start_replicate=5,
        batch_size=2,
        variables=3,
        constraints=2,
    )
    tensor_path = tmp_path / "pilot.pt"
    manifest_path = tmp_path / "pilot.manifest.json"
    manifest = save_pilot_checkpoint(
        trainer,
        tensor_path=tensor_path,
        manifest_path=manifest_path,
        last_telemetry=telemetry[-1],
    )
    assert tensor_path.exists() and manifest_path.exists()
    assert manifest["checkpoint_sha256"]
    assert manifest["protocol_digest"] == "p" * 64
    assert manifest["code_digest"] == "c" * 64
    assert manifest["config_digest"] == build_pilot_config_digest(config)
    assert manifest["curriculum_digest"] == telemetry[-1].curriculum_digest
    assert manifest["evidence_level"] == "EV-E2"
    assert manifest["decision"] == "UNVERIFIED"
    assert manifest["functional_parameters"] > 0
    assert manifest["reserved_parameters"] > 0
    assert manifest["reserve_reconstruction"] == "from_config_zero_fill"
    assert manifest["model_init_seed"] == init_seed
    assert manifest["initial_model_digest"] == trainer.initial_model_digest
    payload = torch.load(tensor_path, weights_only=False)
    assert payload["model_state"]
    assert not any("capacity_reserve" in name for name in payload["model_state"])
    assert validate_pilot_manifest(manifest) == []


def test_pilot_manifest_requires_model_initialization_lineage():
    manifest = {
        "schema": "NLM-STAGE-A-PILOT-CHECKPOINT-V1",
        "protocol_digest": "p",
        "code_digest": "c",
        "config_digest": "cfg",
        "curriculum_digest": "cur",
        "checkpoint_sha256": "ckpt",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "training_step": 1,
        "total_parameters": 2,
        "functional_parameters": 1,
        "reserved_parameters": 1,
        "reserve_reconstruction": "from_config_zero_fill",
    }
    errors = validate_pilot_manifest(manifest)
    assert "missing model_init_seed" in errors
    assert "missing initial_model_digest" in errors


def test_pilot_manifest_requires_reserve_reconstruction_policy():
    manifest = {
        "schema": "NLM-STAGE-A-PILOT-CHECKPOINT-V1",
        "protocol_digest": "p",
        "code_digest": "c",
        "config_digest": "cfg",
        "curriculum_digest": "cur",
        "checkpoint_sha256": "ckpt",
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "training_step": 1,
        "total_parameters": 2,
        "functional_parameters": 1,
        "reserved_parameters": 1,
    }
    errors = validate_pilot_manifest(manifest)
    assert "missing reserve_reconstruction" in errors


def test_pilot_manifest_refuses_neural_promotion_without_confirmatory_execution():
    manifest = {
        "schema": "NLM-STAGE-A-PILOT-CHECKPOINT-V1",
        "protocol_digest": "p",
        "code_digest": "c",
        "config_digest": "cfg",
        "curriculum_digest": "cur",
        "checkpoint_sha256": "ckpt",
        "evidence_level": "EV-E3",
        "decision": "PROMOTE_TO_NEXT_STAGE",
        "training_step": 1,
        "total_parameters": 1,
        "functional_parameters": 1,
        "reserved_parameters": 0,
    }
    errors = validate_pilot_manifest(manifest)
    assert "pilot checkpoint cannot claim EV-E3+" in errors
    assert "pilot checkpoint cannot promote a neural claim" in errors


def test_checkpoint_loader_reconstructs_reserve_from_config_and_restores_functional_state(tmp_path: Path):
    config = NLMConfig.stage_a_pilot_tiny_for_tests()
    curriculum = StageACurriculum(root_seed="restore")
    init_seed = pilot_model_init_seed(curriculum.root_seed)
    source_model = build_seeded_pilot_model(config, root_seed=curriculum.root_seed)
    source_trainer = StageAPilotTrainer(
        source_model,
        curriculum=curriculum,
        lr=1e-3,
        protocol_digest="p",
        code_digest="c",
        model_init_seed=init_seed,
    )
    telemetry = source_trainer.train_steps(
        steps=1, start_replicate=0, batch_size=2, variables=3, constraints=2
    )
    tensor_path = tmp_path / "pilot.pt"
    manifest_path = tmp_path / "pilot.manifest.json"
    save_pilot_checkpoint(
        source_trainer,
        tensor_path=tensor_path,
        manifest_path=manifest_path,
        last_telemetry=telemetry[-1],
    )

    restored_model = build_seeded_pilot_model(config, root_seed=curriculum.root_seed)
    restored_trainer = StageAPilotTrainer(
        restored_model,
        curriculum=StageACurriculum(root_seed="restore"),
        lr=1e-3,
        protocol_digest="p",
        code_digest="c",
        model_init_seed=init_seed,
    )
    load_pilot_checkpoint(
        restored_trainer,
        tensor_path=tensor_path,
        manifest_path=manifest_path,
    )
    source_state = {k: v for k, v in source_model.state_dict().items() if "capacity_reserve" not in k}
    restored_state = {k: v for k, v in restored_model.state_dict().items() if "capacity_reserve" not in k}
    assert source_state.keys() == restored_state.keys()
    assert all(torch.equal(source_state[k], restored_state[k]) for k in source_state)
    for name, parameter in restored_model.named_parameters():
        if "capacity_reserve" in name:
            assert torch.count_nonzero(parameter).item() == 0
    assert restored_trainer.step == source_trainer.step
