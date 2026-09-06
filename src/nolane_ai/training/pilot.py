from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from nolane_ai.model.audit import audit_model
from nolane_ai.model.config import NLMConfig
from nolane_ai.model.nlm import NolaneLivingModel
from nolane_ai.protocol.evidence import canonical_sha256
from nolane_ai.protocol.identity import file_sha256
from nolane_ai.protocol.seeds import derive_stream_seed
from .curriculum import StageACurriculum
from .optimizer import build_functional_optimizer, functional_trainable_named_parameters
from .stage_a import stage_a_multitask_loss
from .tensor_bytes import tensor_byteorder, tensor_raw_bytes


@dataclass(frozen=True, slots=True)
class PilotStepTelemetry:
    step: int
    replicate: int
    total_loss: float
    belief_loss: float
    conflict_loss: float
    fidelity_loss: float
    gradient_norm: float
    curriculum_digest: str


def _config_payload(config: NLMConfig) -> dict[str, Any]:
    return {
        "vocab_size": config.vocab_size,
        "d_model": config.d_model,
        "budget_version": config.budget.version,
        "regions": [
            {
                "name": region.name,
                "parameters": region.parameters,
                "frozen": region.frozen,
            }
            for region in config.budget.regions
        ],
        "active_region_names": list(config.active_region_names),
    }


def build_pilot_config_digest(config: NLMConfig) -> str:
    return canonical_sha256(_config_payload(config))


def functional_model_state_digest(model: NolaneLivingModel) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if "capacity_reserve" in name:
            continue
        cpu = tensor.detach().cpu().contiguous()
        header = json.dumps(
            {"name": name, "shape": list(cpu.shape), "dtype": str(cpu.dtype), "byteorder": tensor_byteorder()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(4, "big"))
        digest.update(header)
        digest.update(tensor_raw_bytes(cpu))
    return digest.hexdigest()


def pilot_model_init_seed(root_seed: str, replicate: int = 0) -> int:
    if not root_seed:
        raise ValueError("root_seed must be non-empty")
    if replicate < 0:
        raise ValueError("replicate must be non-negative")
    return derive_stream_seed(root_seed, "EXP-277", replicate, "model_init")


def build_seeded_pilot_model(
    config: NLMConfig,
    *,
    root_seed: str,
    replicate: int = 0,
    device: str | torch.device | None = None,
) -> NolaneLivingModel:
    seed = pilot_model_init_seed(root_seed, replicate)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return NolaneLivingModel(config, device=device)


class StageAPilotTrainer:
    def __init__(
        self,
        model: NolaneLivingModel,
        *,
        curriculum: StageACurriculum,
        lr: float,
        weight_decay: float = 0.01,
        protocol_digest: str,
        code_digest: str,
        model_init_seed: int,
    ) -> None:
        if not protocol_digest or not code_digest:
            raise ValueError("protocol_digest and code_digest are required")
        if model_init_seed < 0:
            raise ValueError("model_init_seed must be non-negative")
        self.model = model
        self.curriculum = curriculum
        self.protocol_digest = protocol_digest
        self.code_digest = code_digest
        self.model_init_seed = model_init_seed
        self.initial_model_digest = functional_model_state_digest(model)
        self.config_digest = build_pilot_config_digest(model.config)
        self.optimizer = build_functional_optimizer(
            model,
            lr=lr,
            weight_decay=weight_decay,
        )
        self.step = 0

    def _gradient_norm(self) -> float:
        squared = 0.0
        for _, parameter in functional_trainable_named_parameters(self.model):
            if parameter.grad is None:
                continue
            value = float(parameter.grad.detach().norm(2).item())
            squared += value * value
        return math.sqrt(squared)

    def train_steps(
        self,
        *,
        steps: int,
        start_replicate: int,
        batch_size: int,
        variables: int,
        constraints: int,
    ) -> list[PilotStepTelemetry]:
        if steps <= 0:
            raise ValueError("steps must be positive")
        if start_replicate < 0:
            raise ValueError("start_replicate must be non-negative")
        telemetry: list[PilotStepTelemetry] = []
        device = next(self.model.parameters()).device
        self.model.train()
        for offset in range(steps):
            replicate = start_replicate + offset
            curriculum_batch = self.curriculum.make_batch(
                replicate=replicate,
                batch_size=batch_size,
                variables=variables,
                constraints=constraints,
                d_model=self.model.config.d_model,
                device=device,
            )
            self.optimizer.zero_grad(set_to_none=True)
            losses = stage_a_multitask_loss(self.model, curriculum_batch.batch)
            losses["total"].backward()
            gradient_norm = self._gradient_norm()
            self.optimizer.step()
            self.step += 1
            telemetry.append(
                PilotStepTelemetry(
                    step=self.step,
                    replicate=replicate,
                    total_loss=float(losses["total"].detach().item()),
                    belief_loss=float(losses["belief"].detach().item()),
                    conflict_loss=float(losses["conflict"].detach().item()),
                    fidelity_loss=float(losses["fidelity"].detach().item()),
                    gradient_norm=gradient_norm,
                    curriculum_digest=curriculum_batch.digest,
                )
            )
        return telemetry


def validate_pilot_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "schema",
        "protocol_digest",
        "code_digest",
        "config_digest",
        "curriculum_digest",
        "checkpoint_sha256",
        "evidence_level",
        "decision",
        "training_step",
        "total_parameters",
        "functional_parameters",
        "reserved_parameters",
        "reserve_reconstruction",
        "model_init_seed",
        "initial_model_digest",
    )
    for field in required:
        if field not in manifest or manifest[field] in (None, ""):
            errors.append(f"missing {field}")
    if manifest.get("schema") != "NLM-STAGE-A-PILOT-CHECKPOINT-V1":
        errors.append("invalid pilot checkpoint schema")
    if manifest.get("evidence_level") != "EV-E2":
        errors.append("pilot checkpoint cannot claim EV-E3+")
    if manifest.get("decision") != "UNVERIFIED":
        errors.append("pilot checkpoint cannot promote a neural claim")
    if manifest.get("reserve_reconstruction") not in (None, "from_config_zero_fill"):
        errors.append("unsupported reserve reconstruction policy")
    if int(manifest.get("training_step", 0) or 0) <= 0:
        errors.append("training_step must be positive")
    total = int(manifest.get("total_parameters", 0) or 0)
    functional = int(manifest.get("functional_parameters", 0) or 0)
    reserved = int(manifest.get("reserved_parameters", 0) or 0)
    if total > 0 and total != functional + reserved:
        errors.append("parameter accounting mismatch")
    return errors


def save_pilot_checkpoint(
    trainer: StageAPilotTrainer,
    *,
    tensor_path: str | Path,
    manifest_path: str | Path,
    last_telemetry: PilotStepTelemetry,
) -> dict[str, Any]:
    tensor_path = Path(tensor_path)
    manifest_path = Path(manifest_path)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    functional_state = {
        name: tensor
        for name, tensor in trainer.model.state_dict().items()
        if "capacity_reserve" not in name
    }
    payload = {
        "schema": "NLM-STAGE-A-PILOT-TENSORS-V1",
        "training_step": trainer.step,
        "config_digest": trainer.config_digest,
        "reserve_reconstruction": "from_config_zero_fill",
        "model_init_seed": trainer.model_init_seed,
        "initial_model_digest": trainer.initial_model_digest,
        "model_state": functional_state,
        "optimizer_state": trainer.optimizer.state_dict(),
    }
    torch.save(payload, tensor_path)
    checkpoint_sha256 = file_sha256(tensor_path)
    audit = audit_model(trainer.model)
    optimizable_parameters = sum(
        parameter.numel()
        for _, parameter in functional_trainable_named_parameters(trainer.model)
    )
    manifest: dict[str, Any] = {
        "schema": "NLM-STAGE-A-PILOT-CHECKPOINT-V1",
        "protocol_digest": trainer.protocol_digest,
        "code_digest": trainer.code_digest,
        "config_digest": trainer.config_digest,
        "curriculum_digest": last_telemetry.curriculum_digest,
        "checkpoint_sha256": checkpoint_sha256,
        "evidence_level": "EV-E2",
        "decision": "UNVERIFIED",
        "training_step": trainer.step,
        "root_seed": trainer.curriculum.root_seed,
        "last_replicate": last_telemetry.replicate,
        "total_parameters": audit.total_parameters,
        "functional_parameters": audit.functional_parameters,
        "reserved_parameters": audit.reserved_parameters,
        "reserve_reconstruction": "from_config_zero_fill",
        "model_init_seed": trainer.model_init_seed,
        "initial_model_digest": trainer.initial_model_digest,
        "functional_optimizable_parameters": optimizable_parameters,
        "optimizer": {
            "type": type(trainer.optimizer).__name__,
            "param_groups": [
                {
                    key: value
                    for key, value in group.items()
                    if key != "params" and isinstance(value, (int, float, bool, str))
                }
                for group in trainer.optimizer.param_groups
            ],
        },
        "last_telemetry": asdict(last_telemetry),
        "scope": "synthetic-stage-a-development-training-smoke",
    }
    errors = validate_pilot_manifest(manifest)
    if errors:
        raise ValueError("invalid pilot checkpoint manifest: " + "; ".join(errors))
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_pilot_checkpoint(
    trainer: StageAPilotTrainer,
    *,
    tensor_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    tensor_path = Path(tensor_path)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = validate_pilot_manifest(manifest)
    if errors:
        raise ValueError("invalid pilot checkpoint manifest: " + "; ".join(errors))
    if manifest["checkpoint_sha256"] != file_sha256(tensor_path):
        raise ValueError("checkpoint tensor digest mismatch")
    if manifest["config_digest"] != trainer.config_digest:
        raise ValueError("checkpoint config digest does not match trainer model")
    if manifest["protocol_digest"] != trainer.protocol_digest:
        raise ValueError("checkpoint protocol digest does not match trainer")
    if manifest["code_digest"] != trainer.code_digest:
        raise ValueError("checkpoint code digest does not match trainer")
    if int(manifest["model_init_seed"]) != trainer.model_init_seed:
        raise ValueError("checkpoint model-init seed does not match trainer")
    if manifest["initial_model_digest"] != trainer.initial_model_digest:
        raise ValueError("checkpoint initial-model digest does not match trainer")

    payload = torch.load(tensor_path, map_location=next(trainer.model.parameters()).device, weights_only=False)
    if payload.get("schema") != "NLM-STAGE-A-PILOT-TENSORS-V1":
        raise ValueError("invalid pilot tensor checkpoint schema")
    if payload.get("config_digest") != trainer.config_digest:
        raise ValueError("tensor checkpoint config digest mismatch")
    if payload.get("reserve_reconstruction") != "from_config_zero_fill":
        raise ValueError("unsupported reserve reconstruction policy")
    if int(payload.get("model_init_seed", -1)) != trainer.model_init_seed:
        raise ValueError("tensor checkpoint model-init seed mismatch")
    if payload.get("initial_model_digest") != trainer.initial_model_digest:
        raise ValueError("tensor checkpoint initial-model digest mismatch")

    incompatible = trainer.model.load_state_dict(payload["model_state"], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"unexpected checkpoint model keys: {incompatible.unexpected_keys}")
    if any("capacity_reserve" not in key for key in incompatible.missing_keys):
        raise ValueError(f"missing non-reserve model keys: {incompatible.missing_keys}")
    with torch.no_grad():
        for name, parameter in trainer.model.named_parameters():
            if "capacity_reserve" in name:
                parameter.zero_()
    trainer.optimizer.load_state_dict(payload["optimizer_state"])
    trainer.step = int(payload["training_step"])
    return manifest
