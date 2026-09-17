from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn

from .exp301_scientific import build_scientific_arm, forward_scientific_arm
from .exp301_training import Exp301ByteTokenizer, compute_answer_only_loss, effort_for_training_step
from .exp319_contract import (
    COMMON_GATING_EFFORT,
    CONTRACT,
    DIAGNOSTIC_ARMS,
    PRIMARY_ARMS,
    TRAINING_EFFORT_CYCLE,
)
from .exp319_metrics import (
    GenerationMetricRecord,
    answer_token_accuracy,
    global_gradient_norm,
    nonfinite_report,
    parameter_update_norm_ratio,
    snapshot_trainable_parameters,
    summarize_generation_records,
)
from .exp319_worlds import (
    Exp319WorldInstance,
    materialize_stage_a,
    materialize_stage_b,
    verify_world_answer,
)


RECEIPT_SCHEMA = "EXP319-TRAINING-CHAIN-RECEIPT-V1"
CHECKPOINT_SCHEMA = "EXP319-TRAINING-CHECKPOINT-V1"
TRAINING_CONTRACT_SCHEMA = "EXP319-TRAINING-CONTRACT-V1"
DATA_ORDER_SCHEMA = "EXP319-DATA-ORDER-V1"
DATA_CURSOR_SCHEMA = "EXP319-DATA-CURSOR-V1"
SUPPORTED_TRAINING_STAGES = ("A_SANITY", "B_TRAIN")


@dataclass(frozen=True, slots=True)
class Exp319TrainingReceipt:
    schema: str
    stage: str
    run_identity: str
    source_commit_sha: str
    arm_id: str
    root: int
    learning_rate: float
    model_init_seed: int
    chunk_start_step: int
    chunk_end_step: int
    cumulative_step: int
    data_order_digest: str
    data_cursor_digest: str
    model_state_digest: str
    optimizer_state_digest: str
    rng_state_digest: str
    parent_artifact_digest: str | None
    training_contract_digest: str
    snapshot_steps: tuple[int, ...]
    artifact_digest: str

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["snapshot_steps"] = list(self.snapshot_steps)
        return payload

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "Exp319TrainingReceipt":
        expected = {
            "schema",
            "stage",
            "run_identity",
            "source_commit_sha",
            "arm_id",
            "root",
            "learning_rate",
            "model_init_seed",
            "chunk_start_step",
            "chunk_end_step",
            "cumulative_step",
            "data_order_digest",
            "data_cursor_digest",
            "model_state_digest",
            "optimizer_state_digest",
            "rng_state_digest",
            "parent_artifact_digest",
            "training_contract_digest",
            "snapshot_steps",
            "artifact_digest",
        }
        if set(payload) != expected:
            missing = sorted(expected - set(payload))
            extra = sorted(set(payload) - expected)
            raise ValueError(f"training receipt field mismatch: missing={missing}, extra={extra}")
        snapshot_raw = payload["snapshot_steps"]
        if not isinstance(snapshot_raw, (list, tuple)):
            raise ValueError("snapshot_steps must be a JSON array")
        snapshot_steps = tuple(int(step) for step in snapshot_raw)
        if snapshot_steps != tuple(sorted(set(snapshot_steps))):
            raise ValueError("snapshot_steps must be unique and increasing")
        receipt = cls(
            schema=str(payload["schema"]),
            stage=str(payload["stage"]),
            run_identity=str(payload["run_identity"]),
            source_commit_sha=str(payload["source_commit_sha"]),
            arm_id=str(payload["arm_id"]),
            root=int(payload["root"]),
            learning_rate=float(payload["learning_rate"]),
            model_init_seed=int(payload["model_init_seed"]),
            chunk_start_step=int(payload["chunk_start_step"]),
            chunk_end_step=int(payload["chunk_end_step"]),
            cumulative_step=int(payload["cumulative_step"]),
            data_order_digest=str(payload["data_order_digest"]),
            data_cursor_digest=str(payload["data_cursor_digest"]),
            model_state_digest=str(payload["model_state_digest"]),
            optimizer_state_digest=str(payload["optimizer_state_digest"]),
            rng_state_digest=str(payload["rng_state_digest"]),
            parent_artifact_digest=(
                None
                if payload["parent_artifact_digest"] is None
                else str(payload["parent_artifact_digest"])
            ),
            training_contract_digest=str(payload["training_contract_digest"]),
            snapshot_steps=snapshot_steps,
            artifact_digest=str(payload["artifact_digest"]),
        )
        validate_training_receipt(receipt)
        return receipt


@dataclass(frozen=True, slots=True)
class SnapshotMetrics:
    step: int
    answer_only_loss: float
    answer_token_accuracy: float
    exact_match: float
    family_balanced_exact_match: float
    family_exact: tuple[tuple[str, float], ...]
    eos_correctness: float
    invalid_output_rate: float
    nonzero_exact_families: int
    gradient_norm_preclip: float
    parameter_update_norm_ratio: float
    nonfinite_events: int

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["family_exact"] = [list(item) for item in self.family_exact]
        return payload


@dataclass(frozen=True, slots=True)
class LoadedCheckpointBundle:
    receipt: Exp319TrainingReceipt
    model_state_dict: Mapping[str, torch.Tensor]
    optimizer_state_dict: Mapping[str, Any]
    torch_rng_state: torch.Tensor
    snapshots: tuple[Mapping[str, Any], ...] = ()
    initial_answer_only_loss: float | None = None


@dataclass(frozen=True, slots=True)
class TrainingChunkResult:
    receipt: Exp319TrainingReceipt
    snapshots: tuple[SnapshotMetrics, ...]
    initial_answer_only_loss: float | None


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _validate_hex(value: str, *, length: int, label: str) -> None:
    if len(value) != length or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{label} must be {length} lowercase hexadecimal characters")


def _stage_geometry(stage: str) -> tuple[tuple[str, ...], tuple[int, ...], tuple[float, ...], int, int, tuple[int, ...]]:
    if stage == "A_SANITY":
        return (
            tuple(DIAGNOSTIC_ARMS),
            (CONTRACT.stage_a.root,),
            tuple(CONTRACT.stage_a.learning_rates),
            CONTRACT.stage_a.max_chunk_steps,
            max(CONTRACT.stage_a.checkpoints),
            tuple(CONTRACT.stage_a.checkpoints),
        )
    if stage == "B_TRAIN":
        return (
            tuple(PRIMARY_ARMS),
            tuple(CONTRACT.stage_b.roots),
            tuple(CONTRACT.stage_a.learning_rates),
            CONTRACT.stage_b.max_chunk_steps,
            max(CONTRACT.stage_b.checkpoints),
            tuple(CONTRACT.stage_b.checkpoints),
        )
    raise ValueError(f"stage must be one of {SUPPORTED_TRAINING_STAGES}")


def training_worlds(*, stage: str, root: int) -> tuple[Exp319WorldInstance, ...]:
    if stage == "A_SANITY":
        if root != CONTRACT.stage_a.root:
            raise ValueError("A_SANITY is frozen to root=0")
        return materialize_stage_a()
    if stage == "B_TRAIN":
        if root not in CONTRACT.stage_b.roots:
            raise ValueError("B_TRAIN root must be one of (1, 2, 3, 4)")
        return materialize_stage_b(root=root, split="train")
    raise ValueError(f"stage must be one of {SUPPORTED_TRAINING_STAGES}")


def model_init_seed(*, stage: str, arm_id: str, root: int) -> int:
    arms, roots, _, _, _, _ = _stage_geometry(stage)
    if arm_id not in arms:
        raise ValueError(f"{stage} arm mismatch: {arm_id}")
    if root not in roots:
        raise ValueError(f"{stage} root mismatch: {root}")
    digest = hashlib.sha256(
        f"EXP319-MODEL-INIT-V1|{stage}|{arm_id}|root={root}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def training_contract_payload(stage: str) -> dict[str, Any]:
    arms, roots, learning_rates, max_chunk, max_step, checkpoints = _stage_geometry(stage)
    return {
        "schema": TRAINING_CONTRACT_SCHEMA,
        "stage": stage,
        "arms": list(arms),
        "roots": list(roots),
        "learning_rates": list(learning_rates),
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
        "effort_cycle": list(TRAINING_EFFORT_CYCLE),
        "tokenizer_version": Exp301ByteTokenizer.version,
        "answer_only_loss": True,
        "gating_effort": COMMON_GATING_EFFORT,
        "max_chunk_steps": max_chunk,
        "max_cumulative_step": max_step,
        "snapshot_steps": list(checkpoints),
        "data_order": "deterministic_exp319_world_order_cycled_by_global_step",
        "fresh_initialization_required": stage == "B_TRAIN",
        "stage_a_weights_may_be_reused": False,
    }


def training_contract_digest(stage: str) -> str:
    return _sha256_payload(training_contract_payload(stage))


def data_order_digest(*, stage: str, root: int) -> str:
    worlds = training_worlds(stage=stage, root=root)
    payload = {
        "schema": DATA_ORDER_SCHEMA,
        "stage": stage,
        "root": root,
        "ordering": "materialized-family-then-index; cycle-from-zero",
        "world_count": len(worlds),
        "content_ids": [world.content_id for world in worlds],
    }
    return _sha256_payload(payload)


def data_cursor_digest(*, stage: str, root: int, cumulative_step: int) -> str:
    if cumulative_step < 0:
        raise ValueError("cumulative_step must be non-negative")
    _, roots, _, _, max_step, _ = _stage_geometry(stage)
    if root not in roots:
        raise ValueError(f"{stage} root mismatch: {root}")
    if cumulative_step > max_step:
        raise ValueError(f"cumulative_step exceeds frozen {stage} ceiling {max_step}")
    worlds = training_worlds(stage=stage, root=root)
    consumed = [worlds[step % len(worlds)].content_id for step in range(cumulative_step)]
    payload = {
        "schema": DATA_CURSOR_SCHEMA,
        "stage": stage,
        "root": root,
        "cumulative_step": cumulative_step,
        "order_digest": data_order_digest(stage=stage, root=root),
        "consumed_content_ids": consumed,
    }
    return _sha256_payload(payload)


def expected_snapshot_steps(stage: str, chunk_start_step: int, chunk_end_step: int) -> tuple[int, ...]:
    _, _, _, _, max_step, checkpoints = _stage_geometry(stage)
    if chunk_start_step < 0 or chunk_end_step <= chunk_start_step:
        raise ValueError("training chunk interval must be positive and increasing")
    if chunk_end_step > max_step:
        raise ValueError(f"chunk_end_step exceeds frozen {stage} ceiling {max_step}")
    return tuple(step for step in checkpoints if chunk_start_step < step <= chunk_end_step)


def _receipt_digest_payload(receipt: Exp319TrainingReceipt) -> str:
    payload = receipt.to_json_dict()
    payload.pop("artifact_digest", None)
    return _sha256_payload(payload)


def build_training_receipt(
    *,
    stage: str,
    run_identity: str,
    source_commit_sha: str,
    arm_id: str,
    root: int,
    learning_rate: float,
    model_init_seed: int,
    chunk_start_step: int,
    chunk_end_step: int,
    model_state_digest: str,
    optimizer_state_digest: str,
    rng_state_digest: str,
    parent_artifact_digest: str | None,
) -> Exp319TrainingReceipt:
    arms, roots, learning_rates, max_chunk, max_step, _ = _stage_geometry(stage)
    if arm_id not in arms:
        raise ValueError(f"{stage} arm mismatch: {arm_id}")
    if root not in roots:
        raise ValueError(f"{stage} root mismatch: {root}")
    if learning_rate not in learning_rates:
        raise ValueError(f"{stage} learning rate is not preregistered")
    if model_init_seed != globals()["model_init_seed"](stage=stage, arm_id=arm_id, root=root):
        raise ValueError("model_init_seed does not match frozen EXP-319 initialization authority")
    if chunk_start_step < 0 or chunk_end_step <= chunk_start_step:
        raise ValueError("training chunk interval must be positive and increasing")
    chunk_steps = chunk_end_step - chunk_start_step
    if chunk_steps > max_chunk:
        raise ValueError(f"{stage} chunk cannot exceed {max_chunk} optimizer steps")
    if chunk_end_step > max_step:
        raise ValueError(f"chunk_end_step exceeds frozen {stage} ceiling {max_step}")
    if chunk_start_step == 0 and parent_artifact_digest is not None:
        raise ValueError("first training chunk cannot have a parent artifact")
    if chunk_start_step > 0 and parent_artifact_digest is None:
        raise ValueError("continuation chunk requires parent_artifact_digest")

    provisional = Exp319TrainingReceipt(
        schema=RECEIPT_SCHEMA,
        stage=stage,
        run_identity=run_identity,
        source_commit_sha=source_commit_sha,
        arm_id=arm_id,
        root=root,
        learning_rate=float(learning_rate),
        model_init_seed=model_init_seed,
        chunk_start_step=chunk_start_step,
        chunk_end_step=chunk_end_step,
        cumulative_step=chunk_end_step,
        data_order_digest=data_order_digest(stage=stage, root=root),
        data_cursor_digest=data_cursor_digest(
            stage=stage, root=root, cumulative_step=chunk_end_step
        ),
        model_state_digest=model_state_digest,
        optimizer_state_digest=optimizer_state_digest,
        rng_state_digest=rng_state_digest,
        parent_artifact_digest=parent_artifact_digest,
        training_contract_digest=training_contract_digest(stage),
        snapshot_steps=expected_snapshot_steps(stage, chunk_start_step, chunk_end_step),
        artifact_digest="",
    )
    receipt = Exp319TrainingReceipt(
        **{**asdict(provisional), "artifact_digest": _receipt_digest_payload(provisional)}
    )
    validate_training_receipt(receipt)
    return receipt


def validate_training_receipt(receipt: Exp319TrainingReceipt) -> None:
    if receipt.schema != RECEIPT_SCHEMA:
        raise ValueError("training receipt schema mismatch")
    if not receipt.run_identity:
        raise ValueError("run_identity cannot be empty")
    _validate_hex(receipt.source_commit_sha, length=40, label="source_commit_sha")
    arms, roots, learning_rates, max_chunk, max_step, _ = _stage_geometry(receipt.stage)
    if receipt.arm_id not in arms:
        raise ValueError(f"{receipt.stage} arm mismatch")
    if receipt.root not in roots:
        raise ValueError(f"{receipt.stage} root mismatch")
    if receipt.learning_rate not in learning_rates:
        raise ValueError("learning_rate is not preregistered")
    expected_seed = model_init_seed(stage=receipt.stage, arm_id=receipt.arm_id, root=receipt.root)
    if receipt.model_init_seed != expected_seed:
        raise ValueError("model_init_seed mismatch")
    if receipt.chunk_start_step < 0 or receipt.chunk_end_step <= receipt.chunk_start_step:
        raise ValueError("invalid training chunk interval")
    if receipt.chunk_end_step - receipt.chunk_start_step > max_chunk:
        raise ValueError(f"{receipt.stage} chunk cannot exceed {max_chunk} optimizer steps")
    if receipt.chunk_end_step > max_step:
        raise ValueError(f"training chunk exceeds frozen {receipt.stage} ceiling")
    if receipt.cumulative_step != receipt.chunk_end_step:
        raise ValueError("cumulative_step must equal chunk_end_step")
    if receipt.chunk_start_step == 0 and receipt.parent_artifact_digest is not None:
        raise ValueError("first training chunk cannot have a parent artifact")
    if receipt.chunk_start_step > 0 and receipt.parent_artifact_digest is None:
        raise ValueError("continuation chunk requires a parent artifact")

    for label, value in (
        ("data_order_digest", receipt.data_order_digest),
        ("data_cursor_digest", receipt.data_cursor_digest),
        ("model_state_digest", receipt.model_state_digest),
        ("optimizer_state_digest", receipt.optimizer_state_digest),
        ("rng_state_digest", receipt.rng_state_digest),
        ("training_contract_digest", receipt.training_contract_digest),
        ("artifact_digest", receipt.artifact_digest),
    ):
        _validate_hex(value, length=64, label=label)
    if receipt.parent_artifact_digest is not None:
        _validate_hex(receipt.parent_artifact_digest, length=64, label="parent_artifact_digest")

    expected_order = data_order_digest(stage=receipt.stage, root=receipt.root)
    if receipt.data_order_digest != expected_order:
        raise ValueError("data order digest mismatch")
    expected_cursor = data_cursor_digest(
        stage=receipt.stage,
        root=receipt.root,
        cumulative_step=receipt.cumulative_step,
    )
    if receipt.data_cursor_digest != expected_cursor:
        raise ValueError("data cursor digest mismatch")
    if receipt.training_contract_digest != training_contract_digest(receipt.stage):
        raise ValueError("training contract digest mismatch")
    expected_snapshots = expected_snapshot_steps(
        receipt.stage, receipt.chunk_start_step, receipt.chunk_end_step
    )
    if receipt.snapshot_steps != expected_snapshots:
        raise ValueError("snapshot_steps mismatch")
    if receipt.artifact_digest != _receipt_digest_payload(receipt):
        raise ValueError("training receipt artifact digest mismatch")


def validate_continuation(parent: Exp319TrainingReceipt, child: Exp319TrainingReceipt) -> None:
    validate_training_receipt(parent)
    validate_training_receipt(child)
    if child.stage != parent.stage:
        raise ValueError("continuation stage mismatch; Stage B cannot consume Stage A weights")
    same_fields = (
        "run_identity",
        "source_commit_sha",
        "arm_id",
        "root",
        "learning_rate",
        "model_init_seed",
        "data_order_digest",
        "training_contract_digest",
    )
    for field in same_fields:
        if getattr(child, field) != getattr(parent, field):
            raise ValueError(f"continuation {field} mismatch")
    if child.chunk_start_step != parent.chunk_end_step:
        raise ValueError("continuation has skipped or overlapping optimizer steps")
    if child.parent_artifact_digest != parent.artifact_digest:
        raise ValueError("continuation parent artifact digest mismatch")


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    materialized = tensor.detach().cpu().contiguous()
    try:
        return materialized.numpy().tobytes(order="C")
    except Exception:
        return bytes(materialized.view(torch.uint8).reshape(-1).tolist())


def _model_state_dict_digest(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        materialized = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(materialized.dtype).encode("ascii"))
        digest.update(json.dumps(list(materialized.shape), separators=(",", ":")).encode("ascii"))
        digest.update(_tensor_bytes(materialized))
    return digest.hexdigest()


def model_state_digest(model: nn.Module) -> str:
    return _model_state_dict_digest(model.state_dict())


def _update_structured_digest(digest: "hashlib._Hash", value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_canonical_json_bytes(list(tensor.shape)))
        digest.update(_tensor_bytes(tensor))
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: str(item)):
            _update_structured_digest(digest, str(key))
            _update_structured_digest(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"sequence\0")
        digest.update(str(len(value)).encode("ascii"))
        for item in value:
            _update_structured_digest(digest, item)
        return
    if value is None:
        digest.update(b"none\0")
        return
    if isinstance(value, bool):
        digest.update(b"bool\0" + (b"1" if value else b"0"))
        return
    if isinstance(value, int):
        digest.update(b"int\0" + str(value).encode("ascii"))
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("optimizer state contains non-finite scalar")
        digest.update(b"float\0" + value.hex().encode("ascii"))
        return
    if isinstance(value, str):
        digest.update(b"str\0" + value.encode("utf-8"))
        return
    raise TypeError(f"unsupported checkpoint digest value: {type(value).__name__}")


def optimizer_state_dict_digest(state_dict: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    _update_structured_digest(digest, state_dict)
    return digest.hexdigest()


def optimizer_state_digest(optimizer: torch.optim.Optimizer) -> str:
    return optimizer_state_dict_digest(optimizer.state_dict())


def rng_state_digest(state: torch.Tensor | None = None) -> str:
    materialized = torch.get_rng_state() if state is None else state
    if materialized.dtype != torch.uint8 or materialized.ndim != 1:
        raise ValueError("torch RNG state must be a one-dimensional uint8 tensor")
    digest = hashlib.sha256()
    digest.update(b"EXP319-TORCH-CPU-RNG-V1\0")
    digest.update(_tensor_bytes(materialized))
    return digest.hexdigest()


def _canonical_receipt_text(receipt: Exp319TrainingReceipt) -> str:
    return json.dumps(
        receipt.to_json_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def write_checkpoint_bundle(
    checkpoint_path: str | Path,
    receipt_path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    receipt: Exp319TrainingReceipt,
    snapshots: Sequence[SnapshotMetrics | Mapping[str, Any]] = (),
    initial_answer_only_loss: float | None = None,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    receipt_path = Path(receipt_path)
    validate_training_receipt(receipt)
    if checkpoint_path.exists():
        raise FileExistsError(checkpoint_path)
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    if model_state_digest(model) != receipt.model_state_digest:
        raise ValueError("model state digest does not match receipt")
    if optimizer_state_digest(optimizer) != receipt.optimizer_state_digest:
        raise ValueError("optimizer state digest does not match receipt")
    current_rng = torch.get_rng_state().clone()
    if rng_state_digest(current_rng) != receipt.rng_state_digest:
        raise ValueError("RNG state digest does not match receipt")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "schema": CHECKPOINT_SCHEMA,
        "receipt_artifact_digest": receipt.artifact_digest,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": current_rng,
        "snapshots": [
            item.to_json_dict() if isinstance(item, SnapshotMetrics) else dict(item)
            for item in snapshots
        ],
        "initial_answer_only_loss": initial_answer_only_loss,
    }
    try:
        with checkpoint_path.open("xb") as handle:
            torch.save(checkpoint_payload, handle)
        with receipt_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical_receipt_text(receipt))
    except Exception:
        checkpoint_path.unlink(missing_ok=True)
        receipt_path.unlink(missing_ok=True)
        raise


def load_checkpoint_bundle(
    checkpoint_path: str | Path,
    receipt_path: str | Path,
) -> LoadedCheckpointBundle:
    checkpoint_path = Path(checkpoint_path)
    receipt_path = Path(receipt_path)
    receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt_payload, dict):
        raise ValueError("training receipt must be a JSON object")
    receipt = Exp319TrainingReceipt.from_json_dict(receipt_payload)
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or raw.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("training checkpoint schema mismatch")
    if raw.get("receipt_artifact_digest") != receipt.artifact_digest:
        raise ValueError("checkpoint/receipt artifact digest mismatch")
    model_state = raw.get("model_state_dict")
    optimizer_state = raw.get("optimizer_state_dict")
    rng_state = raw.get("torch_rng_state")
    if not isinstance(model_state, Mapping):
        raise ValueError("checkpoint model state is malformed")
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("checkpoint optimizer state is malformed")
    if not isinstance(rng_state, torch.Tensor):
        raise ValueError("checkpoint RNG state is malformed")
    if _model_state_dict_digest(model_state) != receipt.model_state_digest:
        raise ValueError("model state digest mismatch")
    if optimizer_state_dict_digest(optimizer_state) != receipt.optimizer_state_digest:
        raise ValueError("optimizer state digest mismatch")
    if rng_state_digest(rng_state) != receipt.rng_state_digest:
        raise ValueError("RNG state digest mismatch")
    snapshots_raw = raw.get("snapshots", [])
    if not isinstance(snapshots_raw, list):
        raise ValueError("checkpoint snapshots must be a list")
    initial = raw.get("initial_answer_only_loss")
    if initial is not None:
        initial = float(initial)
        if not math.isfinite(initial):
            raise ValueError("initial answer-only loss must be finite")
    return LoadedCheckpointBundle(
        receipt=receipt,
        model_state_dict=model_state,
        optimizer_state_dict=optimizer_state,
        torch_rng_state=rng_state,
        snapshots=tuple(dict(item) for item in snapshots_raw),
        initial_answer_only_loss=initial,
    )


def _encoded_tensors(
    world: Exp319WorldInstance,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    tokenizer = Exp301ByteTokenizer()
    encoded = tokenizer.encode_example(world.model_input, world.canonical_answer)
    input_ids = torch.tensor([encoded.token_ids[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor([encoded.token_ids[1:]], dtype=torch.long, device=device)
    return input_ids, targets, encoded.answer_start


def _diagnostic_generate(compiled: object, *, prompt: str) -> tuple[str, tuple[int, ...], bool]:
    tokenizer = Exp301ByteTokenizer()
    prompt_ids = (tokenizer.bos_id, *tokenizer.encode_text(prompt), tokenizer.separator_id)
    model = getattr(compiled, "model")
    device = next(model.parameters()).device
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated: list[int] = []
    answer_ids: list[int] = []
    stopped_on_eos = False
    invalid = False
    model.eval()
    with torch.inference_mode():
        for _ in range(96):
            logits = forward_scientific_arm(compiled, tokens, effort=COMMON_GATING_EFFORT)
            next_id = int(torch.argmax(logits[0, -1]).item())
            generated.append(next_id)
            if next_id == tokenizer.eos_id:
                stopped_on_eos = True
                break
            if not tokenizer.byte_offset <= next_id < tokenizer.byte_offset + 256:
                invalid = True
                break
            answer_ids.append(next_id)
            tokens = torch.cat(
                (tokens, torch.tensor([[next_id]], dtype=torch.long, device=device)),
                dim=1,
            )
    candidate = "" if invalid else tokenizer.decode(answer_ids)
    return candidate, tuple(generated), stopped_on_eos


def _evaluation_worlds(*, stage: str, root: int) -> tuple[Exp319WorldInstance, ...]:
    if stage == "A_SANITY":
        return materialize_stage_a()
    if stage == "B_TRAIN":
        return materialize_stage_b(root=root, split="iid")
    raise ValueError(f"stage must be one of {SUPPORTED_TRAINING_STAGES}")


def measure_initial_answer_only_loss(compiled: object, *, stage: str, root: int) -> float:
    worlds = materialize_stage_a() if stage == "A_SANITY" else training_worlds(stage=stage, root=root)
    model = getattr(compiled, "model")
    device = next(model.parameters()).device
    weighted_loss = 0.0
    target_count = 0
    model.eval()
    with torch.inference_mode():
        for world in worlds:
            input_ids, targets, answer_start = _encoded_tensors(world, device=device)
            logits = forward_scientific_arm(compiled, input_ids, effort=COMMON_GATING_EFFORT)
            loss = compute_answer_only_loss(logits, targets=targets, answer_start=answer_start)
            count = targets.shape[1] - (answer_start - 1)
            weighted_loss += float(loss.item()) * count
            target_count += count
    if target_count <= 0:
        raise RuntimeError("initial-loss evaluation found no answer targets")
    return weighted_loss / target_count


def evaluate_snapshot(
    compiled: object,
    *,
    stage: str,
    root: int,
    step: int,
    gradient_norm_preclip: float,
    update_norm_ratio: float,
    nonfinite_events: int,
) -> SnapshotMetrics:
    worlds = _evaluation_worlds(stage=stage, root=root)
    model = getattr(compiled, "model")
    device = next(model.parameters()).device
    weighted_loss = 0.0
    correct_tokens = 0.0
    target_count = 0
    records: list[GenerationMetricRecord] = []
    model.eval()
    with torch.inference_mode():
        for world in worlds:
            input_ids, targets, answer_start = _encoded_tensors(world, device=device)
            logits = forward_scientific_arm(compiled, input_ids, effort=COMMON_GATING_EFFORT)
            loss = compute_answer_only_loss(logits, targets=targets, answer_start=answer_start)
            count = targets.shape[1] - (answer_start - 1)
            weighted_loss += float(loss.item()) * count
            correct_tokens += answer_token_accuracy(
                logits, targets=targets, answer_start=answer_start
            ) * count
            target_count += count
            candidate, generated, stopped = _diagnostic_generate(
                compiled, prompt=world.model_input
            )
            tokenizer = Exp301ByteTokenizer()
            invalid = any(
                token != tokenizer.eos_id
                and not tokenizer.byte_offset <= token < tokenizer.byte_offset + 256
                for token in generated
            )
            records.append(
                GenerationMetricRecord(
                    family=world.family,
                    exact=verify_world_answer(world, candidate),
                    stopped_on_eos=stopped,
                    invalid_output=invalid,
                )
            )
    if target_count <= 0:
        raise RuntimeError("snapshot evaluation found no answer targets")
    summary = summarize_generation_records(records)
    return SnapshotMetrics(
        step=step,
        answer_only_loss=weighted_loss / target_count,
        answer_token_accuracy=correct_tokens / target_count,
        exact_match=summary.exact_match,
        family_balanced_exact_match=summary.family_balanced_exact_match,
        family_exact=summary.family_exact,
        eos_correctness=summary.eos_correctness,
        invalid_output_rate=summary.invalid_output_rate,
        nonzero_exact_families=summary.nonzero_exact_families,
        gradient_norm_preclip=gradient_norm_preclip,
        parameter_update_norm_ratio=update_norm_ratio,
        nonfinite_events=nonfinite_events,
    )


def _train_one_step(
    compiled: object,
    world: Exp319WorldInstance,
    *,
    optimizer: torch.optim.Optimizer,
    global_step: int,
) -> tuple[float, float, float, int]:
    model = getattr(compiled, "model")
    device = next(model.parameters()).device
    input_ids, targets, answer_start = _encoded_tensors(world, device=device)
    before = snapshot_trainable_parameters(model.parameters())
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = forward_scientific_arm(
        compiled, input_ids, effort=effort_for_training_step(global_step)
    )
    loss = compute_answer_only_loss(logits, targets=targets, answer_start=answer_start)
    loss.backward()
    preclip = global_gradient_norm(model.parameters())
    pre_report = nonfinite_report(model.parameters(), loss=loss)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    after = snapshot_trainable_parameters(model.parameters())
    update_ratio = parameter_update_norm_ratio(before, after)
    post_report = nonfinite_report(model.parameters(), loss=loss)
    return (
        float(loss.detach().cpu().item()),
        preclip,
        update_ratio,
        pre_report.total + post_report.total,
    )


def run_training_chunk(
    *,
    stage: str,
    arm_id: str,
    root: int,
    learning_rate: float,
    run_identity: str,
    source_commit_sha: str,
    chunk_start_step: int,
    chunk_end_step: int,
    checkpoint_path: str | Path,
    receipt_path: str | Path,
    device: str | torch.device = "cpu",
    parent_checkpoint_path: str | Path | None = None,
    parent_receipt_path: str | Path | None = None,
) -> TrainingChunkResult:
    if str(device) != "cpu":
        raise ValueError("EXP-319 diagnostic execution is frozen to CPU")
    _, _, _, max_chunk, _, _ = _stage_geometry(stage)
    if chunk_end_step - chunk_start_step > max_chunk:
        raise ValueError(f"{stage} chunk cannot exceed {max_chunk} optimizer steps")
    seed = model_init_seed(stage=stage, arm_id=arm_id, root=root)

    parent: LoadedCheckpointBundle | None = None
    if chunk_start_step == 0:
        if parent_checkpoint_path is not None or parent_receipt_path is not None:
            raise ValueError("first chunk cannot consume parent artifacts")
        torch.manual_seed(seed)
    else:
        if parent_checkpoint_path is None or parent_receipt_path is None:
            raise ValueError("continuation requires both parent checkpoint and receipt")
        parent = load_checkpoint_bundle(parent_checkpoint_path, parent_receipt_path)
        if parent.receipt.stage != stage:
            raise ValueError("continuation stage mismatch")
        for label, actual, expected in (
            ("run_identity", parent.receipt.run_identity, run_identity),
            ("source_commit_sha", parent.receipt.source_commit_sha, source_commit_sha),
            ("arm_id", parent.receipt.arm_id, arm_id),
            ("root", parent.receipt.root, root),
            ("learning_rate", parent.receipt.learning_rate, learning_rate),
            ("chunk_start_step", parent.receipt.chunk_end_step, chunk_start_step),
        ):
            if actual != expected:
                raise ValueError(f"parent {label} mismatch")
        torch.set_rng_state(parent.torch_rng_state)

    compiled = build_scientific_arm(arm_id, device=device)
    optimizer = torch.optim.AdamW(
        getattr(compiled, "model").parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    if parent is not None:
        getattr(compiled, "model").load_state_dict(parent.model_state_dict)
        optimizer.load_state_dict(parent.optimizer_state_dict)
        torch.set_rng_state(parent.torch_rng_state)

    initial_loss = (
        measure_initial_answer_only_loss(compiled, stage=stage, root=root)
        if chunk_start_step == 0
        else parent.initial_answer_only_loss if parent is not None else None
    )
    worlds = training_worlds(stage=stage, root=root)
    snapshots: list[SnapshotMetrics] = []
    last_preclip = 0.0
    last_update_ratio = 0.0
    cumulative_nonfinite = 0
    snapshot_schedule = set(expected_snapshot_steps(stage, chunk_start_step, chunk_end_step))
    for global_step in range(chunk_start_step, chunk_end_step):
        world = worlds[global_step % len(worlds)]
        _, last_preclip, last_update_ratio, nonfinite = _train_one_step(
            compiled,
            world,
            optimizer=optimizer,
            global_step=global_step,
        )
        cumulative_nonfinite += nonfinite
        completed_step = global_step + 1
        if completed_step in snapshot_schedule:
            snapshots.append(
                evaluate_snapshot(
                    compiled,
                    stage=stage,
                    root=root,
                    step=completed_step,
                    gradient_norm_preclip=last_preclip,
                    update_norm_ratio=last_update_ratio,
                    nonfinite_events=cumulative_nonfinite,
                )
            )

    receipt = build_training_receipt(
        stage=stage,
        run_identity=run_identity,
        source_commit_sha=source_commit_sha,
        arm_id=arm_id,
        root=root,
        learning_rate=learning_rate,
        model_init_seed=seed,
        chunk_start_step=chunk_start_step,
        chunk_end_step=chunk_end_step,
        model_state_digest=model_state_digest(getattr(compiled, "model")),
        optimizer_state_digest=optimizer_state_digest(optimizer),
        rng_state_digest=rng_state_digest(),
        parent_artifact_digest=(None if parent is None else parent.receipt.artifact_digest),
    )
    if parent is not None:
        validate_continuation(parent.receipt, receipt)
    write_checkpoint_bundle(
        checkpoint_path,
        receipt_path,
        model=getattr(compiled, "model"),
        optimizer=optimizer,
        receipt=receipt,
        snapshots=snapshots,
        initial_answer_only_loss=initial_loss,
    )
    return TrainingChunkResult(
        receipt=receipt,
        snapshots=tuple(snapshots),
        initial_answer_only_loss=initial_loss,
    )
