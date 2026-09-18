from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .exp301_scientific import build_scientific_arm, forward_scientific_arm
from .exp301_training import Exp301ByteTokenizer, compute_answer_only_loss
from .exp319_contract import COMMON_GATING_EFFORT, canonical_json_bytes
from .exp319_metrics import (
    GenerationMetricRecord,
    answer_token_accuracy,
    summarize_generation_records,
)
from .exp319_training_base import (
    Exp319TrainingReceipt,
    load_checkpoint_bundle,
    model_state_digest,
)
from .exp319_worlds import materialize_stage_a, verify_world_answer


POSTMORTEM_SCHEMA = "EXP319D1-STAGE-A-EFFORT-PATH-POSTMORTEM-V1"
SELECTION_SCHEMA = "EXP319-STAGE-A-SELECTION-AUTHORITY-V1"
EFFORTS = (1, 2, 4, 8)
EXPECTED_STAGE = "A_SANITY"
EXPECTED_ROOT = 0
EXPECTED_STEP = 1024


@dataclass(frozen=True, slots=True)
class EffortMetric:
    effort: int
    answer_only_loss: float
    answer_token_accuracy: float
    exact_match: float
    family_balanced_exact_match: float
    family_exact: tuple[tuple[str, float], ...]
    eos_correctness: float
    invalid_output_rate: float
    nonzero_exact_families: int

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["family_exact"] = [list(item) for item in self.family_exact]
        return payload


@dataclass(frozen=True, slots=True)
class StageAEffortPostmortem:
    schema: str
    scientific_run_id: int
    selection_authority_digest: str
    source_commit_sha: str
    run_identity: str
    arm_id: str
    selected_learning_rate: float
    checkpoint_model_state_digest: str
    receipt_artifact_digest: str
    training_contract_digest: str
    model_state_digest_before: str
    model_state_digest_after: str
    gate_effort: int
    efforts: tuple[EffortMetric, ...]
    highest_exact_effort: int
    exact_match_delta_from_gate: tuple[tuple[int, float], ...]
    token_accuracy_delta_from_gate: tuple[tuple[int, float], ...]
    evidence_digest: str

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["efforts"] = [item.to_json_dict() for item in self.efforts]
        payload["exact_match_delta_from_gate"] = [
            [effort, value] for effort, value in self.exact_match_delta_from_gate
        ]
        payload["token_accuracy_delta_from_gate"] = [
            [effort, value] for effort, value in self.token_accuracy_delta_from_gate
        ]
        return payload


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _require_digest(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _finite_unit(value: float, *, label: str) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be finite and in [0,1]")
    return value


def _selection_without_digest(selection: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(selection)
    payload.pop("authority_digest", None)
    return payload


def validate_selection_binding(
    selection: Mapping[str, Any],
    receipt: Exp319TrainingReceipt,
    *,
    expected_authority_digest: str,
    expected_source_commit_sha: str,
    expected_run_identity: str,
) -> Mapping[str, Any]:
    expected_authority_digest = _require_digest(
        expected_authority_digest,
        label="expected_authority_digest",
    )
    if selection.get("schema") != SELECTION_SCHEMA:
        raise ValueError("Stage-A selection schema mismatch")
    actual_authority = selection.get("authority_digest")
    if actual_authority != expected_authority_digest:
        raise ValueError("Stage-A selection authority digest mismatch")
    if _sha256_payload(_selection_without_digest(selection)) != actual_authority:
        raise ValueError("Stage-A selection authority does not reproduce")

    if receipt.stage != EXPECTED_STAGE:
        raise ValueError("postmortem requires an A_SANITY receipt")
    if receipt.root != EXPECTED_ROOT:
        raise ValueError("postmortem requires Stage-A root 0")
    if receipt.cumulative_step != EXPECTED_STEP:
        raise ValueError("postmortem requires the selected step-1024 checkpoint")
    if receipt.source_commit_sha != expected_source_commit_sha:
        raise ValueError("checkpoint source commit mismatch")
    if receipt.run_identity != expected_run_identity:
        raise ValueError("checkpoint run identity mismatch")

    selections = selection.get("selections")
    if not isinstance(selections, list):
        raise ValueError("Stage-A selection authority lacks selections")
    matches = [
        item
        for item in selections
        if isinstance(item, Mapping) and item.get("arm_id") == receipt.arm_id
    ]
    if len(matches) != 1:
        raise ValueError("Stage-A selection does not uniquely bind checkpoint arm")
    selected = matches[0]
    if float(selected.get("selected_learning_rate")) != receipt.learning_rate:
        raise ValueError("checkpoint learning rate is not the frozen selected LR")
    record = selected.get("selected_record")
    if not isinstance(record, Mapping):
        raise ValueError("Stage-A selected record is missing")
    if int(record.get("root", -1)) != EXPECTED_ROOT:
        raise ValueError("Stage-A selected record root mismatch")
    if int(record.get("step", -1)) != EXPECTED_STEP:
        raise ValueError("Stage-A selected record step mismatch")
    if str(record.get("arm_id")) != receipt.arm_id:
        raise ValueError("Stage-A selected record arm mismatch")
    if float(record.get("learning_rate")) != receipt.learning_rate:
        raise ValueError("Stage-A selected record learning-rate mismatch")
    return selected


def _encoded_tensors(
    prompt: str,
    answer: str,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    tokenizer = Exp301ByteTokenizer()
    encoded = tokenizer.encode_example(prompt, answer)
    inputs = torch.tensor([encoded.token_ids[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor([encoded.token_ids[1:]], dtype=torch.long, device=device)
    return inputs, targets, encoded.answer_start


def _generate_at_effort(
    compiled: object,
    *,
    prompt: str,
    effort: int,
) -> tuple[str, tuple[int, ...], bool]:
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
            logits = forward_scientific_arm(compiled, tokens, effort=effort)
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
                (
                    tokens,
                    torch.tensor([[next_id]], dtype=torch.long, device=device),
                ),
                dim=1,
            )
    candidate = "" if invalid else tokenizer.decode(answer_ids)
    return candidate, tuple(generated), stopped_on_eos


def evaluate_effort(compiled: object, *, effort: int) -> EffortMetric:
    if effort not in EFFORTS:
        raise ValueError(f"effort must be one of {EFFORTS}")
    worlds = materialize_stage_a()
    model = getattr(compiled, "model")
    device = next(model.parameters()).device

    weighted_loss = 0.0
    correct_tokens = 0.0
    target_count = 0
    records: list[GenerationMetricRecord] = []

    model.eval()
    with torch.inference_mode():
        for world in worlds:
            inputs, targets, answer_start = _encoded_tensors(
                world.model_input,
                world.canonical_answer,
                device=device,
            )
            logits = forward_scientific_arm(compiled, inputs, effort=effort)
            loss = compute_answer_only_loss(
                logits,
                targets=targets,
                answer_start=answer_start,
            )
            count = targets.shape[1] - (answer_start - 1)
            weighted_loss += float(loss.item()) * count
            correct_tokens += (
                answer_token_accuracy(
                    logits,
                    targets=targets,
                    answer_start=answer_start,
                )
                * count
            )
            target_count += count

            candidate, generated, stopped = _generate_at_effort(
                compiled,
                prompt=world.model_input,
                effort=effort,
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
        raise RuntimeError("Stage-A postmortem found no answer targets")
    summary = summarize_generation_records(records)
    return EffortMetric(
        effort=effort,
        answer_only_loss=weighted_loss / target_count,
        answer_token_accuracy=correct_tokens / target_count,
        exact_match=summary.exact_match,
        family_balanced_exact_match=summary.family_balanced_exact_match,
        family_exact=summary.family_exact,
        eos_correctness=summary.eos_correctness,
        invalid_output_rate=summary.invalid_output_rate,
        nonzero_exact_families=summary.nonzero_exact_families,
    )


def seal_postmortem(
    *,
    scientific_run_id: int,
    selection_authority_digest: str,
    receipt: Exp319TrainingReceipt,
    metrics: Sequence[EffortMetric],
    model_state_digest_before: str,
    model_state_digest_after: str,
) -> StageAEffortPostmortem:
    if scientific_run_id <= 0:
        raise ValueError("scientific_run_id must be positive")
    _require_digest(selection_authority_digest, label="selection_authority_digest")
    _require_digest(receipt.artifact_digest, label="receipt.artifact_digest")
    _require_digest(receipt.training_contract_digest, label="training_contract_digest")
    _require_digest(model_state_digest_before, label="model_state_digest_before")
    _require_digest(model_state_digest_after, label="model_state_digest_after")
    if model_state_digest_before != model_state_digest_after:
        raise ValueError("postmortem evaluation mutated model state")

    ordered = tuple(sorted(metrics, key=lambda item: item.effort))
    if tuple(item.effort for item in ordered) != EFFORTS:
        raise ValueError("postmortem requires exactly efforts 1,2,4,8")
    for item in ordered:
        if not math.isfinite(item.answer_only_loss) or item.answer_only_loss < 0.0:
            raise ValueError("answer_only_loss must be finite and non-negative")
        for label, value in (
            ("answer_token_accuracy", item.answer_token_accuracy),
            ("exact_match", item.exact_match),
            ("family_balanced_exact_match", item.family_balanced_exact_match),
            ("eos_correctness", item.eos_correctness),
            ("invalid_output_rate", item.invalid_output_rate),
        ):
            _finite_unit(value, label=label)

    by_effort = {item.effort: item for item in ordered}
    gate = by_effort[COMMON_GATING_EFFORT]
    highest = max(
        ordered,
        key=lambda item: (
            item.exact_match,
            item.answer_token_accuracy,
            -item.answer_only_loss,
            -item.effort,
        ),
    )
    exact_deltas = tuple(
        (item.effort, item.exact_match - gate.exact_match) for item in ordered
    )
    token_deltas = tuple(
        (
            item.effort,
            item.answer_token_accuracy - gate.answer_token_accuracy,
        )
        for item in ordered
    )
    without_digest = {
        "schema": POSTMORTEM_SCHEMA,
        "scientific_run_id": scientific_run_id,
        "selection_authority_digest": selection_authority_digest,
        "source_commit_sha": receipt.source_commit_sha,
        "run_identity": receipt.run_identity,
        "arm_id": receipt.arm_id,
        "selected_learning_rate": receipt.learning_rate,
        "checkpoint_model_state_digest": receipt.model_state_digest,
        "receipt_artifact_digest": receipt.artifact_digest,
        "training_contract_digest": receipt.training_contract_digest,
        "model_state_digest_before": model_state_digest_before,
        "model_state_digest_after": model_state_digest_after,
        "gate_effort": COMMON_GATING_EFFORT,
        "efforts": [item.to_json_dict() for item in ordered],
        "highest_exact_effort": highest.effort,
        "exact_match_delta_from_gate": [list(item) for item in exact_deltas],
        "token_accuracy_delta_from_gate": [list(item) for item in token_deltas],
    }
    evidence_digest = _sha256_payload(without_digest)
    return StageAEffortPostmortem(
        schema=POSTMORTEM_SCHEMA,
        scientific_run_id=scientific_run_id,
        selection_authority_digest=selection_authority_digest,
        source_commit_sha=receipt.source_commit_sha,
        run_identity=receipt.run_identity,
        arm_id=receipt.arm_id,
        selected_learning_rate=receipt.learning_rate,
        checkpoint_model_state_digest=receipt.model_state_digest,
        receipt_artifact_digest=receipt.artifact_digest,
        training_contract_digest=receipt.training_contract_digest,
        model_state_digest_before=model_state_digest_before,
        model_state_digest_after=model_state_digest_after,
        gate_effort=COMMON_GATING_EFFORT,
        efforts=ordered,
        highest_exact_effort=highest.effort,
        exact_match_delta_from_gate=exact_deltas,
        token_accuracy_delta_from_gate=token_deltas,
        evidence_digest=evidence_digest,
    )


def analyze_stage_a_checkpoint(
    *,
    checkpoint_path: str | Path,
    receipt_path: str | Path,
    selection_path: str | Path,
    scientific_run_id: int,
    expected_selection_authority_digest: str,
    expected_source_commit_sha: str,
    expected_run_identity: str,
    device: str | torch.device = "cpu",
) -> StageAEffortPostmortem:
    if str(device) != "cpu":
        raise ValueError("EXP-319D1 execution is frozen to CPU")
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    if not isinstance(selection, Mapping):
        raise ValueError("Stage-A selection artifact must contain a JSON object")

    bundle = load_checkpoint_bundle(checkpoint_path, receipt_path)
    receipt = bundle.receipt
    validate_selection_binding(
        selection,
        receipt,
        expected_authority_digest=expected_selection_authority_digest,
        expected_source_commit_sha=expected_source_commit_sha,
        expected_run_identity=expected_run_identity,
    )

    torch.manual_seed(receipt.model_init_seed)
    compiled = build_scientific_arm(receipt.arm_id, device=device)
    model = getattr(compiled, "model")
    model.load_state_dict(bundle.model_state_dict, strict=True)
    before = model_state_digest(model)
    if before != receipt.model_state_digest:
        raise ValueError("loaded checkpoint model-state digest mismatch")

    metrics = tuple(evaluate_effort(compiled, effort=effort) for effort in EFFORTS)
    after = model_state_digest(model)
    return seal_postmortem(
        scientific_run_id=scientific_run_id,
        selection_authority_digest=expected_selection_authority_digest,
        receipt=receipt,
        metrics=metrics,
        model_state_digest_before=before,
        model_state_digest_after=after,
    )
