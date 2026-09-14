from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from nolane_ai.experiments.exp301_ceremony import (
    build_arm_selection_receipt,
    build_root_selection_manifest,
)
from nolane_ai.experiments.exp301_runner import load_frozen_implementation_identity
from nolane_ai.experiments.exp301_scientific import build_scientific_arm, frozen_trial_plan
from nolane_ai.experiments.exp301_scientific_executor import (
    _load_checkpoint_payload,
    checkpoint_path_for_plan,
    write_root_selection_manifest,
)
from nolane_ai.experiments.exp301r_recovery import (
    FROZEN_ARMS,
    FROZEN_IMPLEMENTATION_DIGEST,
    build_selection_recovery_receipt,
    build_trial_recovery_receipt,
    write_once_json,
)


def _env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise SystemExit(f"EXP-301R requires workflow-bound {name}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select one frozen checkpoint per EXP-301 arm")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frozen-implementation-identity", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = int(_env("EXP301R_ROOT"))
    run_id = _env("EXP301R_RUN_ID")
    workflow_digest = _env("EXP301R_WORKFLOW_DIGEST")
    device = _env("EXP301_DEVICE")
    if device != "cpu":
        raise SystemExit("EXP-301R recovery is frozen to CPU execution")

    frozen = load_frozen_implementation_identity(args.frozen_implementation_identity)
    if frozen.frozen_implementation_digest != FROZEN_IMPLEMENTATION_DIGEST:
        raise SystemExit("EXP-301R frozen implementation digest mismatch")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    results_by_arm = {arm: [] for arm in FROZEN_ARMS}
    trial_receipt_digests = []
    result_by_coordinate = {}

    for plan in frozen_trial_plan(root=root):
        checkpoint = checkpoint_path_for_plan(input_dir, plan)
        compiled, result = _load_checkpoint_payload(
            checkpoint,
            arm_id=plan.arm_id,
            root=root,
            learning_rate=plan.learning_rate,
            device=device,
            arm_builder=build_scientific_arm,
        )
        del compiled
        if result.model_init_seed != plan.model_init_seed or result.training_steps != 512:
            raise SystemExit("frozen trial checkpoint identity mismatch")

        receipt_path = checkpoint.with_name(f"{checkpoint.stem}-recovery.json")
        raw_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected_receipt = build_trial_recovery_receipt(
            plan,
            result,
            frozen_implementation_digest=frozen.frozen_implementation_digest,
            run_id=run_id,
            workflow_digest=workflow_digest,
        )
        if raw_receipt != expected_receipt:
            raise SystemExit(f"recovery trial receipt mismatch for {plan.arm_id} trial {plan.trial_index}")

        results_by_arm[plan.arm_id].append(result)
        trial_receipt_digests.append(expected_receipt["receipt_digest"])
        result_by_coordinate[(plan.arm_id, plan.learning_rate)] = result

    arm_receipts = tuple(build_arm_selection_receipt(results_by_arm[arm]) for arm in FROZEN_ARMS)
    selection = build_root_selection_manifest(arm_receipts)
    root_dir = output_dir / f"root-{root}"
    write_root_selection_manifest(root_dir / "selection-manifest.json", selection)

    selected_checkpoint_digests = []
    for arm_receipt in selection.arm_selections:
        selected_plan = next(
            plan
            for plan in frozen_trial_plan(root=root)
            if plan.arm_id == arm_receipt.arm_id
            and plan.learning_rate == arm_receipt.selected_learning_rate
        )
        source = checkpoint_path_for_plan(input_dir, selected_plan)
        target = checkpoint_path_for_plan(output_dir, selected_plan)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        selected_checkpoint_digests.append(
            result_by_coordinate[(selected_plan.arm_id, selected_plan.learning_rate)].checkpoint_digest
        )

    receipt = build_selection_recovery_receipt(
        root=root,
        selection_manifest_digest=selection.selection_manifest_digest,
        selected_checkpoint_digests=tuple(selected_checkpoint_digests),
        trial_recovery_receipt_digests=tuple(trial_receipt_digests),
        run_id=run_id,
        workflow_digest=workflow_digest,
    )
    write_once_json(root_dir / "selection-recovery-receipt.json", receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
