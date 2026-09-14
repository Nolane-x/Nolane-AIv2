from __future__ import annotations

import argparse
import os
from pathlib import Path

from nolane_ai.experiments.exp301_runner import load_frozen_implementation_identity
from nolane_ai.experiments.exp301_scientific import frozen_trial_plan, run_scientific_trial
from nolane_ai.experiments.exp301_scientific_executor import checkpoint_path_for_plan
from nolane_ai.experiments.exp301r_recovery import (
    FROZEN_IMPLEMENTATION_DIGEST,
    build_trial_recovery_receipt,
    resolve_trial_plan,
    write_once_json,
)


def _env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise SystemExit(f"EXP-301R requires workflow-bound {name}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one frozen EXP-301 trial for EXP-301R")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frozen-implementation-identity", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = int(_env("EXP301R_ROOT"))
    arm_id = _env("EXP301R_ARM")
    trial_index = int(_env("EXP301R_TRIAL_INDEX"))
    workflow_digest = _env("EXP301R_WORKFLOW_DIGEST")
    run_id = _env("EXP301R_RUN_ID")
    device = _env("EXP301_DEVICE")
    if device != "cpu":
        raise SystemExit("EXP-301R recovery is frozen to CPU execution")

    frozen = load_frozen_implementation_identity(args.frozen_implementation_identity)
    if frozen.frozen_implementation_digest != FROZEN_IMPLEMENTATION_DIGEST:
        raise SystemExit("EXP-301R frozen implementation digest mismatch")

    plan = resolve_trial_plan(
        frozen_trial_plan(root=root),
        root=root,
        arm_id=arm_id,
        trial_index=trial_index,
    )
    output_dir = Path(args.output_dir)
    checkpoint = checkpoint_path_for_plan(output_dir, plan)
    result = run_scientific_trial(plan, device=device, checkpoint_path=checkpoint)
    receipt = build_trial_recovery_receipt(
        plan,
        result,
        frozen_implementation_digest=frozen.frozen_implementation_digest,
        run_id=run_id,
        workflow_digest=workflow_digest,
    )
    write_once_json(checkpoint.with_name(f"{checkpoint.stem}-recovery.json"), receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
