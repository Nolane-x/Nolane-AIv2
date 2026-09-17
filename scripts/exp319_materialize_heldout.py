from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from nolane_ai.experiments.exp319_challenge import StageBSelectionSeal, materialize_stage_c_manifest, seal_stage_b_selection_authority


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize post-selection EXP-319 Stage C heldout manifest")
    parser.add_argument("--run-identity", required=True)
    parser.add_argument("--source-digest", required=True)
    parser.add_argument("--selection-authority", required=True)
    parser.add_argument("--root", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser


def _seal_from_payload(payload: dict) -> StageBSelectionSeal:
    raw = payload.get("seal", payload)
    return StageBSelectionSeal(
        arm_id=str(raw["arm_id"]),
        root=int(raw["root"]),
        selected_step=int(raw["selected_step"]),
        passes_floor=bool(raw["passes_floor"]),
        checkpoint_digest=str(raw["checkpoint_digest"]),
        selection_digest=str(raw["selection_digest"]),
    )


def _load_authority(path: str):
    target = Path(path)
    payloads = []
    if target.is_dir():
        payloads = [json.loads(item.read_text(encoding="utf-8")) for item in sorted(target.glob("*.json"))]
    else:
        data = json.loads(target.read_text(encoding="utf-8"))
        if data.get("schema") == "EXP319-STAGE-B-SELECTION-AUTHORITY-V1" and "selections" in data:
            payloads = list(data["selections"])
        else:
            payloads = [data]
    return seal_stage_b_selection_authority(_seal_from_payload(item) for item in payloads)


def _write_once(path: str, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    authority = _load_authority(args.selection_authority)
    beacon = args.run_identity if args.run_identity.endswith("-exp319-v1") else f"{args.run_identity}-exp319-v1"
    manifest = materialize_stage_c_manifest(
        root=args.root,
        challenge_beacon=beacon,
        source_digest=args.source_digest,
        selection_authority=authority,
    )
    _write_once(args.output, asdict(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
