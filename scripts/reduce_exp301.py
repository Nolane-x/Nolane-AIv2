from __future__ import annotations

import argparse
from pathlib import Path

from nolane_ai.experiments.exp301_cross_root import (
    reduce_root_evidence_files,
    write_cross_root_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reduce the four frozen EXP-301 root evidence artifacts")
    parser.add_argument("--root-artifacts-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root_dir = Path(args.root_artifacts_dir)
    paths = tuple(sorted(root_dir.rglob("root-evidence.json")))
    if len(paths) != 4:
        raise SystemExit(f"EXP-301 cross-root reducer requires exactly 4 root-evidence.json files, found {len(paths)}")
    artifact = reduce_root_evidence_files(paths)
    output_dir = Path(args.output_dir)
    write_cross_root_artifact(output_dir / "cross-root-evidence.json", artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
