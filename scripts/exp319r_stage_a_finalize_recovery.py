from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request


REPO = "Nolane-x/Nolane-AIv2"
SOURCE_RUN_ID = 35311529822
SOURCE_JOB_ID = 105499052525
SOURCE_ARTIFACT_ID = 10533822077
SOURCE_ARTIFACT_NAME = "exp319-stage-a-selection"
SOURCE_ARTIFACT_DIGEST = "sha256:ee65f07a5f63c5b7f472eba5aea5ab17a624697138c23266d515810024bfb559"
SOURCE_SELECTION_FILE_SHA256 = "4da236ea06a76df94ab6aa3e8141c6855a97f427f0b8614fc380d5c71e88e34a"
SOURCE_SELECTION_AUTHORITY_DIGEST = "39e5e846dbe87f3d5b34e438f0c8864626caa7049148fac934f2c893032e623f"
SOURCE_MARKER = "124584061616ab0864355219645ed00c1298c575"
SOURCE_COMMIT = "2002a42322c7b919c3c9dc3da7d9cb0f546d4431"
SOURCE_TREE_DIGEST = "ca4970a96b9eb9cb2a9b895683b8f623160b1d134e25e0e91ace6172c38b18a6"
SOURCE_WORKFLOW_SHA256 = "72b84f2fe32a7ded194b98620ac7e7701e0d18fdbb683aa4b3759145c268e237"
TRAINING_DIGEST = "cd2997f1899e21d3eee6f59e968bb4313db1d439258b6575d32357fe6c53a4e1"
SCORING_DIGEST = "13c6d3520330457135bd167a7f8c14806bc27ef3d445e1511c4adbeb0be4b0c1"
EXECUTION_DIGEST = "675655dfec9499ac8c49f3429771d1c5456e7ee652a3d7f309117bbc8691c9ba"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _request(url: str, *, token: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "exp319r-stage-a-finalization-recovery",
        },
    )


def _verify_frozen_inputs(root: Path) -> None:
    marker = root / "marker"
    frozen = root / "frozen"
    if not marker.is_dir() or not frozen.is_dir():
        raise SystemExit("marker/frozen checkout is missing")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(marker / "src")
    subprocess.run(
        [
            sys.executable,
            str(marker / "scripts" / "verify_exp319_freeze.py"),
            "--repo-root",
            str(marker),
            "--source-commit-sha",
            SOURCE_COMMIT,
            "--marker-json",
            str(marker / "protocols" / "v017" / "exp319_execution_identity_v1.json"),
            "--marker-sha256",
            str(marker / "protocols" / "v017" / "exp319_execution_identity_v1.sha256"),
        ],
        check=True,
        env=env,
    )

    workflow = frozen / ".github" / "workflows" / "exp319-learnability-foundation-diagnostic.yml"
    if _sha256_file(workflow) != SOURCE_WORKFLOW_SHA256:
        raise SystemExit("frozen workflow SHA-256 mismatch")


def _download_selection(root: Path, token: str) -> tuple[Path, dict[str, object]]:
    out = root / "recovery-out"
    inp = root / "recovered-input"
    out.mkdir(exist_ok=True)
    inp.mkdir(exist_ok=True)

    metadata_url = f"https://api.github.com/repos/{REPO}/actions/artifacts/{SOURCE_ARTIFACT_ID}"
    with urllib.request.urlopen(_request(metadata_url, token=token)) as response:
        metadata = json.load(response)

    if int(metadata["id"]) != SOURCE_ARTIFACT_ID:
        raise SystemExit("source artifact id mismatch")
    if metadata["name"] != SOURCE_ARTIFACT_NAME:
        raise SystemExit("source artifact name mismatch")
    if metadata.get("expired") is not False:
        raise SystemExit("source artifact is expired")
    if metadata.get("digest") != SOURCE_ARTIFACT_DIGEST:
        raise SystemExit(f"source artifact digest mismatch: {metadata.get('digest')!r}")
    workflow_run = metadata.get("workflow_run") or {}
    if workflow_run and int(workflow_run.get("id", -1)) != SOURCE_RUN_ID:
        raise SystemExit("source artifact workflow-run mismatch")

    (out / "source-artifact-metadata.json").write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    selection_path = inp / "stage-a-selection.json"
    if not selection_path.is_file():
        raise SystemExit("actions/download-artifact did not materialize stage-a-selection.json")
    if _sha256_file(selection_path) != SOURCE_SELECTION_FILE_SHA256:
        raise SystemExit("Stage-A selection file SHA-256 mismatch")
    return selection_path, metadata


def _reconstruct_and_finalize(root: Path, selection_path: Path) -> tuple[dict[str, object], dict[str, str]]:
    frozen = root / "frozen"
    sys.path.insert(0, str(frozen / "src"))
    from nolane_ai.experiments.exp319_contract import canonical_json_bytes
    from nolane_ai.experiments.exp319_evidence import StageASelectionSeal, _seal_root_evidence

    raw = json.loads(selection_path.read_text(encoding="utf-8"))
    if raw.get("schema") != "EXP319-STAGE-A-SELECTION-AUTHORITY-V1":
        raise SystemExit("unexpected Stage-A selection schema")
    if raw.get("authority_digest") != SOURCE_SELECTION_AUTHORITY_DIGEST:
        raise SystemExit("Stage-A selection authority digest mismatch")

    by_arm = {item["arm_id"]: item for item in raw["selections"]}
    if set(by_arm) != {"A_FIXED", "B_LOOP_SIMPLE", "C_NRS_CORE"}:
        raise SystemExit("unexpected Stage-A selection arm coverage")

    primary = tuple(
        StageASelectionSeal(
            arm_id=arm_id,
            passes_floor=bool(by_arm[arm_id]["passes_floor"]),
            selected_learning_rate=float(by_arm[arm_id]["selected_learning_rate"]),
            selection_digest=str(by_arm[arm_id]["selection_digest"]),
        )
        for arm_id in ("A_FIXED", "C_NRS_CORE")
    )
    if all(item.passes_floor for item in primary):
        raise SystemExit("frozen Stage A does not stop; seal-only recovery is invalid")

    sentinel = hashlib.sha256(
        canonical_json_bytes(
            {
                "schema": "EXP319-STAGE-B-NOT-RUN-V1",
                "reason": "frozen-stage-a-stop",
            }
        )
    ).hexdigest()

    out = root / "recovery-out"
    target = out / "stage-a-stop"
    target.mkdir(parents=True, exist_ok=False)
    for evidence_root in (1, 2, 3, 4):
        evidence = _seal_root_evidence(
            root=evidence_root,
            provenance_valid=True,
            source_digest=SOURCE_TREE_DIGEST,
            training_contract_digest=TRAINING_DIGEST,
            scoring_contract_digest=SCORING_DIGEST,
            stage_a_selections=primary,
            stage_b_selection_authority_digest=sentinel,
            stage_b_control_pass=False,
            stage_b_nrs_pass=False,
            stage_c_manifest_digest=None,
            stage_c_commitment_grid_digest=None,
            stage_c_scoring_digest=None,
            stage_c_control_pass=None,
            stage_c_nrs_pass=None,
        )
        (target / f"root-{evidence_root}.json").write_text(
            json.dumps(asdict(evidence), sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    source_copy = out / "source-stage-a-selection.json"
    source_copy.write_bytes(selection_path.read_bytes())

    final_path = out / "final.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(frozen / "src")
    subprocess.run(
        [
            sys.executable,
            str(frozen / "scripts" / "exp319_finalize.py"),
            "--root-evidence",
            *(str(target / f"root-{evidence_root}.json") for evidence_root in (1, 2, 3, 4)),
            "--output",
            str(final_path),
        ],
        check=True,
        env=env,
    )
    final = json.loads(final_path.read_text(encoding="utf-8"))

    unsigned_final = dict(final)
    claimed_final_digest = str(unsigned_final.pop("evidence_digest"))
    reproduced_final_digest = _sha256_bytes(_canonical(unsigned_final))
    if claimed_final_digest != reproduced_final_digest:
        raise SystemExit("final evidence digest does not reproduce")
    if final.get("roots") != [1, 2, 3, 4]:
        raise SystemExit("final roots mismatch")
    if final.get("stage_b_control_pass_roots") != 0 or final.get("stage_b_nrs_pass_roots") != 0:
        raise SystemExit("Stage B must remain absent after Stage-A stop")
    if final.get("stage_c_control_pass_roots") is not None or final.get("stage_c_nrs_pass_roots") is not None:
        raise SystemExit("Stage C must remain absent after Stage-A stop")
    if any(bool(value) for value in final.get("authorizations", {}).values()):
        raise SystemExit("finalizer emitted forbidden authorization")
    for key in (
        "exp302_implementation_authorized",
        "exp320_implementation_authorized",
        "scale_authorized",
        "authorized_30m",
        "authorized_100m",
    ):
        if final.get(key) is not False:
            raise SystemExit(f"forbidden authorization field: {key}")

    expected_primary = {
        item["arm_id"]: (
            bool(item["passes_floor"]),
            float(item["selected_learning_rate"]),
            str(item["selection_digest"]),
        )
        for item in raw["selections"]
        if item["arm_id"] in {"A_FIXED", "C_NRS_CORE"}
    }
    root_digests: dict[str, str] = {}
    for evidence_root in (1, 2, 3, 4):
        path = target / f"root-{evidence_root}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("root") != evidence_root or payload.get("provenance_valid") is not True:
            raise SystemExit(f"root-{evidence_root} provenance mismatch")
        if payload.get("source_digest") != SOURCE_TREE_DIGEST:
            raise SystemExit(f"root-{evidence_root} source digest mismatch")
        if payload.get("training_contract_digest") != TRAINING_DIGEST:
            raise SystemExit(f"root-{evidence_root} training digest mismatch")
        if payload.get("scoring_contract_digest") != SCORING_DIGEST:
            raise SystemExit(f"root-{evidence_root} scoring digest mismatch")
        if payload.get("stage_b_control_pass") is not False or payload.get("stage_b_nrs_pass") is not False:
            raise SystemExit(f"root-{evidence_root} Stage-B flags mismatch")
        for key in (
            "stage_c_manifest_digest",
            "stage_c_commitment_grid_digest",
            "stage_c_scoring_digest",
            "stage_c_control_pass",
            "stage_c_nrs_pass",
        ):
            if payload.get(key) is not None:
                raise SystemExit(f"root-{evidence_root} unexpectedly contains Stage-C evidence")
        if payload.get("stage_c_summaries") != []:
            raise SystemExit(f"root-{evidence_root} unexpectedly contains Stage-C summaries")

        actual_primary = {
            item["arm_id"]: (
                bool(item["passes_floor"]),
                float(item["selected_learning_rate"]),
                str(item["selection_digest"]),
            )
            for item in payload["stage_a_selections"]
        }
        if actual_primary != expected_primary:
            raise SystemExit(f"root-{evidence_root} Stage-A seal mismatch")

        unsigned_root = dict(payload)
        claimed_root_digest = str(unsigned_root.pop("artifact_digest"))
        if claimed_root_digest != _sha256_bytes(_canonical(unsigned_root)):
            raise SystemExit(f"root-{evidence_root} artifact digest does not reproduce")
        root_digests[str(evidence_root)] = claimed_root_digest

    return final, root_digests


def main() -> int:
    root = Path.cwd()
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise SystemExit("GH_TOKEN is required")

    _verify_frozen_inputs(root)
    selection_path, _ = _download_selection(root, token)
    final, root_digests = _reconstruct_and_finalize(root, selection_path)

    final_path = root / "recovery-out" / "final.json"
    receipt = {
        "schema": "EXP319R-STAGE-A-FINALIZATION-RECOVERY-RECEIPT-V1",
        "source_run_id": SOURCE_RUN_ID,
        "source_job_id": SOURCE_JOB_ID,
        "source_artifact_id": SOURCE_ARTIFACT_ID,
        "source_artifact_digest": SOURCE_ARTIFACT_DIGEST,
        "source_marker": SOURCE_MARKER,
        "source_commit": SOURCE_COMMIT,
        "source_tree_digest": SOURCE_TREE_DIGEST,
        "source_workflow_sha256": SOURCE_WORKFLOW_SHA256,
        "training_contract_digest": TRAINING_DIGEST,
        "scoring_contract_digest": SCORING_DIGEST,
        "execution_digest": EXECUTION_DIGEST,
        "stage_a_selection_authority_digest": SOURCE_SELECTION_AUTHORITY_DIGEST,
        "stage_a_selection_file_sha256": SOURCE_SELECTION_FILE_SHA256,
        "recovery_commit_sha": os.environ["GITHUB_SHA"],
        "recovery_run_id": int(os.environ["GITHUB_RUN_ID"]),
        "root_artifact_digests": root_digests,
        "final_json_sha256": _sha256_file(final_path),
        "final_evidence_digest": final["evidence_digest"],
        "final_decision": final["decision"],
        "no_training_rerun": True,
        "no_inference_rerun": True,
        "no_lr_selection_rerun": True,
        "no_scientific_selection_rerun": True,
        "exp302_implementation_authorized": False,
        "exp320_implementation_authorized": False,
        "scale_authorized": False,
        "authorized_30m": False,
        "authorized_100m": False,
    }
    receipt["receipt_digest"] = _sha256_bytes(_canonical(receipt))
    receipt_path = root / "recovery-out" / "recovery-receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    print(f"RECOVERY_FINAL_DECISION={receipt['final_decision']}")
    print(f"RECOVERY_FINAL_EVIDENCE_DIGEST={receipt['final_evidence_digest']}")
    print(f"RECOVERY_FINAL_JSON_SHA256={receipt['final_json_sha256']}")
    print(f"RECOVERY_RECEIPT_DIGEST={receipt['receipt_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
