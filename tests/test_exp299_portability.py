from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp299_native_runner import run_exp299_root


def _tiny_root():
    return run_exp299_root(
        root_seed="20260913-exp299-portability",
        canonical_index=0,
        fit_replicates=1,
        fit_start_replicate=0,
        eval_replicates=1,
        eval_start_replicate=20_000,
        d_model=8,
        hidden_size=8,
        target_parameters=12_000,
        batch_size=16,
        learning_rate=3e-4,
        weight_decay=1e-4,
        gradient_clip_norm=1.0,
        authority_threshold=0.5,
        max_exact_probes=4096,
        protocol_digest="a" * 64,
        geometry_digest="b" * 64,
        code_digest="c" * 64,
        repository_head="d" * 40,
    )


def test_root_receipt_validates_in_fresh_python_process(tmp_path):
    payload = _tiny_root()
    receipt = tmp_path / "exp299-root.json"
    receipt.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    code = """
import json
import pathlib
import sys
from nolane_ai.experiments.exp299_native_runner import validate_exp299_root
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))
errors = validate_exp299_root(payload)
if errors:
    print('\\n'.join(errors), file=sys.stderr)
    raise SystemExit(2)
print(payload['artifact_digest'])
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(receipt)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == payload["artifact_digest"]


def test_default_validation_does_not_retrain_or_depend_on_process_local_model_state(
    tmp_path,
):
    payload = _tiny_root()
    receipt = tmp_path / "exp299-root-no-reconstruct.json"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    code = """
import json
import pathlib
import sys
import nolane_ai.experiments.exp299_native_runner as runner

def forbidden(*args, **kwargs):
    raise AssertionError('validator attempted process-local reconstruction')

runner.run_exp299_root = forbidden
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))
errors = runner.validate_exp299_root(payload)
if errors:
    print('\\n'.join(errors), file=sys.stderr)
    raise SystemExit(2)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(receipt)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
