from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp287_receipts import (
    build_exp287_root_receipt,
    receipt_sha256,
    reduce_exp287_cross_root,
    validate_exp287_root_receipt,
)
from nolane_ai.protocol.identity import source_tree_digest
from tests.test_exp287_receipts import _primitive_root


ROOT = Path(__file__).resolve().parents[1]
ROOT_CLI = ROOT / "scripts" / "run_exp287_learned_conflict_localization_dev.py"
CODE_DIGEST = "a" * 64
OTHER_CODE_DIGEST = "b" * 64


def _sealed_root(index: int, *, code_digest: str = CODE_DIGEST) -> dict:
    primitive = _primitive_root(index)
    primitive["code_digest"] = code_digest
    return build_exp287_root_receipt(primitive)


def test_root_validator_requires_code_digest_even_after_rehash() -> None:
    receipt = _sealed_root(0)
    receipt.pop("code_digest")
    receipt["artifact_digest"] = receipt_sha256(
        {key: value for key, value in receipt.items() if key != "artifact_digest"}
    )
    with pytest.raises(ValueError, match="code.*digest|digest.*code"):
        validate_exp287_root_receipt(receipt)


def test_cross_receipt_preserves_single_code_identity_and_rejects_mixed_code_roots() -> None:
    roots = [_sealed_root(index) for index in range(4)]
    cross = reduce_exp287_cross_root(roots)
    assert cross["code_digest"] == CODE_DIGEST

    mixed = [_sealed_root(index) for index in range(3)] + [
        _sealed_root(3, code_digest=OTHER_CODE_DIGEST)
    ]
    with pytest.raises(ValueError, match="code.*digest|digest.*code|identity"):
        reduce_exp287_cross_root(mixed)


def test_root_cli_injects_exact_source_tree_digest_before_sealing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location("exp287_root_cli_code_identity", ROOT_CLI)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    primitive = _primitive_root(0)
    monkeypatch.setattr(
        module,
        "run_exp287_root_development",
        lambda canonical_index, protocol_digest: primitive,
    )
    output = tmp_path / "receipt.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(ROOT_CLI), "--canonical-index", "0", "--output", str(output)],
    )

    assert module.main() == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["code_digest"] == source_tree_digest(ROOT)
