from __future__ import annotations

import json

from nolane_ai.experiments.exp301r_recovery import build_challenge_manifest


class _Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _hex(ch: str) -> str:
    return ch * 64


def test_challenge_manifest_survives_json_roundtrip_without_semantic_type_drift() -> None:
    challenge = _Obj(
        root=2,
        challenge_nonce=_hex("a"),
        materialization_digest=_hex("b"),
        worlds=tuple(_Obj(content_id=f"world-{i:03d}") for i in range(512)),
    )
    manifest = build_challenge_manifest(
        challenge,
        run_id=34908691022,
        workflow_digest=_hex("c"),
        selection_manifest_digest=_hex("d"),
    )

    reloaded = json.loads(json.dumps(manifest, sort_keys=True))

    assert reloaded == manifest
