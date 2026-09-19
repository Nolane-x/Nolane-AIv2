import hashlib
import json
from pathlib import Path

from nolane_ai.experiments.exp330_contract import (
    PARAMETER_GROUPS,
    PROBE_ROUNDS,
    GroupProbeRecord,
    directed_damage,
    effort_for_round,
    parameter_group_for_name,
    reduce_group_localization,
)


def _row(round_index, group, *, world0=False, world2=False):
    return GroupProbeRecord(
        round_index=round_index,
        effort=effort_for_round(round_index),
        parameter_group=group,
        gradient_dot=-1.0,
        gradient_cosine=-0.5,
        world0_gradient_norm=2.0,
        world2_gradient_norm=1.0,
        world0_update_self_delta=-0.1,
        world0_update_cross_delta_on_world2=0.1 if world0 else 0.0,
        world2_update_self_delta=-0.1,
        world2_update_cross_delta_on_world0=0.1 if world2 else 0.0,
        world0_direct_damage=world0,
        world2_direct_damage=world2,
        nonfinite_events=0,
    )


def _lattice(mark=None):
    mark = mark or {}
    return [
        _row(round_index, group, **mark.get((round_index, group), {}))
        for round_index in PROBE_ROUNDS
        for group in PARAMETER_GROUPS
    ]


def test_probe_rounds_are_exact_parent_damage_union():
    assert PROBE_ROUNDS == (1, 4, 8, 16, 24, 31)
    assert [effort_for_round(i) for i in PROBE_ROUNDS] == [2, 1, 1, 1, 1, 8]


def test_parameter_partition_routes_all_registered_structures():
    assert parameter_group_for_name("backbone.token_embedding.weight") == "embedding_output"
    assert parameter_group_for_name("backbone.layers.0.attention.qkv.weight") == "layer0"
    assert parameter_group_for_name("backbone.layers.1.feed_forward.up_proj.weight") == "layer1"
    assert parameter_group_for_name("backbone.layers.2.norm_ff.weight") == "layer2"
    assert parameter_group_for_name("backbone.final_norm.weight") == "final_norm"
    assert parameter_group_for_name("capacity_exchange.up.weight") == "capacity_exchange"


def test_directed_damage_requires_self_improvement_and_target_harm():
    assert directed_damage(self_delta=-1e-3, cross_delta=1e-3)
    assert not directed_damage(self_delta=0.0, cross_delta=1e-3)
    assert not directed_damage(self_delta=-1e-3, cross_delta=0.0)


def test_single_bidirectional_group_is_localized():
    rows = _lattice({
        (4, "layer1"): {"world0": True},
        (8, "layer1"): {"world2": True},
    })
    decision, world0, world2, both = reduce_group_localization(rows, parent_reproduced=True)
    assert decision == "SINGLE_GROUP_BIDIRECTIONAL_CROSS_DAMAGE"
    assert world0 == ("layer1",)
    assert world2 == ("layer1",)
    assert both == ("layer1",)


def test_directionally_split_groups_are_distinguished():
    rows = _lattice({
        (4, "layer0"): {"world0": True},
        (8, "capacity_exchange"): {"world2": True},
    })
    decision, _, _, both = reduce_group_localization(rows, parent_reproduced=True)
    assert decision == "DIRECTIONALLY_SPLIT_GROUP_CROSS_DAMAGE"
    assert both == ()


def test_parent_reproduction_precedes_group_claim():
    rows = _lattice({(4, "layer0"): {"world0": True, "world2": True}})
    assert reduce_group_localization(rows, parent_reproduced=False)[0] == "PARENT_REPRODUCTION_MISMATCH"


def test_reducer_fail_closes_incomplete_lattice():
    assert reduce_group_localization(_lattice()[:-1], parent_reproduced=True)[0] == "INVALID_PARAMETER_GROUP_COURT"


def test_preregistration_canonical_and_exact_file_digests_are_locked():
    from nolane_ai.experiments.exp330_contract import APPROVED_PREREGISTRATION_DIGEST

    root = Path(__file__).resolve().parents[1]
    path = root / "protocols/v017/exp330_preregistration_v1.json"
    raw = path.read_bytes()
    payload = json.loads(raw)
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == APPROVED_PREREGISTRATION_DIGEST
    checksum = (root / "protocols/v017/exp330_preregistration_v1.sha256").read_text().split()[0]
    assert hashlib.sha256(raw).hexdigest() == checksum
