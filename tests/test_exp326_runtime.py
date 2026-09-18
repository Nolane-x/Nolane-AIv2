from __future__ import annotations

import pytest

pytest.importorskip("torch")

from nolane_ai.experiments.exp326_contract import EFFORT_CYCLE, GROUP_MEMBERS
from nolane_ai.experiments.exp326_runtime import group_training_schedule


@pytest.mark.parametrize("group_id", tuple(GROUP_MEMBERS))
def test_group_schedule_gives_every_world_identical_balanced_effort_trace(group_id: str) -> None:
    schedule=group_training_schedule(group_id)
    members=GROUP_MEMBERS[group_id]
    assert len(schedule)==len(members)*32
    for world_index in members:
        rows=[x for x in schedule if x[0]==world_index]
        assert [x[1] for x in rows]==list(range(32))
        efforts=[x[2] for x in rows]
        assert efforts==list(EFFORT_CYCLE)*8


def test_group_schedule_is_round_major_then_ascending_world_index() -> None:
    schedule=group_training_schedule("Q0123")
    assert schedule[:4]==(
        (0,0,1),(1,0,1),(2,0,1),(3,0,1),
    )
    assert schedule[4:8]==(
        (0,1,2),(1,1,2),(2,1,2),(3,1,2),
    )
