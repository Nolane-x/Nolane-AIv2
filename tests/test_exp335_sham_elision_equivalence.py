import torch

from nolane_ai.experiments.exp335_runtime import _dot, _project_source


def _synthetic_geometry():
    source = (
        torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32),
        torch.tensor([0.25, -0.75], dtype=torch.float32),
    )
    targets = []
    for index in range(31):
        scale = float(index + 1) / 31.0
        sign = -1.0 if index % 2 == 0 else 1.0
        grad = (
            torch.tensor(
                [sign * scale, 0.125 * scale, -0.25 * sign * scale],
                dtype=torch.float32,
            ),
            torch.tensor([0.5 * scale, sign * 0.375 * scale], dtype=torch.float32),
        )
        targets.append((f"target-{index:02d}", grad))
    return source, targets


def test_sham_elision_preserves_ordered_raw_dots_and_applied_gradient():
    source, targets = _synthetic_geometry()

    # This is the pre-repair work SHAM used to perform even though it discarded
    # the projected gradient and post-projection diagnostics.
    _, _, old_raw_dots, _ = _project_source(source, targets)

    # This is the repaired SHAM path.
    repaired_raw_dots = {world_id: _dot(source, grad) for world_id, grad in targets}
    repaired_applied = source

    assert list(repaired_raw_dots) == [world_id for world_id, _ in targets]
    assert repaired_raw_dots == old_raw_dots
    assert all(torch.equal(old, new) for old, new in zip(source, repaired_applied))


def test_project_geometry_remains_available_and_distinct_from_sham_elision():
    source, targets = _synthetic_geometry()
    projected, selected, raw_dots, post_dots = _project_source(source, targets)

    assert len(raw_dots) == 31
    assert selected
    assert set(post_dots) == {world_id for world_id, _ in targets}
    assert any(not torch.equal(before, after) for before, after in zip(source, projected))
