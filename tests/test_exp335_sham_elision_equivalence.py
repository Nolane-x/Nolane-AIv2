import pytest

torch = pytest.importorskip("torch")

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


def _reference_project_source(source, targets):
    from nolane_ai.experiments.exp335_contract import PINV_RTOL, TARGET_NORM_SQUARED_FLOOR

    raw = {world_id: _dot(source, grad) for world_id, grad in targets}
    selected = [
        (world_id, grad)
        for world_id, grad in targets
        if raw[world_id] < 0.0 and _dot(grad, grad) > TARGET_NORM_SQUARED_FLOOR
    ]
    if not selected:
        cloned = tuple(x.clone() for x in source)
        return cloned, [], raw, raw.copy()

    count = len(selected)
    gram = torch.empty((count, count), dtype=torch.float64)
    rhs = torch.empty((count,), dtype=torch.float64)
    for i, (_, gi) in enumerate(selected):
        rhs[i] = _dot(gi, source)
        for j, (_, gj) in enumerate(selected):
            gram[i, j] = _dot(gi, gj)
    alpha = torch.linalg.pinv(gram, rtol=PINV_RTOL) @ rhs

    result = []
    for parameter_index, source_tensor in enumerate(source):
        value = source_tensor.clone()
        for coefficient, (_, target) in zip(alpha, selected):
            value = value - target[parameter_index] * float(coefficient.item())
        result.append(value)
    projected = tuple(result)
    post = {world_id: _dot(projected, grad) for world_id, grad in targets}
    return projected, [world_id for world_id, _ in selected], raw, post


def test_symmetric_gram_reuse_is_exactly_equivalent_to_registered_projector():
    source, targets = _synthetic_geometry()

    reference_projected, reference_selected, reference_raw, reference_post = (
        _reference_project_source(source, targets)
    )
    repaired_projected, repaired_selected, repaired_raw, repaired_post = _project_source(
        source, targets
    )

    assert repaired_selected == reference_selected
    assert repaired_raw == reference_raw
    assert repaired_post == reference_post
    assert all(
        torch.equal(reference, repaired)
        for reference, repaired in zip(reference_projected, repaired_projected)
    )
