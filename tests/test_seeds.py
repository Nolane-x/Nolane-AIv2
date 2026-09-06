from importlib import import_module


def _seeds_module():
    module = import_module("nolane_ai.protocol.seeds")
    assert hasattr(module, "derive_stream_seed")
    assert hasattr(module, "derive_challenge_seed")
    return module


def test_stream_seed_is_deterministic_and_domain_separated():
    seeds = _seeds_module()
    a = seeds.derive_stream_seed("20260906", "EXP-277", 0, "model_init")
    b = seeds.derive_stream_seed("20260906", "EXP-277", 0, "model_init")
    c = seeds.derive_stream_seed("20260906", "EXP-277", 0, "evaluation")
    assert a == b
    assert a != c
    assert 0 <= a < 2**63


def test_challenge_seed_changes_with_future_beacon():
    seeds = _seeds_module()
    a = seeds.derive_challenge_seed("abc", "beacon-A", "EXP-277", "evaluation", 0)
    b = seeds.derive_challenge_seed("abc", "beacon-B", "EXP-277", "evaluation", 0)
    assert a != b
