from nolane_ai.reasoning.fidelity import FidelityCourt, compile_valid
from nolane_ai.reasoning.worlds import make_fidelity_pair


def test_compile_only_cannot_distinguish_semantic_trap():
    original, faithful, wrong = make_fidelity_pair(seed=21)
    assert compile_valid(faithful)
    assert compile_valid(wrong)
    court = FidelityCourt(max_exact_assignments=4096)
    assert court.accept(original, faithful)
    assert not court.accept(original, wrong)
