from nolane_ai.reasoning.belief import ExplicitBeliefState, RecurrentEvidenceState


def test_explicit_belief_is_normalized_and_updates_toward_repeated_evidence():
    belief = ExplicitBeliefState(prior=(0.5, 0.5), observation_accuracy=0.8)
    for observation in (1, 1, 1):
        belief.observe(observation)
    assert abs(sum(belief.probabilities) - 1.0) < 1e-9
    assert belief.probabilities[1] > 0.95
    assert belief.predict() == 1


def test_recurrent_baseline_exposes_bounded_hidden_state():
    state = RecurrentEvidenceState(decay=0.75)
    for observation in (1, 1, 0, 1):
        state.observe(observation)
    assert -1.0 <= state.hidden <= 1.0
    assert 0.0 <= state.probability_one <= 1.0
