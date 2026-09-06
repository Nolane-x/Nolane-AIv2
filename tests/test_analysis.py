from nolane_ai.experiments.analysis import build_smoke_evidence_packet, summarize_bundle
from nolane_ai.experiments.harness import run_stage_a_smoke
from nolane_ai.protocol.evidence import validate_evidence_packet


def test_analyzer_produces_one_summary_per_gate_with_ci_and_observed_effect():
    bundle = run_stage_a_smoke(replicates=6, root_seed="analysis")
    summaries = summarize_bundle(bundle, bootstrap_samples=200)
    assert set(summaries) == set(bundle.experiments)
    for summary in summaries.values():
        assert summary.metric
        assert summary.baseline_arm
        assert summary.candidate_arm
        assert summary.ci_low <= summary.observed_effect <= summary.ci_high
        assert summary.replicates == 6


def test_smoke_evidence_packet_is_valid_but_cannot_promote_neural_claims():
    bundle = run_stage_a_smoke(replicates=3, root_seed="packet")
    packet = build_smoke_evidence_packet(
        bundle,
        protocol_digest="abc123",
        code_digest="def456",
    )
    assert packet["evidence_level"] == "EV-E2"
    assert packet["decision"] == "UNVERIFIED"
    assert packet["raw_per_replicate_metrics"] == bundle.raw_per_replicate_metrics
    assert validate_evidence_packet(packet) == []
