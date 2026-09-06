from nolane_ai.experiments.harness import FIRST_STAGE_A_GATES, run_stage_a_smoke


def test_stage_a_smoke_runs_all_six_frozen_gates_and_stays_ev_e2():
    bundle = run_stage_a_smoke(replicates=3, root_seed="20260906")
    assert tuple(bundle.experiments) == FIRST_STAGE_A_GATES
    assert bundle.evidence_level == "EV-E2"
    assert bundle.decision == "UNVERIFIED"
    assert len(bundle.raw_per_replicate_metrics) >= 3 * len(FIRST_STAGE_A_GATES) * 2
    assert all(row["experiment_id"].startswith("EXP-") for row in bundle.raw_per_replicate_metrics)


def test_stage_a_smoke_is_deterministic_for_same_seed():
    a = run_stage_a_smoke(replicates=2, root_seed="same")
    b = run_stage_a_smoke(replicates=2, root_seed="same")
    assert a.raw_per_replicate_metrics == b.raw_per_replicate_metrics


def test_smoke_arms_are_explicitly_labeled_as_proxies_for_protocol_arms():
    bundle = run_stage_a_smoke(replicates=1, root_seed="proxy-boundary")
    for row in bundle.raw_per_replicate_metrics:
        assert row["implementation_tier"] == "EV-E2_ALGORITHMIC_PROXY"
        assert row["protocol_arm_id"]
        assert row["arm"].endswith("_proxy")
        assert row["arm"] != row["protocol_arm_id"]
