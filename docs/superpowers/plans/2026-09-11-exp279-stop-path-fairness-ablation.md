# EXP-279 Stop-Path Fairness Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the positive DEVELOPMENT utility seen after PR #40 came from genuine stop-path representation allocation, auxiliary hybrid training, or the cheaper inference path created when branch capacity is inactive.

**Architecture:** Add a DEVELOPMENT-only four-arm ablation on the existing EXP-279 worlds. Compare `propagation_clean`, exact `raw_stop_clean`, exact `raw_stop_aux`, and an `active_reclaimed_stop_clean` arm that executes the branch-GRU capacity as a one-step summary without multi-step branch search. The active reclaimed arm must have the same parameter state structure and the same accounted inference FLOPs as propagation-only; raw-stop arms must explicitly report inactive branch-GRU execution capacity and are diagnostic-only.

**Tech Stack:** Python 3.11+, PyTorch, pytest, existing `Exp279RoutingGenerator`, existing matched EXP-279 arm modules and analytical FLOP ledger, GitHub Actions.

**Spec:** `docs/NLM_REASONING_ARCHITECTURE_SPEC.md` plus the frozen `protocols/stage_a_v1.json`; this ablation is DEVELOPMENT-only and must not mutate either frozen source.

## Global Constraints

- Base commit is `main@934cb0a6f19a0d4ea582f0e1a036cdc9bf5348ac`.
- Do not alter `protocols/stage_a_v1.json`, MESI, alpha, power, Holm family, sample-size ceiling, or confirmatory lineage.
- Do not consume confirmatory data, create a confirmatory beacon, or claim confirmatory evidence.
- Use the existing EXP-279 DEVELOPMENT generator and paired train/eval replicate lineage.
- Evaluation route masks for every stop-only arm must be all false.
- `raw_stop_clean` must be exactly the current hybrid stop forward path with route threshold forced to `1.0` and no hybrid auxiliary losses.
- `raw_stop_aux` must use the same inference path as `raw_stop_clean` but the current hybrid DEVELOPMENT auxiliary training objective, isolating the training-objective confound.
- `active_reclaimed_stop_clean` must activate branch-GRU capacity only as a one-step summary and must not perform multi-step branch search.
- `active_reclaimed_stop_clean` accounted FLOPs per episode must equal `propagation_clean` accounted FLOPs per episode for the same geometry.
- All four arms must begin from byte-identical model state.
- Outputs must state `scientific_evidence_eligible=false`, `evidence_level=EV-E2`, and `decision=UNVERIFIED`.

---

### Task 1: RED contract for the ablation arms

**Files:**
- Create: `tests/test_exp279_stop_path_ablation.py`
- Create: `.github/workflows/exp279-stop-path-ablation-ci.yml`

**Interfaces:**
- Consumes: existing `HybridRoutingArm`, `PropagationOnlyArm`, `_hybrid_stop_decision_logits`, and `_compute_ledgers`.
- Produces: required symbols `ActiveReclaimedStopArm`, `build_stop_path_ablation_arms`, `stop_path_ablation_compute_ledger`, `run_exp279_stop_path_ablation`.

- [ ] **Step 1: Write failing exact-stop test**

```python
def test_raw_stop_is_exact_hybrid_stop_path():
    arms = build_stop_path_ablation_arms(root_seed="stop-ablation-test", d_model=16, hidden_size=12, target_parameters=12000)
    batch = make_test_batch(...)
    raw = arms["raw_stop_clean"](batch.surface_events, batch.variable_states, batch.incidence)
    expected = _hybrid_stop_decision_logits(arms["raw_stop_clean"], surface_events=batch.surface_events, variable_states=batch.variable_states, incidence=batch.incidence)
    assert not raw.branch_route_mask.any()
    assert torch.allclose(raw.decision_logits, expected)
```

- [ ] **Step 2: Write failing active-capacity test**

```python
def test_active_reclaimed_stop_executes_one_step_branch_capacity_without_routing():
    arm = build_stop_path_ablation_arms(...)["active_reclaimed_stop_clean"]
    calls = []
    handle = arm.branch_gru.register_forward_hook(lambda *_: calls.append(True))
    output = arm(...)
    handle.remove()
    assert calls == [True]
    assert not output.branch_route_mask.any()
    assert output.representation_semantics == "propagation_stop_with_one_step_active_reclaimed_branch_capacity"
```

- [ ] **Step 3: Write failing ledger fairness test**

```python
def test_active_reclaimed_stop_matches_propagation_inference_flops():
    ledger = stop_path_ablation_compute_ledger(...)
    assert ledger["active_reclaimed_stop_clean"]["accounted_flops_per_episode"] == ledger["propagation_clean"]["accounted_flops_per_episode"]
    assert ledger["raw_stop_clean"]["accounted_flops_per_episode"] < ledger["propagation_clean"]["accounted_flops_per_episode"]
    assert ledger["raw_stop_clean"]["inactive_execution_parameters"] > 0
    assert ledger["active_reclaimed_stop_clean"]["inactive_execution_parameters"] == 0
```

- [ ] **Step 4: Write failing clean-vs-aux training isolation test**

```python
def test_aux_training_changes_raw_stop_shared_state_while_clean_training_does_not_touch_branch_gru():
    arms = build_stop_path_ablation_arms(...)
    before_clean_branch = clone_prefix(arms["raw_stop_clean"], "branch_gru.")
    train_clean_step(arms["raw_stop_clean"], ...)
    assert prefix_equal(before_clean_branch, clone_prefix(arms["raw_stop_clean"], "branch_gru."))
    before_aux_branch = clone_prefix(arms["raw_stop_aux"], "branch_gru.")
    train_aux_step(arms["raw_stop_aux"], ...)
    assert not prefix_equal(before_aux_branch, clone_prefix(arms["raw_stop_aux"], "branch_gru."))
```

- [ ] **Step 5: Commit RED tests and CI gate.**

### Task 2: GREEN arm and ledger implementation

**Files:**
- Create: `src/nolane_ai/experiments/exp279_stop_path_ablation.py`
- Modify: `tests/test_exp279_stop_path_ablation.py`

**Interfaces:**
- `ActiveReclaimedStopArm.forward(surface_events, variable_states, incidence) -> Exp279ArmOutput`
- `build_stop_path_ablation_arms(...) -> dict[str, nn.Module]`
- `stop_path_ablation_compute_ledger(arm, timesteps, variables, constraints) -> dict[str, dict]`
- `_train_clean_step(...) -> float`
- `_train_aux_step(...) -> float`

- [ ] **Step 1: Implement `ActiveReclaimedStopArm`.** Compute `propagation_state`, run branch GRU on `events.mean(dim=1, keepdim=True)`, expand the one-step context across variables, apply `reclaimed_projection(propagation_state + context)`, add it residually, compute the diagnostic routing statistic, and always return an all-false route mask.
- [ ] **Step 2: Implement byte-identical four-arm construction.** Seed once, instantiate one propagation arm plus three hybrid-compatible arms, then load the propagation state dict into all other arms before any training.
- [ ] **Step 3: Implement analytical ledgers.** Reuse `_compute_ledgers`; raw-stop cost is `hybrid.stop_accounted_flops_per_episode`; active reclaimed cost is raw-stop cost plus the existing propagation ledger's `reclaimed_branch_summary`, which must equal propagation total exactly. Report branch-GRU parameter count as inactive only for raw-stop inference arms.
- [ ] **Step 4: Implement clean training.** Use one decision CE for the executed stop path and optional head-only routing calibration with the same episode-failure target, without forced-branch or duplicated stop auxiliary losses.
- [ ] **Step 5: Implement aux training by calling the current EXP-279 hybrid training step on `raw_stop_aux`, preserving threshold `1.0` so inference remains stop-only.
- [ ] **Step 6: Run the Task 1 tests and focused EXP-279 regressions; commit GREEN implementation.**

### Task 3: Paired DEVELOPMENT runner and CLI

**Files:**
- Modify: `src/nolane_ai/experiments/exp279_stop_path_ablation.py`
- Create: `scripts/run_exp279_stop_path_ablation.py`
- Modify: `tests/test_exp279_stop_path_ablation.py`

**Interfaces:**
- `run_exp279_stop_path_ablation(root_seed, d_model, hidden_size, target_parameters, train_replicates, eval_replicates, eval_start_replicate, batch_size, timesteps, variables, constraints, noise_std, lr, weight_decay) -> dict`

- [ ] **Step 1: Train all four arms on identical augmentation batches and strata.** Use `STRATA` round-robin and retain paired batch digests.
- [ ] **Step 2: Evaluate all four arms on identical held-out DEVELOPMENT evaluation batches.** Record exact solution rate, decision accuracy, accounted FLOPs, utility/FLOP, route fraction, and stratum.
- [ ] **Step 3: Aggregate paired descriptive deltas.** Report `raw_clean_vs_propagation`, `raw_aux_vs_raw_clean`, and `active_reclaimed_vs_propagation` for utility and solution rate; do not run confirmatory/Holm inference.
- [ ] **Step 4: Emit fairness classification.** `raw_stop_*` are `DIAGNOSTIC_INACTIVE_CAPACITY`; `active_reclaimed_stop_clean` vs propagation is `FAIR_ACTIVE_CAPACITY_SAME_FLOPS`.
- [ ] **Step 5: Validate hard boundaries.** Assert no route activation, no confirmatory data consumption/materialization, identical initial digests, and exact same-FLOP equality for the fair pair.
- [ ] **Step 6: Add a CLI that writes canonical JSON output and prints a compact summary.**
- [ ] **Step 7: Run tests and commit.**

### Task 4: DEVELOPMENT sweep and falsification record

**Files:**
- Create: `.github/workflows/exp279-stop-path-ablation-sweep.yml`
- Update PR description with results; do not merge solely because descriptive utility is positive.

- [ ] **Step 1: Add matrix `train_replicates=[15,60,120]`, eval IDs `20000..20032`, seed `20260906-exp279-paired-dev`, `d_model=64`, `hidden_size=48`, target parameters `500000`, batch `8`, timesteps `4`, variables `6`, constraints `3`, noise `0.05`, lr `0.002`, weight decay `0.0`.
- [ ] **Step 2: Verify `scripts/verify_protocol.py` before each matrix run.**
- [ ] **Step 3: Upload the complete DEVELOPMENT artifact and print the three diagnostic contrasts.
- [ ] **Step 4: Interpret results conservatively:**
  - raw clean advantage only -> cheaper inactive-capacity execution/accounting effect;
  - raw aux > raw clean -> auxiliary-training confound contributes;
  - active reclaimed > propagation at equal FLOPs and equal active capacity -> representation-allocation hypothesis survives DEVELOPMENT;
  - active reclaimed <= propagation -> representation-allocation hypothesis falsified under this geometry.
- [ ] **Step 5: Keep confirmatory state untouched and record `UNVERIFIED / EV-E2` regardless of descriptive outcome.
