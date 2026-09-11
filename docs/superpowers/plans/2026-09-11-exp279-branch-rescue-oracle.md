# EXP-279 Branch Rescue Capacity / Oracle Economics Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use test-driven-development and verification-before-completion. Keep this DEVELOPMENT-only.

**Goal:** Determine whether the trained hybrid branch path has any recoverable exact-solution value independent of the route head, and whether a perfect post-hoc oracle could exploit that value profitably after charged branch FLOPs.

**Architecture:** Replay canonical EXP-279 hybrid training exactly. On each identical DEVELOPMENT evaluation batch compute the canonical hybrid stop logits and forced-branch logits using the same trained weights. Partition episodes into stop/branch exact outcomes, then construct a non-deployable ground-truth rescue oracle that branches only on episodes where stop fails and forced branch succeeds. Charge the oracle the frozen hybrid stop/branch FLOP ledger. This diagnostic never changes route threshold, training teacher, model weights, canonical artifacts or confirmatory state.

**Frozen settings:** same protocol/seed/model/evaluation lineage as PR #40/#43; output stays `EV-E2 / UNVERIFIED`, scientific evidence ineligible.

## Task 1 — RED contract
- Create `tests/test_exp279_branch_rescue_oracle.py` and a focused CI workflow.
- Require exact outcome partition: stop-only success, branch-only rescue, stop-only harm, both fail/succeed.
- Require oracle output/cost identity and classification.
- Require tiny replay final hybrid digest + training batch digests to match canonical paired runner.
- Confirm RED because implementation module is absent.

## Task 2 — Implement same-weights rescue diagnostic
- Create `src/nolane_ai/experiments/exp279_branch_rescue_oracle.py`.
- Train hybrid via canonical `_build_seeded_triplet`, `_train_step`, generator and optimizer.
- Evaluate canonical `_hybrid_stop_decision_logits` and `_hybrid_forced_branch_decision_logits` on identical evaluation batches.
- Record per-batch and aggregate:
  - stop exact successes/failures;
  - forced-branch exact successes/failures;
  - branch rescues (`stop fail & branch success`);
  - branch harms (`stop success & branch fail`);
  - both success / both fail;
  - forced-branch vs stop solution rates;
  - stop utility, always-branch utility;
  - rescue-oracle route fraction, solution rate, charged FLOPs and utility.
- Classify:
  - `NO_BRANCH_RESCUE_CAPACITY` if rescues are zero;
  - `RESCUES_EXIST_BUT_ORACLE_NOT_COST_EFFECTIVE` if rescues exist but rescue-oracle utility does not beat stop;
  - `BRANCH_RESCUE_CAPACITY_COST_EFFECTIVE` if rescues exist and rescue-oracle utility beats stop.

## Task 3 — DEVELOPMENT sweep
- Add deterministic CLI and matrix `[15,60,120]`.
- Use seed `20260906-exp279-paired-dev`, `d_model=64`, `hidden_size=48`, target params `500000`, threshold `0.5`, eval IDs `20000..20032`, batch `8`, timesteps `4`, variables `6`, constraints `3`, noise `0.05`, lr `0.002`, wd `0`.
- Upload 90-day receipts and assert no confirmatory/challenge/promotion.

## Task 4 — Decision rule
- If no rescue capacity: stop teacher/threshold work and redesign branch representation/training.
- If rescues exist but oracle is uneconomic: redesign branch compute/path economics before routing.
- If oracle is cost-effective: routing remains a viable mechanism, but route-head learning must be redesigned to approximate the oracle using only allowed information.
- Do not open confirmatory gates in any case from this DEVELOPMENT diagnostic alone.
