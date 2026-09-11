# EXP-279 Routing Marginal Shadow Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the marginal value of EXP-279 routing by evaluating the trained hybrid arm twice on each identical DEVELOPMENT episode: normal routed execution and a forced-stop counterfactual using the same weights.

**Architecture:** Reuse the frozen main-branch hybrid initialization, training batches, optimizer, routing teacher, and compute ledger. Add a DEVELOPMENT-only shadow evaluator that reconstructs the hybrid stop-path output without branch execution and compares exact-solution outcomes and charged utility against the real hybrid output. This is diagnostic only and never modifies the canonical paired artifact or confirmatory protocol.

**Tech Stack:** Python, PyTorch, pytest, GitHub Actions.

**Spec:** `protocols/stage_a_v1.json` EXP-279 plus `src/nolane_ai/experiments/exp279_paired_runner.py` on `main@934cb0a6f19a0d4ea582f0e1a036cdc9bf5348ac`.

## Global Constraints
- DEVELOPMENT-only, `EV-E2`, `UNVERIFIED`, no confirmatory data/challenge.
- Same root seed, model-init stream, augmentation/evaluation lineage, optimizer and hybrid training step as the canonical runner.
- Shadow stop uses the same trained hybrid weights and same evaluation batch.
- Actual hybrid is charged path-dependent stop/branch FLOPs; shadow is charged hybrid stop FLOPs.
- No change to route threshold, teacher, MESI, sample-size ceiling, protocol bytes or canonical artifact schema.

---

### Task 1: Freeze RED contract
**Files:** create `tests/test_exp279_routing_marginal_shadow.py` and `.github/workflows/exp279-routing-marginal-shadow-ci.yml`.
- [ ] Require forced-stop output to have zero branch mask and identical stop-path logits to canonical stop helper.
- [ ] Require per-episode rescue/harm accounting to close exactly.
- [ ] Require training replay compatibility by matching final hybrid functional digest against the canonical paired runner on a tiny run.
- [ ] Confirm RED before implementation.

### Task 2: Implement shadow evaluator
**Files:** create `src/nolane_ai/experiments/exp279_routing_marginal.py`.
- [ ] Build `_forced_stop_output` from the trained `HybridRoutingArm` without branch recurrence.
- [ ] Train hybrid with canonical `_build_seeded_triplet`, `_train_step`, worlds and optimizer.
- [ ] Evaluate actual and forced-stop outputs on identical DEVELOPMENT batches.
- [ ] Record routed count, rescues, harms, unchanged successes/failures, actual/shadow solution rates, costs and utilities.
- [ ] Aggregate routing marginal solution delta and routing marginal utility gain; classify as `ROUTING_INACTIVE`, `ROUTING_ADDS_POSITIVE_UTILITY`, `ROUTING_HARMS_UTILITY`, or `ROUTING_UTILITY_NEUTRAL`.

### Task 3: Run 15/60/120 DEVELOPMENT sweep
**Files:** create `scripts/run_exp279_routing_marginal_dev.py`; extend focused workflow with matrix `[15,60,120]`.
- [ ] Use seed `20260906-exp279-paired-dev`, `d_model=64`, `hidden_size=48`, target params `500000`, threshold `0.5`, eval IDs `20000..20032`, batch `8`, timesteps `4`, variables `6`, constraints `3`, noise `0.05`, lr `0.002`, weight decay `0.0`.
- [ ] Upload receipts for each train count.
- [ ] Keep all outputs descriptive DEVELOPMENT evidence only.

### Task 4: Verify and interpret
- [ ] Run generic CI, focused EXP-279 contract and shadow-control CI.
- [ ] If routing utility is non-positive across sweep, stop proposing new teachers until branch expert/path economics are redesigned.
- [ ] If routing adds positive utility at any train count, isolate where/why before any confirmatory step.
