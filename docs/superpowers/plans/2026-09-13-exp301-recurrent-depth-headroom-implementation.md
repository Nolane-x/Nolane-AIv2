# EXP-301 Recurrent-Depth Headroom Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the frozen NLM V0.17 EXP-301 court that tests whether a 10,000,000-parameter shared recurrent latent core has reproducible verified-generalization headroom over strong fixed-depth and simple recurrent rivals under matched resident-parameter and accounted-compute constraints.

**Architecture:** EXP-301 is deliberately narrow. It adds a new V0.17 model family, deterministic challenge generators, a matched-arm compiler, an accounted-FLOP ledger, a sealed paired runner, and a preregistered reducer. The implementation must not add EXP-302+ sidecars (halting, event routing, uncertainty, conflict memory, structural transfer, fidelity distillation). All three arms must have exactly 10,000,000 active trainable parameters; no dead padding is permitted. Fixed-depth comparison uses a frontier of independently compiled fixed-depth Transformer models, one per frozen comparison budget, so recurrence is not advantaged by giving the fixed rival only one compute point.

**Tech Stack:** Python 3.11/3.13, PyTorch >=2.7, pytest >=8, GitHub Actions, repository-native protocol/evidence utilities.

**Spec:** `docs/superpowers/specs/2026-09-13-nlm-v017-10m-native-recursive-substrate-design.md` and `protocols/v017/exp301_preregistration_v1.json`

## Global Constraints

- The V0.16.1 main authority remains immutable history. EXP-301 is a scientifically distinct V0.17 lineage.
- Frozen authority parent: `main@a7d14a97ae74ace9a0f4be4e72fade434cbaa96b`.
- EXP-301 preregistration semantic digest is `110701d1054fe0743bac2b65d16faa602182a1d90ab85bdc7d3d426de0212253`.
- Digest convention: remove the top-level `registration_digest` field, serialize JSON with `sort_keys=True` and separators `(",", ":")`, UTF-8 encode, then SHA-256.
- Resident trainable parameters MUST equal `10_000_000` for every scientific arm/model instance.
- V0.17 frozen base geometry: vocab 4,608; d_model 448; heads 7 x 64; shared block 3 layers; d_ff 1,152; frozen base count 9,120,832; Capacity Exchange Envelope 879,168.
- Every parameter in the Capacity Exchange Envelope must participate in the forward computation. No filler tensor or unreachable parameter is permitted.
- Training roots are exactly `[0,1,2,3]`.
- Training loop distribution is exactly `[1,2,4,8]`.
- Challenge loop budgets are exactly `[1,2,4,8,12,16]`.
- Arms are exactly `A_FIXED`, `B_LOOP_SIMPLE`, `C_NRS_CORE`.
- EXP-301 contains no adaptive halting, event router, explicit uncertainty sidecar, conflict memory, structural-transfer memory, teacher/scaffold path, external tools, or hidden retrieval.
- Challenge generators must be materialized only after a frozen pre-data identity is recorded. Generator identities must be disjoint from train/development identities.
- The primary decision rule is frozen: all protected floors; RCG >= +0.05 vs `A_FIXED`; >= +5pp verified success on at least two reasoning families at one common compute budget; language-control regression <=2pp; directional positive on >=3/4 roots; 95% paired root-stratified hierarchical-bootstrap CI excludes zero.
- A valid negative result kills `H-RD-01`; no post-result rescue by loop schedule, thresholds, roots, task weights, sample count, or compute accounting.
- Normal CI is TEST-ONLY and MUST NOT produce scientific evidence or promotion authority.
- Scientific execution must be a separately sealed workflow with explicit frozen identities and evidence eligibility.

---

## Task 1: Freeze protocol semantics and V0.17 budget authority

**Files:**
- Modify: `src/nolane_ai/model/budget.py`
- Create: `src/nolane_ai/protocol/v017.py`
- Create: `tests/test_v017_budget.py`
- Create: `tests/test_exp301_protocol.py`
- Create: `protocols/v017/exp301_preregistration_v1.sha256`

- [ ] **Step 1: RED — budget identity test**

Add tests asserting:
- `authoritative_v017_10m_budget().total_parameters == 10_000_000`;
- frozen base + CEE equals 10M;
- CEE is exactly 879,168;
- no V0.16 region is silently reused as V0.17 architecture authority.

Expected RED: import/name failure because V0.17 budget function does not exist.

- [ ] **Step 2: Verify RED in GitHub CI**

Commit tests only. Confirm exact-head CI fails for the expected missing V0.17 symbols, not unrelated baseline failures.

- [ ] **Step 3: GREEN — minimal V0.17 budget authority**

Implement immutable V0.17 budget constants and `authoritative_v017_10m_budget()` in `budget.py`. Keep V0.16 functions unchanged.

- [ ] **Step 4: RED — preregistration semantic digest tests**

Tests must verify:
- semantic digest convention excludes `registration_digest`;
- recomputation equals frozen digest;
- changing any scientific field invalidates the digest;
- whitespace/key ordering does not change semantic digest;
- unknown experiment ID/root/loop/arm modifications fail validation.

- [ ] **Step 5: GREEN — protocol validator**

Create `src/nolane_ai/protocol/v017.py` with:
- `canonical_registration_payload(payload) -> bytes`
- `registration_digest(payload) -> str`
- `load_exp301_registration(path) -> dict`
- `validate_exp301_registration(payload) -> None`
- frozen constants for EXP-301 arm IDs, roots, training/challenge loops, parameter ceiling, and expected semantic digest.

- [ ] **Step 6: Add sidecar**

`protocols/v017/exp301_preregistration_v1.sha256` stores the semantic digest plus a comment-free one-line convention identifier in tests/documentation; validator treats the JSON semantic digest, not raw-file bytes, as scientific identity.

- [ ] **Step 7: Verify focused + full CI**

Focused:
`pytest -q tests/test_v017_budget.py tests/test_exp301_protocol.py`

Then exact-head normal CI.

---

## Task 2: Implement the exact-10M active-capacity model primitives

**Files:**
- Create: `src/nolane_ai/model/v017_recursive.py`
- Create: `tests/test_v017_recursive_model.py`

- [ ] **Step 1: RED — shared block geometry**

Tests specify:
- RMSNorm, causal multi-head attention, SwiGLU FFN;
- 448-dimensional stream;
- 7 attention heads of width 64;
- d_ff 1,152;
- 3-layer shared recurrent block;
- tied input embedding/output projection;
- base model parameter count exactly 9,120,832 before Capacity Exchange allocation.

- [ ] **Step 2: RED — active Capacity Exchange Envelope**

Define test behavior for an `ActiveCapacityExchange` module:
- consumes exactly the requested parameter budget;
- every trainable parameter receives gradient in a forward/backward smoke;
- changing its weights changes output;
- no parameter named/marked as padding/filler/reserve is permitted;
- total scientific model trainable count is exactly 10,000,000.

The module may use a factorized/gated tokenwise residual MLP plus a small exact-budget projection tail, but every scalar must contribute to the output graph.

- [ ] **Step 3: RED — recurrent semantics**

Tests specify:
- `B_LOOP_SIMPLE`: same shared 3-layer block repeated `loops` times, no loop-index conditioning;
- `C_NRS_CORE`: same block repeated `loops` times with learned bounded loop conditioning;
- loop conditioning parameters are paid from CEE and matched by equal active generic capacity in rival arms;
- `loops <= 0` rejected;
- no adaptive stopping in EXP-301.

- [ ] **Step 4: GREEN — primitives**

Implement only what tests require:
- `RMSNorm`
- `CausalSelfAttention`
- `SwiGLU`
- `V017TransformerLayer`
- `ActiveCapacityExchange`
- `LoopConditioner`
- `V017RecurrentLM`

- [ ] **Step 5: Verify parameter/gradient invariants**

Run focused tests and require exact equality, not tolerance.

---

## Task 3: Build a strongest-fair fixed-depth frontier and matched arm compiler

**Files:**
- Create: `src/nolane_ai/experiments/exp301_arms.py`
- Create: `tests/test_exp301_arms.py`

- [ ] **Step 1: RED — arm contract**

Tests define:
- `Exp301ArmId = A_FIXED | B_LOOP_SIMPLE | C_NRS_CORE`;
- same vocab/token interface;
- each instantiated arm has exactly 10,000,000 active trainable parameters;
- same output vocabulary and verifier-facing API.

- [ ] **Step 2: RED — fixed-depth frontier**

A single one-pass fixed model is not a valid RCG rival. Tests require a fixed-depth frontier keyed by the frozen common compute budgets. Each frontier point:
- is independently parameter-compiled;
- has no weight tying across depth;
- is exactly 10M active parameters;
- uses the strongest feasible width/FFN allocation under the 10M resident budget;
- has accounted FLOPs within frozen match tolerance of its recurrent comparison point.

If no fixed architecture can satisfy both active-10M and FLOP match for a budget, that comparison budget is invalid for all arms rather than silently favoring recurrence.

- [ ] **Step 3: GREEN — deterministic compiler**

Implement:
- `compile_fixed_frontier_point(...)`
- `build_simple_recurrent_arm(...)`
- `build_nrs_core_arm(...)`
- `audit_active_parameters(model)`
- `audit_no_dead_capacity(model, smoke_batch)`
- deterministic architecture receipts describing depth, width, FFN, CEE allocation, and count.

- [ ] **Step 4: Verify rival-strength invariants**

Test that A_FIXED is not intentionally starved:
- no dead CEE;
- optimizer-visible parameters all active;
- model depth/width receipt is frozen before evaluation;
- same training-token budget and optimizer family enforced downstream.

---

## Task 4: Implement accounted compute ledger and common-budget matcher

**Files:**
- Create: `src/nolane_ai/experiments/exp301_compute.py`
- Create: `tests/test_exp301_compute.py`

- [ ] **Step 1: RED — FLOP accounting tests**

Define deterministic analytical accounting for:
- embedding/output matmul where applicable;
- Q/K/V/O projections;
- attention score/value products as sequence length changes;
- SwiGLU projections;
- loop conditioning;
- active CEE path;
- verification/output overhead that differs by arm.

- [ ] **Step 2: RED — no free recurrent loops**

Increasing loops must strictly increase accounted FLOPs. Any hidden recurrent state/memory update cost is charged.

- [ ] **Step 3: RED — common-budget matching**

For each frozen comparison point:
- all compared arms must be within preregistered FLOP tolerance;
- tolerance is frozen before challenge materialization;
- unmatched point becomes `INVALID_COMPUTE_MATCH`, never interpolated post hoc.

- [ ] **Step 4: GREEN — ledger**

Implement immutable compute receipts and common-budget matcher. Keep wall-clock latency diagnostic only; primary compute authority is accounted FLOPs.

---

## Task 5: Materialize deterministic train/development worlds

**Files:**
- Create: `src/nolane_ai/experiments/exp301_worlds.py`
- Create: `tests/test_exp301_worlds.py`

- [ ] **Step 1: RED — family identities**

Implement tests for four frozen families:
1. `iterative-grid-and-maze`
2. `algorithmic-sequence-transform`
3. `generator-heldout-abstract-transformation`
4. `language-sequence-control`

Every instance must have:
- stable content ID;
- generator family/version;
- root;
- split;
- difficulty/required-step metadata where meaningful;
- deterministic verifier.

- [ ] **Step 2: RED — disjointness**

Tests require zero content ID overlap between train/development/challenge and zero generator-template identity overlap where the preregistration promises generator-heldout behavior.

- [ ] **Step 3: RED — anti-shortcut metamorphics**

For reasoning families, paired transformations must preserve solution while changing superficial token/template features. A model receipt cannot use challenge generator metadata as input.

- [ ] **Step 4: GREEN — world generators**

Implement small but nontrivial programmatic generators with exact verifiers. Keep generators independent of any arm implementation.

- [ ] **Step 5: Freeze development geometry only**

Normal CI can materialize TEST-ONLY development examples. Scientific challenge seeds/templates are not committed pre-freeze if that would leak challenge identity.

---

## Task 6: Add training contract shared by all arms

**Files:**
- Create: `src/nolane_ai/experiments/exp301_training.py`
- Create: `tests/test_exp301_training.py`

- [ ] **Step 1: RED — matched training receipt**

Tests require identical across arms/root:
- tokenizer and examples;
- batch/token budget;
- optimizer family;
- learning-rate search budget;
- number of allowed search trials;
- checkpoint selection rule;
- data order identity.

Architecture-specific unavoidable hyperparameters must be enumerated before challenge execution.

- [ ] **Step 2: RED — fair search budget**

Hyperparameter search is allowed only on development data and must have the same total trial/compute allowance per arm. The winning configuration/receipt is frozen before challenge materialization.

- [ ] **Step 3: GREEN — trainer**

Implement a compact deterministic trainer supporting:
- root seeding;
- mixed training loops `{1,2,4,8}` for recurrent arms;
- fixed-frontier training per operating point;
- gradient clipping and optimizer state receipt;
- checkpoint digest.

Scientific scale runs may be longer than CI; CI uses tiny geometry but must traverse identical code paths with `scientific_evidence_eligible=false`.

---

## Task 7: Implement paired held-out evaluator and protected floors

**Files:**
- Create: `src/nolane_ai/experiments/exp301_evaluation.py`
- Create: `tests/test_exp301_evaluation.py`

- [ ] **Step 1: RED — verifier-first scoring**

Tests require predictions to be committed before verifier outcome is revealed to model/training code.

- [ ] **Step 2: RED — protected floors**

Evaluator must measure:
- verified success by family/root/compute point;
- invalid-output rate;
- language-sequence control regression;
- challenge identity leakage checks;
- parameter and compute receipts.

- [ ] **Step 3: RED — unseen-depth diagnostics**

Loops 12/16 are marked extrapolation/stability probes. They cannot alone satisfy the promotion rule. Representation collapse/NaN/divergence is recorded, not dropped.

- [ ] **Step 4: GREEN — evaluator**

Return immutable per-instance rows plus root/family aggregates. Failed model outputs remain failures unless infrastructure invalidation is proven.

---

## Task 8: Implement preregistered RCG reducer and hierarchical bootstrap

**Files:**
- Create: `src/nolane_ai/experiments/exp301_analysis.py`
- Create: `tests/test_exp301_analysis.py`

- [ ] **Step 1: RED — normalized AUC/RCG**

Freeze one implementation of normalized area under verified-success versus `log2(accounted_FLOPs)` over the intersection of valid common budgets. No post-outcome interpolation rule changes.

- [ ] **Step 2: RED — paired hierarchical bootstrap**

Bootstrap:
- root replication strata;
- paired held-out instances within root/family;
- deterministic bootstrap seed recorded separately from train roots;
- 95% CI for aggregate RCG.

- [ ] **Step 3: RED — exact conjunctive decision**

Synthetic fixtures must cover:
- full promotion;
- RCG pass but language floor fail -> no promotion;
- two-family +5pp fail -> no promotion;
- only 2/4 roots positive -> no promotion;
- CI touches zero -> no promotion;
- infrastructure invalid -> `INVALID_COURT`, not kill/promote;
- valid negative -> `KILL_H_RD_01`.

- [ ] **Step 4: GREEN — reducer**

Output one of:
- `PROMOTE_H_RD_01_TO_EXP302_DESIGN_ONLY`
- `KILL_H_RD_01`
- `RECURRENCE_STABILITY_DIAGNOSTIC_ONLY`
- `INVALID_COURT`

Promotion authorizes only EXP-302 design/preregistration, never direct implementation of EXP-302+ or scale.

---

## Task 9: Build sealed runner, receipts, and rerun policy

**Files:**
- Create: `src/nolane_ai/experiments/exp301_runner.py`
- Create: `tests/test_exp301_runner.py`
- Create: `scripts/run_exp301.py`

- [ ] **Step 1: RED — execution identity**

Runner must bind:
- source commit SHA;
- source-tree/code digest;
- prereg semantic digest;
- frozen architecture receipts;
- root;
- train/dev/challenge generator digests;
- selected hyperparameter receipt;
- parameter audit;
- compute ledger version.

- [ ] **Step 2: RED — no overwrite / no duplicate authority**

Scientific result path is write-once. Same scientific run identity cannot be silently executed twice. Infrastructure-invalid retries require a new retry receipt linking the failed attempt and proving no scientific outcome was consumed.

- [ ] **Step 3: RED — TEST-ONLY boundary**

`--test-only`:
- may use tiny models/geometry;
- sets `scientific_evidence_eligible=false`;
- cannot emit promote/kill scientific authority;
- must exercise train/evaluate/reduce/receipt code.

- [ ] **Step 4: GREEN — runner and CLI**

CLI has no tuning flags for frozen scientific fields. Runtime knobs limited to output location, test-only mode, and explicitly frozen execution resources.

---

## Task 10: CI integration without accidental scientific execution

**Files:**
- Modify: `.github/workflows/ci.yml`
- Create: `.github/workflows/exp301-scientific-court.yml`
- Create: `tests/test_exp301_workflows.py`

- [ ] **Step 1: RED — workflow contract tests**

Tests require normal CI to run:
- EXP-301 protocol tests;
- parameter identity/dead-capacity tests;
- compute/world/analysis tests;
- one tiny end-to-end TEST-ONLY smoke.

Tests require normal CI to contain no scientific EXP-301 execution.

- [ ] **Step 2: GREEN — normal CI**

Add focused EXP-301 tests to `model-smoke`, plus one tiny CLI smoke with explicit `--test-only`.

- [ ] **Step 3: GREEN — scientific workflow skeleton**

Scientific workflow must:
- be manually/authority triggered only after freeze marker;
- verify exact source/prereg/challenge identity;
- fan out exactly roots 0–3;
- upload immutable root artifacts;
- run one cross-root reducer from artifact bytes;
- verify no duplicate scientific run exists;
- never expose challenge outcome to training jobs before prediction commitments.

Do NOT run the scientific workflow until all pre-data gates and exact-head CI are green.

---

## Task 11: Pre-data freeze review

**Files:**
- Create: `docs/superpowers/specs/2026-09-13-exp301-pre-data-freeze.md`
- Create: `protocols/v017/exp301_execution_identity_v1.json`
- Create: `protocols/v017/exp301_execution_identity_v1.sha256`

- [ ] **Step 1: Freeze exact implementation identity**

Record exact head SHA and digests for:
- code paths;
- protocol;
- training geometry;
- task generators;
- compute ledger;
- analysis code;
- workflow.

- [ ] **Step 2: Verify no scientific run exists**

Confirm zero prior authoritative EXP-301 scientific workflow runs/artifacts.

- [ ] **Step 3: Independent anti-Goodhart audit**

Before challenge materialization verify:
- fixed frontier genuinely uses matched compute;
- CEE is active in all arms;
- challenge metadata cannot enter model input;
- no optional V0.17 sidecars;
- 12/16 loops cannot alone promote;
- invalid runs cannot disappear from reducer.

- [ ] **Step 4: Exact-head CI**

Require:
- Python 3.11 core GREEN;
- Python 3.13 core GREEN;
- model-smoke GREEN;
- EXP-301 dedicated contract GREEN.

Only then is the pre-data head eligible for a marker-only freeze commit.

---

## Task 12: Execute EXP-301 once and close the scientific disposition

**Files generated by workflow, not hand-edited:**
- `evidence/exp301/<run-identity>/root-0/...`
- `evidence/exp301/<run-identity>/root-1/...`
- `evidence/exp301/<run-identity>/root-2/...`
- `evidence/exp301/<run-identity>/root-3/...`
- `evidence/exp301/<run-identity>/cross-analysis.json`
- `evidence/exp301/<run-identity>/PROVENANCE.md`
- `evidence/exp301/<run-identity>/SHA256SUMS`

- [ ] **Step 1: Marker-only release**

Marker commit changes identity metadata only. No thresholds, model code, data, generator, optimizer, or reducer changes.

- [ ] **Step 2: One authoritative run**

Run exactly roots 0–3 with frozen geometry.

- [ ] **Step 3: Byte-level evidence audit**

Recompute:
- artifact hashes;
- semantic protocol digest;
- parameter receipts;
- compute receipts;
- per-instance metrics;
- RCG;
- bootstrap CI;
- conjunctive decision.

- [ ] **Step 4: Close authority honestly**

If promotion:
- only `EXP-302 DESIGN/PREREGISTRATION` becomes authorized.
- W5 advances from r8 only to stages justified by actual evidence.

If valid negative:
- close `H-RD-01` as killed;
- do not implement EXP-302 on this lineage;
- preserve diagnostics as evidence for a scientifically distinct recurrence hypothesis.

If invalid:
- no scientific decision;
- rerun only under frozen infrastructure-invalid retry policy.

---

## Task 13: Final verification and branch disposition

- [ ] Run exact-head CI and dedicated EXP-301 checks.
- [ ] Inspect PR diff for accidental V0.16 changes or EXP-302+ code.
- [ ] Confirm no dead/filler parameters.
- [ ] Confirm all scientific claims have evidence artifacts.
- [ ] Invoke `superpowers:verification-before-completion`.
- [ ] Invoke `superpowers:requesting-code-review`.
- [ ] Invoke `superpowers:finishing-a-development-branch`.
- [ ] Merge only documentation/protocol/implementation that is supported by GREEN checks; scientific result closure must state the actual court outcome, including negative/invalid outcomes without reinterpretation.
