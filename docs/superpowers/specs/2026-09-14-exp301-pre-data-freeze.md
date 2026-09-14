# EXP-301 Pre-Data Freeze Closure

**Status:** PRE-MARKER REVIEW AUTHORITY  
**Lineage:** NLM V0.17 / EXP-301 Recurrent-Depth Headroom  
**Scientific outcome:** UNVERIFIED  
**Challenge materialized:** NO  
**EXP-302 implementation authorized:** NO  
**Scale authorized:** NO

## 1. Purpose

This document closes the implementation-side pre-data review for EXP-301. It is intentionally committed **before** the marker-only execution-identity commit so it becomes part of the frozen source tree.

The marker commit that follows this review is allowed to add exactly two files and nothing else:

- `protocols/v017/exp301_execution_identity_v1.json`
- `protocols/v017/exp301_execution_identity_v1.sha256`

Any model, workflow, analysis, threshold, generator, training, documentation, or test change after the selected pre-marker source commit requires a new scientifically distinct freeze identity. It may not be smuggled into the marker commit.

## 2. Frozen scientific constitution

Current EXP-301 preregistration authority is V2 with semantic digest:

`1660a728990290f1c605941d531be1d52fe82591c619a327d104e4b87c1e6bad`

Resident trainable parameter budget is exactly 10,000,000 for each scientific arm:

- frozen base: 9,120,832;
- Capacity Exchange Envelope: 879,168;
- A_FIXED: 10,000,000 active trainable parameters;
- B_LOOP_SIMPLE: 10,000,000 active trainable parameters;
- C_NRS_CORE: 10,000,000 active trainable parameters;
- dead/filler capacity is forbidden.

The frozen scientific roots are `(0, 1, 2, 3)`. Primary trained efforts are `(1, 2, 4, 8)`. Challenge effort diagnostics are `(1, 2, 4, 8, 12, 16)`; efforts 12/16 cannot independently satisfy promotion.

Per root and per task family:

- training: 128 instances;
- development: 64 instances;
- challenge: 128 instances;
- four task families;
- one training epoch;
- batch size 1;
- 512 optimizer steps per trial;
- exactly two learning-rate trials per arm;
- six scientific trials per root;
- maximum sequence length 512;
- scientific decode budget exactly 96 generated tokens per arm/effort/example;
- bootstrap sample count exactly 10,000.

No public scientific API or workflow exposes learning rate, loop count, thresholds, task weights, sample counts, bootstrap samples, generation length, or other post-freeze tuning controls.

## 3. Causal execution order

Each root court is required to execute in this order:

1. validate frozen preregistration and scientific-execution contract before training;
2. run exactly six frozen train/development trials;
3. select one checkpoint per arm from development evidence only;
4. persist the selection manifest write-once;
5. reload and cryptographically verify selected checkpoints;
6. materialize challenge data only from a post-freeze workflow beacon;
7. build the runtime root identity;
8. commit the complete prediction grid before verifier scoring;
9. verify the complete grid;
10. build and write the root scientific evidence artifact write-once.

Cross-root analysis is allowed only after all four root artifacts are independently audited. No root may be omitted because its result is unfavorable.

## 4. Evidence and audit boundaries

Root evidence is self-auditing without the neural runtime. The auditor rematerializes challenge worlds from the frozen implementation plus post-freeze beacon, validates every commitment, reconstructs runtime identity, and recomputes verifier rows.

Cross-root evidence requires exactly roots 0–3, a single frozen implementation digest, unique run identities, 10,000 bootstrap samples, and the frozen EXP-301 reducer. EXP-301 can never authorize implementation or scaling directly. A positive result can authorize only EXP-302 design/preregistration.

Normal CI and TEST-ONLY execution remain scientifically ineligible.

## 5. Reproducible freeze identity

The freeze identity is not hand-authored. The builder binds:

- the exact pre-marker source commit SHA;
- its Git tree object;
- preregistration V2 digest;
- canonical architecture receipts;
- exact parameter constitution;
- scientific execution contract digest;
- train/development/challenge generator contracts derived from the frozen `exp301_worlds.py` Git blob;
- compute-ledger version;
- analysis/evidence/reducer Git blobs;
- the scientific workflow Git blob.

The challenge contract at freeze time contains only generator identity and the requirement for a post-freeze beacon. It contains no challenge nonce and materializes no challenge example.

`verify_exp301_freeze.py` must reconstruct this identity from Git history. The marker commit must have the frozen source commit as its direct parent and its complete diff must contain exactly the two marker files.

## 6. Resource audit

The preregistered court is deliberately nontrivial. Across four roots it contains:

- 24 scientific training trials;
- 12,288 optimizer steps total;
- 2,048 held-out challenge worlds total;
- 3 arms;
- 6 effort points;
- a fixed 96-token scientific decode budget;
- 3,538,944 autoregressive generation-token steps before multiplying by recurrent/restart effort;
- approximately 25,362,432 recurrent/restart block-passes when weighted by the frozen effort sum `1+2+4+8+12+16 = 43`, before accounting for sequence-length-dependent attention cost.

The current scientific workflow targets GitHub `ubuntu-latest`, which is a CPU runner in the observed CI environment. This creates a serious execution-resource risk. It is **not** scientific permission to shrink roots, examples, training steps, decode length, effort points, bootstrap samples, or model size after freeze.

If the authoritative scientific workflow cannot complete on the available runner class, the disposition is `RESOURCE_BLOCKED_BEFORE_SCIENTIFIC_OUTCOME`. The correct next action is to provide a suitable execution resource while preserving the frozen contract, not to weaken the court.

## 7. Pre-marker gates

Before the marker-only freeze commit is created, the exact pre-marker source head must have fresh evidence for all of the following:

- core Python 3.11 GREEN;
- core Python 3.13 GREEN;
- full model-smoke GREEN;
- dedicated V0.17 EXP-301 contract GREEN;
- TEST-ONLY EXP-301 smoke GREEN;
- freeze-builder contract GREEN;
- scientific workflow static isolation/ceremony contract GREEN;
- compileall GREEN.

The marker commit must then be independently verified as marker-only. No scientific workflow should be dispatched merely because the marker exists; resource readiness is a separate prerequisite.

## 8. Scientific decision rule remains unchanged

Promotion requires the complete preregistered conjunction: protected floors, aggregate RCG at least +0.05 versus A_FIXED, at least +5 percentage points verified success on at least two reasoning families at one common compute budget, language-control regression no worse than 2 percentage points, positive direction on at least 3/4 roots, and a 95% paired root-stratified hierarchical-bootstrap interval excluding zero.

A valid negative result kills `H-RD-01`. No threshold, root, task-weight, loop schedule, sample count, bootstrap count, or compute-accounting rescue is authorized after outcome inspection.

## 9. Current authority

At this pre-marker stage:

- implementation machinery is eligible for final exact-head verification;
- challenge data remains unmaterialized;
- no scientific result exists;
- EXP-302+ remains blocked;
- 30M/100M scaling remains blocked;
- architecture performance remains unverified.

This document is a pre-data execution closure, not a capability claim.
