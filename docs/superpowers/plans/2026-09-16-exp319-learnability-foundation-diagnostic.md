# EXP-319 Learnability Foundation Diagnostic Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task, and superpowers:test-driven-development for every implementation task.

**Goal:** Implement a scientifically distinct EXP-319 diagnostic that determines whether the unchanged 10M supervised learning stack can (A) fit a tiny sanity distribution, (B) reach a preregistered IID capability floor, and (C) retain nonzero competence on fresh generator-heldout data, without rescuing EXP-301 H-RD-01 or authorizing scale.

**Architecture:** Reuse the existing 10M arm implementations and byte-level training primitives, but create a new EXP-319 namespace for contracts, fresh generators, metrics, resumable checkpoint chains, selection, heldout challenge sealing, provenance, evidence, reducer logic, scripts, and workflow. Scientific evaluation uses fixed `effort=4`; Stage A is inspectable tuning, while Stages B/C are confirmatory. Execution is chunked for standard GitHub `ubuntu-latest` CPU runners and every continuation is cryptographically bound to its exact parent.

**Tech Stack:** Python 3.10–3.13, PyTorch, dataclasses/JSON/SHA-256, pytest, GitHub Actions (`ubuntu-latest`), and existing `nolane_ai.experiments.exp301_*` arm/tokenizer primitives only where explicitly allowed by the design.

**Scientific authority:** `docs/superpowers/specs/2026-09-16-exp319-learnability-foundation-diagnostic-design.md`. If this plan and that design disagree, stop and resolve the discrepancy before implementation. EXP-301 remains `KILL_H_RD_01`; `exp302_implementation_authorized=false`, `exp320_implementation_authorized=false`, and all scale authorization remains false.

---

## Task 1 — Freeze the machine-readable EXP-319 contract before scientific code

**Files:**
- Create: `protocols/v017/exp319_preregistration_v1.json`
- Create: `protocols/v017/exp319_preregistration_v1.sha256`
- Create: `src/nolane_ai/experiments/exp319_contract.py`
- Test: `tests/test_exp319_contract.py`

**Step 1: Write failing contract tests.**

Assert exactly: EXP-319-specific schema; common gating effort `4`; Stage A root `0`, 32 examples, checkpoints `128/512/1024`, LRs `1e-4/3e-4`; Stage B roots `1/2/3/4`, 512 train + 256 IID dev per root, checkpoints `512/1024/2048`; Stage C 512 heldout examples/root and zero gradient updates; all eight dispositions; all authorization flags false; Stage A/B chunk ceilings `256/512`.

Run: `python -m pytest tests/test_exp319_contract.py -q`

Expected: RED because the module/protocol do not exist.

**Step 2: Implement the smallest typed contract layer.** Use dataclasses/constants only. Add canonical JSON hashing. Put every threshold and tie-break rule from the approved design into the preregistration, including control-first Stage C ordering.

**Step 3: Generate and verify the sidecar digest.** The `.sha256` binds exact canonical JSON bytes. Test that any threshold mutation changes the digest.

**Step 4: Run:** `python -m pytest tests/test_exp319_contract.py -q` → GREEN.

**Step 5: Commit:** `git commit -m "test(exp319): freeze learnability diagnostic contract"`

---

## Task 2 — Build a fresh EXP-319 world namespace and leakage barriers

**Files:**
- Create: `src/nolane_ai/experiments/exp319_worlds.py`
- Test: `tests/test_exp319_worlds.py`
- Reference only: `src/nolane_ai/experiments/exp301_worlds.py`

**Step 1: Write failing generator tests.** Cover: Stage A/B/C use `exp319-*` content namespaces; Stage A is exactly 8/family; Stage B train/IID is root-bound and disjoint from A; Stage C requires run beacon + sealed Stage B selection digest; Stage C generator/template identities differ from Stage B rather than merely seed changes; content IDs bind stage/root/family/template/index and Stage C beacon; deterministic rematerialization; EXP-301 challenge IDs/nonces are rejected; persisted dataclasses survive JSON round-trip without tuple/list drift.

Run: `python -m pytest tests/test_exp319_worlds.py -q` → RED.

**Step 2: Implement four bounded family generators.** Implement the approved complexity ladder for iterative state/grid, algorithmic sequence transform, abstract rule transform, and language/sequence control. Reuse verifier semantics only where identical; never reuse EXP-301 challenge materializations.

**Step 3: Implement canonical generator manifests/digests** for A, B train/IID, C heldout. Persist JSON-native structures.

**Step 4: Run:** `python -m pytest tests/test_exp319_worlds.py tests/test_exp301_worlds.py -q` → GREEN.

**Step 5: Commit:** `git commit -m "feat(exp319): add fresh staged diagnostic worlds"`

---

## Task 3 — Implement diagnostic metrics independently of selection

**Files:**
- Create: `src/nolane_ai/experiments/exp319_metrics.py`
- Test: `tests/test_exp319_metrics.py`
- Reference: `src/nolane_ai/experiments/exp301_training.py`
- Reference: `src/nolane_ai/experiments/exp301_scientific.py`

**Step 1: Write failing tests** for answer-only loss including EOS, answer-token accuracy including EOS and excluding prompt targets, EOS correctness, invalid-output rate, exact/family-balanced exact match, pre-clip global gradient norm, parameter-update norm ratio, NaN/Inf detection, and fixed effort-4/96-token gating generation.

Use tiny deterministic tensors/models; no 10M model in unit tests.

Run: `python -m pytest tests/test_exp319_metrics.py -q` → RED.

**Step 2: Implement pure metric functions.** No threshold decisions here; thresholds come only from `exp319_contract.py`.

**Step 3: Run:** `python -m pytest tests/test_exp319_metrics.py tests/test_exp301_training.py -q` → GREEN.

**Step 4: Commit:** `git commit -m "feat(exp319): add learnability diagnostic metrics"`

---

## Task 4 — Build immutable resumable training-chain primitives

**Files:**
- Create: `src/nolane_ai/experiments/exp319_training.py`
- Test: `tests/test_exp319_training.py`
- Reference: `src/nolane_ai/experiments/exp301_scientific.py`
- Reference: `src/nolane_ai/experiments/exp301_arms.py`

**Step 1: Write failing continuation-chain tests.** A checkpoint receipt must bind stage, run/source identity, arm/root/LR, model-init seed, cumulative step, exact chunk interval, data-order/cursor digest, model-state digest, optimizer-state digest, RNG-state digest, parent artifact digest, and training-contract digest. Reject cross-root/arm/LR/run/source continuation, skipped/overlapping steps, chunks over 256/512, altered cursor/order, bad parent digest, and malformed JSON round-trips. Prove Stage B cannot consume Stage A weights.

Run: `python -m pytest tests/test_exp319_training.py -q` → RED.

**Step 2: Implement checkpoint-chain schema and hashing.** Persist model + optimizer state and a separate canonical receipt; write once.

**Step 3: Implement bounded chunk execution.** Reuse `build_scientific_arm`, `Exp301ByteTokenizer`, answer-only loss semantics, AdamW weight decay `0.01`, gradient clip `1.0`, and the `{1,2,4,8}` training effort cycle. Do not change arm geometry.

**Step 4: Add snapshot evaluation.** A snapshots at 128/512/1024; B snapshots at 512/1024/2048; all gating evaluation effort 4.

**Step 5: Run:** `python -m pytest tests/test_exp319_training.py tests/test_exp301_scientific.py tests/test_exp301_arms.py -q` → GREEN.

**Step 6: Commit:** `git commit -m "feat(exp319): add immutable resumable training chains"`

---

## Task 5 — Freeze Stage A LR selection and Stage B checkpoint selection

**Files:**
- Create: `src/nolane_ai/experiments/exp319_selection.py`
- Test: `tests/test_exp319_selection.py`

**Step 1: Write failing tests.** Stage A LR tie order: higher step-1024 exact, higher token accuracy, lower loss, lower LR. Stage B checkpoint: earliest passing all floors; otherwise higher IID family-balanced exact, higher IID token accuracy, lower loss, earlier step. Reject missing required checkpoints/roots.

Run: `python -m pytest tests/test_exp319_selection.py -q` → RED.

**Step 2: Implement deterministic pure selectors** over completed receipts/metrics only.

**Step 3: Run:** `python -m pytest tests/test_exp319_selection.py tests/test_exp319_contract.py -q` → GREEN.

**Step 4: Commit:** `git commit -m "feat(exp319): freeze diagnostic selection rules"`

---

## Task 6 — Implement Stage C post-selection heldout ceremony and commitments

**Files:**
- Create: `src/nolane_ai/experiments/exp319_challenge.py`
- Test: `tests/test_exp319_challenge.py`
- Reference: `src/nolane_ai/experiments/exp301_ceremony.py`
- Reference: `src/nolane_ai/experiments/exp301_scientific_executor.py`

**Step 1: Write failing ceremony tests.** No heldout manifest before all required B selections are sealed; manifest binds beacon/source/generator/selection digests + roots; exactly 512 heldout worlds/root; only A_FIXED/C_NRS_CORE are primary; effort exactly 4; no gradient API in C; commitments contain predictions/provenance only; scoring rejects incomplete/duplicate grids; shards cover each root exactly once.

Run: `python -m pytest tests/test_exp319_challenge.py -q` → RED.

**Step 2: Implement heldout manifest and beacon binding.** Same accepted run identity must reproduce the same manifest; a new run gets a new beacon/materialization.

**Step 3: Implement prediction commitments and merge.** Use 16 shards/root ×32 worlds; each shard evaluates both primary arms at effort 4. Root scoring requires exactly 16 unique shards.

**Step 4: Run:** `python -m pytest tests/test_exp319_challenge.py tests/test_exp301_scientific_executor.py -q` → GREEN.

**Step 5: Commit:** `git commit -m "feat(exp319): add sealed heldout challenge ceremony"`

---

## Task 7 — Build root evidence and the closed causal reducer

**Files:**
- Create: `src/nolane_ai/experiments/exp319_evidence.py`
- Test: `tests/test_exp319_evidence.py`

**Step 1: Write failing reducer fixtures** for every exact disposition: `INVALID_DIAGNOSTIC`, `TRAINING_STACK_NOT_LEARNABLE`, `NRS_LOCAL_TRAINABILITY_FAIL`, `SUPERVISED_FOUNDATION_UNDERTRAINED`, `NRS_IID_GENERALIZATION_FLOOR_FAIL`, `SUPERVISED_HELDOUT_FOUNDATION_FAIL`, `NRS_HELDOUT_GENERALIZATION_FLOOR_FAIL`, `FOUNDATION_READY_FOR_EXP320_DESIGN_ONLY`.

Test control-first Stage C ordering explicitly. Every disposition must have all authorization flags false.

Run: `python -m pytest tests/test_exp319_evidence.py -q` → RED.

**Step 2: Implement root evidence builders** with A selection, B chain/selection, C manifest/commitment/scoring, provenance digests and exact thresholds used.

**Step 3: Implement final reducer as a pure fail-closed function.** Provenance invalidity precedes capability interpretation.

**Step 4: Run:** `python -m pytest tests/test_exp319_evidence.py tests/test_exp319_contract.py -q` → GREEN.

**Step 5: Commit:** `git commit -m "feat(exp319): add closed diagnostic reducer"`

---

## Task 8 — Add source/execution identity and freeze verifier

**Files:**
- Create: `src/nolane_ai/experiments/exp319_identity.py`
- Create: `src/nolane_ai/experiments/exp319_freeze.py`
- Create: `tests/test_exp319_identity.py`
- Create: `tests/test_exp319_freeze.py`
- Create only after source/workflow freeze: `protocols/v017/exp319_execution_identity_v1.json`
- Create only after freeze: `protocols/v017/exp319_execution_identity_v1.sha256`

**Step 1: Write failing identity tests.** Bind approved prereg digest, historical EXP-301 final cross-root artifact digest, EXP-301R repair provenance digest, exact source-tree digest, generator/training/selection/scoring digests, workflow SHA-256, device CPU, and Stage C shard geometry 16×32.

**Step 2: Write freeze allowlist tests.** Fail if unrelated scientific files change after freeze; permit only explicitly enumerated EXP-319 files and marker identity files.

Run: `python -m pytest tests/test_exp319_identity.py tests/test_exp319_freeze.py -q` → RED.

**Step 3: Implement identity/freeze modules.** Canonical JSON only; validate sidecar on read; never compare persisted tuple/list Python shapes.

**Step 4: Run:** `python -m pytest tests/test_exp319_identity.py tests/test_exp319_freeze.py -q` → GREEN.

**Step 5: Commit source identity machinery only:** `git commit -m "feat(exp319): add execution identity and freeze contracts"`

---

## Task 9 — Add narrow CLIs with no scientific tuning flags

**Files:**
- Create: `scripts/verify_exp319_contract.py`
- Create: `scripts/exp319_stage_a_chunk.py`
- Create: `scripts/exp319_stage_a_select.py`
- Create: `scripts/exp319_stage_b_chunk.py`
- Create: `scripts/exp319_stage_b_select.py`
- Create: `scripts/exp319_materialize_heldout.py`
- Create: `scripts/exp319_predict_heldout_shard.py`
- Create: `scripts/exp319_seal_root.py`
- Create: `scripts/exp319_finalize.py`
- Create: `scripts/verify_exp319_freeze.py`
- Test: `tests/test_exp319_cli.py`

**Step 1: Write failing CLI tests.** Allow only identity/IO coordinates already preregistered: arm/root/LR, chunk/shard index, paths, run identity, fixed CPU device where applicable. Forbid threshold, effort, optimizer budget, generator size, roots, tokenizer, or scoring overrides.

Run: `python -m pytest tests/test_exp319_cli.py -q` → RED.

**Step 2: Implement thin CLIs over library functions.** Scientific logic remains in the library. All outputs are write-once canonical receipts/evidence.

**Step 3: Run:** `python -m pytest tests/test_exp319_cli.py tests/test_exp319_*.py -q` → GREEN.

**Step 4: Commit:** `git commit -m "feat(exp319): add sealed diagnostic command surfaces"`

---

## Task 10 — Build the standard-runner workflow as immutable stages

**Files:**
- Create: `.github/workflows/exp319-learnability-foundation-diagnostic.yml`
- Create: `.github/workflows/v017-exp319-contract.yml`
- Test: `tests/test_exp319_workflow.py`

**Step 1: Write failing workflow-contract tests.** Require `workflow_dispatch`, `ubuntu-latest`, no larger/self-hosted runner, no scientific tuning inputs, preflight prereg/freeze/identity verification, A = six chains × four sequential 256-step chunks, A selection only after all six chains, B = eight fresh-init chains × four sequential 512-step chunks, valid negative B stops finalized rather than workflow-failed, C only after both B cross-root gates, C = 4 roots ×16 shards, four root sealers, one reducer/finalizer, artifact coordinates bound in names, final authorization verification false.

Run: `python -m pytest tests/test_exp319_workflow.py -q` → RED.

**Step 2: Implement `v017-exp319-contract.yml`** for fast PR contract/freeze tests.

**Step 3: Implement scientific workflow.** Use explicit chunk stages. Each later chunk downloads and validates exactly one parent artifact for the same matrix coordinate. Stage C stores commitments first and scores only after 16/16 root shards.

**Step 4: Run:** `python -m pytest tests/test_exp319_workflow.py -q` → GREEN.

**Step 5: Commit:** `git commit -m "ci(exp319): add standard-runner staged diagnostic court"`

---

## Task 11 — Full pre-freeze verification and execution-identity seal

**Files:**
- Create now, and only now: `protocols/v017/exp319_execution_identity_v1.json`
- Create: `protocols/v017/exp319_execution_identity_v1.sha256`

**Step 1:** `python -m pytest tests/test_exp319_*.py -q` → GREEN.

**Step 2:** inherited regressions: `python -m pytest tests/test_exp301_arms.py tests/test_exp301_training.py tests/test_exp301_scientific.py tests/test_exp301_worlds.py tests/test_exp301r_recovery.py -q` → GREEN.

**Step 3:** run the repository’s configured pre-commit/lint/type/test gates exactly as `ci.yml` defines. No reduced substitute is acceptable for freeze readiness.

**Step 4:** compute final source-tree, workflow, prereg and contract digests. Create execution identity with no result fields and a matching canonical SHA sidecar.

**Step 5:** marker-only seal. The marker commit adds exactly the two execution-identity files. Verify `git diff --name-only HEAD^ HEAD` plus exact source/tree digest bindings.

**Step 6: Commit:** `git commit -m "chore(exp319): seal diagnostic execution identity"`

---

## Task 12 — Human review gate before scientific execution

Update the EXP-319 PR with exact source commit, marker commit, source-tree digest, prereg digest, workflow digest, execution-identity digest, exact CI run IDs/conclusions, EXP-301=`KILL_H_RD_01`, and all implementation/scale authorization flags false.

Then stop. Do not dispatch the scientific workflow in the same step that seals it. Scientific execution requires a separate explicit approval.

---

## Task 13 — Scientific execution procedure after explicit run approval

1. Register the sealed workflow on default branch only if GitHub requires it, byte-identical to the reviewed workflow.
2. Dispatch exactly one authoritative run on the sealed marker.
3. Verify preflight before interpreting artifacts.
4. Stage A is inspectable; seal LR selections before B.
5. During B inspect operational metadata only; never alter the court from metrics.
6. A valid B stop disposition finalizes without C.
7. If B authorizes C under frozen rules, materialize heldout only after selections seal.
8. Do not inspect partial C commitments/scores for tuning.
9. Require 16/16 shards/root before root scoring.
10. Final reducer emits one of the eight frozen dispositions with every authorization flag false.
11. Independently verify canonical digests for final/root evidence, manifests, receipts and execution identity before claiming completion.
12. Update provenance comments; do not merge or scale automatically.

---

## Verification checklist before saying “EXP-319 implementation complete”

- Approved design remains the governing authority.
- Prereg JSON and SHA sidecar match.
- All persisted objects survive JSON round-trip and canonical digest reproduction.
- 10M arm parameter counts remain unchanged.
- Gating inference effort is exactly 4 everywhere.
- Stage A/B chunk ceilings are enforced by code and workflow.
- Stage B fresh initialization is proven by tests/receipts.
- Stage C cannot materialize before Stage B selection seal and has no training path.
- Reducer has exactly eight dispositions with control-first ordering.
- All authorization flags are false for every disposition.
- Full EXP-319 tests GREEN.
- Required EXP-301/EXP-301R regressions GREEN.
- Repository CI GREEN on the exact source head.
- Marker commit changes exactly two execution-identity files.
- No scientific run dispatched before separate approval.

## Non-goals / hard prohibitions

Do not implement EXP-302, EXP-320, online plasticity, THINK/HUMAN/TOOL routing, 30M, or 100M in this plan. Do not reinterpret EXP-301’s negative result. Do not add a third LR, extra roots, extra training budget, alternative gating effort, or post-hoc threshold adjustment after freeze. Any discovered correctness bug requires a separately documented repair lineage and re-seal before scientific execution.
