# EXP-290 Structural Clause Transfer Court Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a preregistered DEVELOPMENT / EV-E2 court testing whether a learned source→target clause translator can reuse observed one-literal nogoods across non-identity variable surface permutations while preserving EXP-289 safety floors.

**Architecture:** Generate paired source/target views of one latent EXP-289-style binary problem, run one common causal source phase, and evaluate one frozen model in `LOCAL_ONLY_CONTROL`, `LEARNED_STRUCTURAL_TRANSFER`, and `ORACLE_ISOMORPHIC_TRANSFER_UPPER_BOUND`. Every target mode keeps exact episode-local target memory; only pre-target source transfer differs. Root and cross reducers recompute all decisions from primitive receipts and keep scientific/confirmatory authority closed.

**Tech Stack:** Python 3.11/3.13, PyTorch CPU, pytest, repository seed/protocol/evidence helpers, canonical JSON/SHA-256 receipts, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-13-exp290-structural-clause-transfer-design.md`

## Global Constraints

- Base authority is `main@e7a034365e542895b772701a247f6ad9b931bfc9`.
- `protocols/stage_a_v1.json` and digest `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440` remain byte-unchanged.
- Evidence level is DEVELOPMENT / EV-E2 only; no challenge/confirmatory entropy may be materialized.
- Four decision roots: `0,1,2,3`.
- Geometry: train64/eval32, eval start 40000, batch8, d64/h48/500k, timesteps4, variables8, decoys3, restarts4, search24, noise .05, AdamW lr .002/wd0.
- Held-out source→target variable permutation must be non-identity on every pair; value labels remain `{0,1}` and are not remapped in V1.
- All three target modes execute and charge learned transfer scoring; model state must remain byte-identical across modes.
- Target exact episode-local nogood memory is enabled in every mode.
- Learned transfer is deterministic top-1 target-variable translation with ascending-index tie break and no threshold/calibration.
- Root success requires positive fresh oracle headroom, positive learned headroom, capture >=0.50, learned over-prune <=0.005, solution rate >= control-0.01, positive learned transferred-prune count, non-identity surfaces, and zero evaluation/oracle mapping leakage.
- Only all-four-root recurrence may authorize `DESIGN_EXP291_ENCODING_COUNTEREXAMPLE_COURT_ONLY`.
- No held-out DEVELOPMENT execution before an exact-head all-GREEN pre-marker gate and marker-only release commit.

---

### Task 1: Frozen geometry, paired worlds, and matched transfer model

**Files:**
- Create: `protocols/exp290_structural_clause_transfer_v1.json`
- Create: `protocols/exp290_structural_clause_transfer_v1.sha256`
- Create: `src/nolane_ai/experiments/exp290_development_geometry.py`
- Create: `src/nolane_ai/experiments/exp290_transfer_worlds.py`
- Create: `src/nolane_ai/experiments/matched_clause_transfer_arms.py`
- Test: `tests/test_exp290_structural_clause_transfer.py`

**Interfaces:**
- `load_exp290_development_geometry(manifest_path, digest_path, *, protocol_digest) -> tuple[dict[str, Any], str]`
- `Exp290TransferGenerator(root_seed: str).make_pair(...) -> Exp290TransferPairBatch`
- `build_matched_exp290_model(...) -> Exp290ClauseTransferModel`
- `Exp290ClauseTransferModel.score_transfer(source_variable_states, target_variable_states, source_variable_index, literal_value) -> Tensor[batch,target_variables]`

- [ ] **Step 1: Write import/geometry RED tests.** Require exact frozen geometry, EV-E2-only authority, four roots, disjoint train/eval indices, parent safety floors, and unchanged Stage-A digest.
- [ ] **Step 2: Run `pytest -q tests/test_exp290_structural_clause_transfer.py` and verify failure is missing EXP-290 modules, not syntax/import mistakes.**
- [ ] **Step 3: Extend RED tests for paired worlds.** Require deterministic pair digest, identical latent problem digest, mandatory non-identity target permutation, source/target visible IDs differing, independent surface/event ordering, shared value alphabet, hidden mapping absent from model-visible tensors, and evaluator metadata marked not delivered.
- [ ] **Step 4: Extend RED tests for model geometry.** Require one shared 500k model, deterministic top-1 tie break, one score per target variable, same parameter/active-functional/optimizer-visible counts across target modes, and transfer scorer callable without hidden mapping.
- [ ] **Step 5: Commit test-only RED.**
- [ ] **Step 6: Implement geometry loader plus canonical manifest and digest.** Loader rejects digest/protocol/authority/threshold/root/geometry drift and any challenge/beacon field.
- [ ] **Step 7: Implement paired generator.** Derive the permitted augmentation/evaluation stream seed with `derive_stream_seed(..., "EXP-290", ...)`, then domain-separate deterministic source/target sub-seeds with SHA-256; resample/rotate target permutation deterministically if identity is drawn.
- [ ] **Step 8: Implement transfer model.** Reuse repository event/variable/GRU patterns; encode source literal from source variable representation + fixed binary-value embedding; score all target variable representations; keep branch/verifier/local-memory substrate in the same model and close to the 500k target using `finalize_region_budget`.
- [ ] **Step 9: Run focused tests, `python scripts/verify_protocol.py`, and `python -m compileall -q src scripts`; commit GREEN.**

### Task 2: Common source phase and same-weights target court

**Files:**
- Create: `src/nolane_ai/experiments/exp290_structural_clause_transfer.py`
- Extend: `tests/test_exp290_structural_clause_transfer.py`

**Interfaces:**
- `run_exp290_root(*, canonical_index: int, geometry: Mapping[str, Any], protocol_digest: str, geometry_digest: str, code_digest: str) -> dict[str, Any]`
- internal common source phase returns sealed source clauses plus evaluator-only target opportunity manifest digest.
- target evaluator returns per-mode episode receipts and pooled primitive metrics.

- [ ] **Step 1: Write RED tests for common source causality.** Source clauses must be inserted only after reached public contradictions, must be identical across target modes, and must not use valid-solution/evaluator/future-target truth.
- [ ] **Step 2: Write RED tests for the three target modes.** All modes keep target-local exact memory and execute transfer scorer; control ignores translations, learned uses deterministic top-1 predictions, oracle uses hidden mapping only after scorer computation and is explicitly non-deployable.
- [ ] **Step 3: Write RED leakage tests.** Evaluation mapping labels may appear only in evaluator/oracle/post-hoc receipts, never learned action inputs or training; model digest before/after all modes must match exactly.
- [ ] **Step 4: Write RED accounting tests.** Transferred memory uses fixed source-clause slots; duplicates do not reduce charged comparison count; local-memory canonicalization/insertion/comparison and neural/transfer compute are charged.
- [ ] **Step 5: Write RED training tests.** One model/root, augmentation pairs only, fixed loss `branch CE + verifier BCE + transfer-mapping CE` with coefficients 1/1/1, no calibration/threshold/class weights/focal/early stop/sweep hooks.
- [ ] **Step 6: Verify RED and commit.**
- [ ] **Step 7: Implement training, freeze model digest, common source execution, fixed-slot transfer memory, and target target-mode execution.**
- [ ] **Step 8: Implement primitive metrics.** Pool source-equivalent target dead-end numerator/denominator, `R0/RL/RO`, oracle/learned headroom, unclipped capture, over-prune, solution rate, translated-prune counts, mapping accuracy diagnostics, RDER, and accounted cost.
- [ ] **Step 9: Implement root classification exactly from the spec.** No favorable reinterpretation or threshold flexibility.
- [ ] **Step 10: Run focused tests + parent EXP-289 regression tests + protocol verifier + compile; commit GREEN.**

### Task 3: Fail-closed root receipts and cross-root reducer

**Files:**
- Create: `src/nolane_ai/experiments/exp290_receipts.py`
- Test: `tests/test_exp290_receipts.py`

**Interfaces:**
- `build_exp290_root_receipt(payload: Mapping[str, Any]) -> dict[str, Any]`
- `validate_exp290_root_receipt(receipt: Mapping[str, Any]) -> None`
- `reduce_exp290_cross_root(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]`
- `canonical_receipt_bytes(receipt) -> bytes`

- [ ] **Step 1: Write RED tests rejecting forged root classifications, mixed geometry/code/protocol, wrong canonical index/root prefix, model drift, train/eval overlap, identity held-out permutations, learned oracle/evaluator mapping leakage, unsafe ground-truth action gates, boundary overclaim, and noncanonical serialization.**
- [ ] **Step 2: Write RED cross tests requiring exactly roots 0/1/2/3 once each, recomputed root decisions, one common code/geometry/protocol identity, and exact recurrence/intermittent/not-established/oracle-incomplete logic.**
- [ ] **Step 3: Require only recurrent disposition to set `successor_design_authorized=true` and exact scope `DESIGN_EXP291_ENCODING_COUNTEREXAMPLE_COURT_ONLY`; every scientific/confirmatory/promotion flag remains false.**
- [ ] **Step 4: Verify RED and commit.**
- [ ] **Step 5: Implement canonical JSON, SHA helpers, root builder/validator, artifact digest, and cross reducer.**
- [ ] **Step 6: Run receipt tests + complete EXP-290 suite + protocol verifier + compile; commit GREEN.**

### Task 4: Frozen CLI and publication surface

**Files:**
- Create: `scripts/run_exp290_structural_clause_transfer_dev.py`
- Create: `scripts/reduce_exp290_structural_clause_transfer_dev.py`
- Test: `tests/test_exp290_cli.py`

**Interfaces:**
- root CLI accepts only canonical index and output directory/path; all scientific geometry comes from frozen manifest.
- cross CLI accepts exactly four root receipt paths and output path.

- [ ] **Step 1: Write RED CLI tests.** `--help` must expose no knobs for roots/sample sizes/surface randomization/top-k/loss weights/capture/safety thresholds/model geometry.
- [ ] **Step 2: Require atomic no-overwrite publication of `receipt.json` + matching `receipt.json.sha256`, validation before publication, and failure without partial output.**
- [ ] **Step 3: Verify RED and commit.**
- [ ] **Step 4: Implement root/cross CLIs with fixed geometry loading, source-tree code digest injection, canonical receipt publication, and fail-closed validation.**
- [ ] **Step 5: Run CLI + receipt + complete EXP-290 tests, protocol verifier, compile; commit GREEN.**

### Task 5: Exact-head CI, marker-only DEVELOPMENT workflow, and freeze guard

**Files:**
- Create: `.github/workflows/exp290-structural-clause-transfer-ci.yml`
- Create: `.github/workflows/exp290-structural-clause-transfer-development.yml`
- Create: `.github/workflows/exp290-structural-clause-transfer-freeze-guard.yml`
- Test: `tests/test_exp290_workflow_contract.py`

**Interfaces:**
- dedicated CI validates all EXP-290 tests/protocol/compile.
- DEVELOPMENT workflow exists before marker but executes held-out court only when the exact release-marker path is added at PR head.
- freeze guard audits first authoritative run and rejects duplicates.

- [ ] **Step 1: Write RED workflow tests requiring dedicated path-scoped CI, no `workflow_dispatch`, marker-only activation, exact-head checkout/binding, `GITHUB_RUN_ATTEMPT == 1`, per-PR concurrency with `cancel-in-progress:false`, preflight-before-data, four root jobs then one cross reducer, finite artifact retention, no secrets/challenge/confirmatory path.**
- [ ] **Step 2: Require workflows to verify `receipt.code_digest == source_tree_digest(exact checkout)` before upload and cross receipt to preserve the same digest.**
- [ ] **Step 3: Require prerelease sentinel that release marker is absent until all pre-data gates are GREEN.**
- [ ] **Step 4: Verify RED and commit.**
- [ ] **Step 5: Implement workflows without creating release marker.**
- [ ] **Step 6: Run exact-head dedicated CI, generic core3.11/core3.13/model-smoke, protocol verifier, compile, and freeze guard; require zero prior EXP-290 DEVELOPMENT runs.**
- [ ] **Step 7: Commit GREEN only when pre-marker exact-head evidence is complete.**

### Task 6: One-shot release, sealed DEVELOPMENT execution, audit, and closure

**Files:**
- Create only after pre-data GREEN: `protocols/exp290-structural-clause-transfer-release.lock`
- Update after sealed result: `docs/superpowers/specs/2026-09-13-v0161-program-closure-matrix-design.md`
- Add final closure record under `docs/superpowers/specs/` after disposition is known.

- [ ] **Step 1: Re-read exact pre-marker head and all verification runs.** Require dedicated EXP-290 GREEN, generic core3.11/core3.13/model-smoke GREEN, protocol digest unchanged, freeze guard zero-run state, and no marker.
- [ ] **Step 2: Create one marker-only commit binding pre-marker head, root set, frozen geometry, Stage-A digest, and decision constants.** Audit commit changes exactly one marker file.
- [ ] **Step 3: Observe exactly one authoritative DEVELOPMENT run and no duplicates.** Do not trigger rerun/replacement manually.
- [ ] **Step 4: Require preflight, 4/4 roots, cross reducer, receipt/sidecar verification, and artifact uploads to complete on that authoritative run.**
- [ ] **Step 5: Independently download all artifacts and recompute ZIP SHA, receipt sidecar SHA, canonical byte roundtrip, artifact digest, root decisions, cross decision, source-tree code identity, model-state equality, and boundary flags.**
- [ ] **Step 6: Update program matrix from the sealed outcome.** Recurrent may authorize EXP-291 design only; every weaker valid result leaves EXP-291 blocked on this path; oracle-incomplete is not mislabeled learned failure.
- [ ] **Step 7: Preserve a negative/non-authorizing experiment PR unmerged.** Reusable docs may be integrated separately; do not merge a falsified mechanism merely to retain code.
- [ ] **Step 8: Freshly re-read final PR state and `main` before any completion claim.**

## Self-review checklist

- Every spec section maps to a task above.
- No production EXP-290 code precedes a witnessed RED test.
- The plan contains no confirmatory/challenge execution path.
- The plan does not mutate frozen Stage-A bytes.
- Control retains target-local memory, so EXP-290 cannot win by re-proving EXP-289.
- Fixed transferred slots prevent learned duplicates from receiving cheaper memory accounting.
- Value-label remapping and cross-domain transfer remain explicit non-claims.
- EXP-291 authorization is design-only and recurrent-only.
