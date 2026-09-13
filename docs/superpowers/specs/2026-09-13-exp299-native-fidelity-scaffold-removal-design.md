# EXP-299 Native Fidelity Scaffold-Removal Court Design

## Authority

EXP-299 / `H-NATIVE-01` is authorized only by the sealed EXP-298 DEVELOPMENT disposition `CROSS_DOMAIN_FIDELITY_TRANSFER_RECURRENT`, whose scope is `DESIGN_EXP299_SCAFFOLD_REMOVAL_COURT_ONLY`. The EXP-298 scientific branch remains sealed and unmerged. This EXP-299 lineage starts from the exact EXP-298 pre-data implementation substrate `68e1205fd56e5b1acb8f02cea56fc9af1a881734`, before any EXP-298 release marker existed, so the reusable deterministic domain/court machinery is inherited without mutating the sealed EXP-298 branch. EXP-300 remains unauthorized.

The frozen Stage-A protocol is unchanged and this lane MUST NOT modify `protocols/stage_a_v1.json` or `protocols/stage_a_v1.sha256`.

## Question

After learning with the EXP-298 behavioral-fidelity scaffold on controlled fit data, does a matched neural model retain a material cross-domain fidelity/authority advantage when the scaffold is removed completely from causal inference on held-out candidates?

This is an internalization/scaffold-removal court. It does not establish unrestricted semantic understanding, arbitrary program verification, causal discovery, open-language reasoning, durable lifelong consolidation, or integrated V0.16 superiority.

## Core causal contrast

EXP-299 uses two matched neural arms with byte-identical initialization and geometry:

1. `null_scaffold_control`: fit and evaluate with the canonical null receipt only.
2. `scaffold_then_native`: fit with the public EXP-298 behavioral-fidelity receipt, then evaluate after that receipt channel is forcibly zeroed and inaccessible.

During held-out evaluation both arms receive exactly the same causal inputs: source digest, candidate digest, executable bit, and canonical null receipt vector. The exact behavioral court remains evaluator-only during held-out evaluation and cannot affect either arm's action or authority output.

The only intended causal difference is fit-time scaffold access.

## Domains

Reuse the frozen deterministic EXP-298 domain families and exact court semantics:

- `code_invariant`
- `causal_diagnosis`
- `grounded_language_ambiguity`

The frozen candidate strata remain unchanged. Evaluation labels (`is_faithful`, `stratum`) are evaluator-only. No evaluation candidate, exact-court decision, witness, or stratum may enter fit data or model causal inputs.

## Data partition and independence

For each `(root, domain)` create two deterministic non-overlapping partitions:

- fit: 96 replicates, 16 candidates/replicate;
- held-out evaluation: 32 replicates, 16 candidates/replicate.

Fit and evaluation use domain-separated RNG labels and independently generated surface randomization. Candidate digests must be disjoint across partitions. Evaluation truth/court receipts are generated only after model fit is complete and are consumed only by the evaluator.

Each root therefore evaluates 1,536 held-out candidates and 4 roots evaluate 6,144 held-out candidates.

## Model and fit contract

Both arms reuse the EXP-298 matched geometry:

- `d_model=64`
- `hidden_size=64`
- `target_parameters=500000`

The fit objective is fixed binary fidelity prediction from the arm's neural authority/verifier logits. The target is derived from fit-time exact-court outcome (`court_accept` = faithful, all exact rejects = wrong). `court_inconclusive` examples are excluded from gradient targets and their count must be reported; no root may pass if any fit or evaluation candidate is inconclusive under the frozen exact-probe bound.

Both arms see the same fit examples in the same order, use the same optimizer family, steps, batch ordering, learning rate, gradient clipping, and seed. Only `scaffold_then_native` receives the public receipt vector during fit. The control receives the canonical null vector.

Frozen optimizer geometry:

- AdamW
- learning rate `3e-4`
- weight decay `1e-4`
- one pass over the frozen fit set
- batch size `32`
- gradient clip norm `1.0`

No threshold sweep, extra epochs, learning-rate tuning, root replacement, post-result domain removal, or sample escalation is permitted after one-shot release.

## Native evaluation rule

After fit:

1. snapshot both model-state digests;
2. disable gradients;
3. force the receipt adapter input to the canonical all-zero null vector for both arms;
4. prohibit any call path that passes a `BehavioralFidelityReceipt` into either arm;
5. run the held-out candidate set once;
6. compare neural fidelity/authority output against evaluator-only exact court truth.

The scaffolded arm is not allowed to use exact-court acceptance directly to grant authority in EXP-299 evaluation. Its authority decision is neural and uses a frozen threshold `0.5` on `authority_score`. The same threshold is used by the control.

The evaluator separately records wrong-authority and faithful-rejection rates from exact held-out truth.

## Primary and protected endpoints

For each domain `d`:

- `BA_native[d]`: balanced accuracy of `scaffold_then_native` after receipt removal;
- `BA_control[d]`: balanced accuracy of `null_scaffold_control`;
- `native_gain[d] = BA_native[d] - BA_control[d]`;
- `wrong_authority_native[d]`;
- `faithful_rejection_native[d]`.

Frozen V1 success thresholds:

- `BA_native[d] >= 0.70` for all three domains;
- `native_gain[d] >= 0.10` for all three domains;
- `wrong_authority_native[d] <= 0.05` for all three domains;
- `faithful_rejection_native[d] <= 0.10` for all three domains.

The root primary endpoint is `min_d native_gain[d]`. Safety endpoints are conjunctive and cannot be traded against gain.

## Root decision

A canonical root returns `NATIVE_FIDELITY_SURVIVES_SCAFFOLD_REMOVAL` only when:

- all three domains meet every frozen primary/protected threshold;
- fit/eval candidate sets are proven disjoint;
- both arms were matched at initialization and optimizer geometry;
- held-out evaluation consumed no non-null receipt in either arm;
- model state is unchanged during held-out evaluation;
- raw rows reconstruct every aggregate exactly;
- evaluator labels/court receipts are absent from model causal inputs;
- all evidence-boundary flags remain false.

Otherwise it returns `NATIVE_FIDELITY_NOT_ESTABLISHED`.

## Cross-root decision and successor authority

The cross reducer consumes exactly roots `[0,1,2,3]`.

Only 4/4 roots with `NATIVE_FIDELITY_SURVIVES_SCAFFOLD_REMOVAL` may return:

`NATIVE_FIDELITY_SURVIVES_SCAFFOLD_REMOVAL_RECURRENT`

with:

- `successor_design_authorized=true`
- `authorization_scope=DESIGN_EXP300_INTEGRATED_COURT_ONLY`

Every other valid result returns `NATIVE_FIDELITY_NOT_ESTABLISHED_RECURRENT` with successor authorization false/NONE. EXP-299 may authorize design of EXP-300 only; it cannot itself authorize an EXP-300 run or any broad NLM claim.

## Portability hardening inherited from EXP-298 incident

EXP-298 exposed a cross-process byte-reconstruction portability defect after all root receipts had already self-validated. EXP-299 therefore freezes an explicit validator portability rule before data:

- canonical receipts may contain only JSON-stable scalar/list/map values with string map keys;
- no process-local object identity, storage address, unordered set, Python hash randomization, or device-dependent raw storage representation may enter deterministic reconstruction;
- root validators must be executed in both producer process and a fresh subprocess in pre-data CI on synthetic fixture roots;
- cross reducer validates sealed receipt bytes/digests and deterministic semantic recomputation, but does not require reconstruction through process-unstable representations.

This is pre-data hardening for a new experiment, not a reinterpretation or repair of the sealed EXP-298 result.

## Cost accounting

Report per arm/domain/root:

- training neural FLOPs proxy;
- evaluation neural FLOPs proxy;
- fit-time semantic probe operations;
- held-out evaluator semantic probe operations;
- total accounted cost proxy;
- `hardware_profiler_flops_claimed=false`.

Held-out exact-court evaluator cost is charged but is not a causal model input.

## Artifacts and schemas

Root schema: `NLM-EXP-299-NATIVE-FIDELITY-ROOT-V1`.

Cross schema: `NLM-EXP-299-NATIVE-FIDELITY-CROSS-V1`.

Canonical JSON uses SHA-256 sidecars plus internal `artifact_digest` computed with that field excluded. Root receipts publish fit/eval partition digests, model init/post-fit/post-eval state digests, causal-input audit, raw-row digest, aggregate metrics, cost ledger, and decision.

## Evidence boundary

EXP-299 V1 is DEVELOPMENT / EV-E2 only. Root and cross receipts must keep false:

- `scientific_evidence_eligible`
- `confirmatory_data_consumed`
- `challenge_materialized`
- `promotion_claimed`
- `unrestricted_semantic_authority_claimed`
- `open_language_understanding_claimed`
- `causal_discovery_claimed`
- `general_code_reasoning_claimed`
- `durable_lifelong_internalization_claimed`
- `exp300_execution_authorized`

## Release discipline

Design/plan → test-first implementation → dedicated + generic exact-head CI → fresh-process portability gate → zero-run freeze guard → marker-only release → exactly one four-root DEVELOPMENT run → cross reducer → duplicate-run guard → independent artifact audit → close experiment PR unmerged → integrate only documentation/closure/program-matrix evidence to `main` unless separately authorized.

A negative result closes this frozen V1 hypothesis. No same-hypothesis tuning/rerun is authorized after held-out visibility.