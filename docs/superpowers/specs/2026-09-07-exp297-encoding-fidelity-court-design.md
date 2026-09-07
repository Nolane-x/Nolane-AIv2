# EXP-297 Encoding Fidelity Court Design

## Status and authority

This design implements the already-approved DEVELOPMENT architecture for frozen Stage-A `EXP-297`. It MUST NOT modify `protocols/stage_a_v1.json` or `protocols/stage_a_v1.sha256`. The frozen scientific contrast remains `compile_only` versus `fidelity_court`, primary `semantic_fidelity_balanced_accuracy`, protected `wrong_formalization_authority_rate <= 0.05` and `faithful_formalization_rejection_rate <= 0.10`, and MESI `+0.10` balanced accuracy.

This lane is engineering evidence only: `EV-E2`, `UNVERIFIED`, `confirmatory_ready=false`, `confirmatory_data_consumed=false`, `challenge_seed_materialized=false`, `challenge_materialized=false`, `decision_rule_executed=false`, `hidden_trap_family_consumed=false`, and `semantic_authority_promoted=false`.

## Scientific question

`EXP-297` asks whether semantic-checking information can reject compile-valid but semantically wrong formalizations better than compile/proof validity alone under matched neural substrate, identical candidate inputs, identical formal backend, and fully charged verification cost.

Formal compilation is not semantic authority. A candidate can compile while dropping, inventing, strengthening, weakening, rebinding, or flipping source meaning. Ground-truth candidate labels belong to the evaluator and MUST NOT enter either arm's causal decision path.

## Candidate generator

Each deterministic replicate constructs a source `CanonicalProblemState` and an ordered, frozen candidate tuple before either arm executes. Both arms receive byte-identical serialized candidates in the same order.

DEVELOPMENT covers these construction-provenance strata:

1. `faithful_equivalent`
2. `relation_shift`
3. `constraint_drop`
4. `constraint_strengthen`
5. `constraint_weaken`
6. `variable_binding_swap`
7. `domain_mapping_error`
8. `negation_or_relation_flip`

Each candidate carries evaluator-only construction provenance containing its expected fidelity and stratum. Arm-visible input excludes those fields. Candidate/source digests are computed from canonical CPS structure, not `world_id` or evaluator labels.

The DEVELOPMENT batch is balanced by construction: every source has faithful and wrong candidates, and the runner reports class/stratum support rather than trusting an aggregate emitted by the generator.

## Bidirectional exact witness court

`FidelityCourt` becomes a fail-closed adjudicator while preserving `accept()` as a compatibility wrapper.

For matching variable names/domains, it enumerates the complete assignment space only when the space is `<= max_exact_assignments`. For each assignment it evaluates source and candidate independently.

A semantic difference is witnessed in either direction:

- `source_to_candidate`: source accepts and candidate rejects. Candidate has lost source-permitted semantics / over-strengthened.
- `candidate_to_source`: candidate accepts and source rejects. Candidate has invented semantics / under-constrained.

The receipt records:

- `decision`: `court_accept | court_reject | court_inconclusive`
- `direction`
- canonical assignment witness when present
- `source_accepts`
- `candidate_accepts`
- source/candidate constraint-check counts
- `assignments_enumerated`
- `assignment_space_size`
- `termination_reason`
- `semantic_verification_operations`

If names/domains are structurally incompatible, the court rejects with a typed structural witness. If the exact assignment space exceeds the ceiling, it returns `court_inconclusive`; absence of a discovered witness MUST NOT be upgraded to proof of fidelity.

`accept()` returns true only for `court_accept`. Therefore old callers remain fail-closed.

## Matched neural arms

Create a matched EXP-297 pair with exactly the same trainable geometry and initialization:

- deterministic CPS source encoder
- deterministic CPS candidate encoder
- recurrent comparison core
- existing `FidelityCourtRegion`
- witness/receipt adapter
- authority head
- verifier head

Both arms execute the same neural path and expose exact total, functional, active-functional, and optimizer-visible parameter counts. Initial trainable tensor digests must match.

`compile_only` performs compile validity and feeds a canonical null-fidelity receipt through the receipt adapter. `fidelity_court` performs compile validity plus the public witness court and feeds only observed receipt features. Hidden trap family, faithful/wrong label, construction provenance, challenge identity, or evaluator verdict MUST NOT enter model inputs.

For DEVELOPMENT authority semantics, compile-only can authorize any compile-valid candidate. Fidelity-court authorizes only compile-valid candidates whose exact court decision is `court_accept`; `court_reject` and `court_inconclusive` remain restricted. Neural scores are recorded as falsifiable substrate outputs, not allowed to override a deterministic rejection/inconclusive safety result.

## Cost accounting

Do not call abstract accounting hardware FLOPs.

The artifact separates:

- `neural_accounted_flops`: analytical neural path accounting shared by both arms
- `semantic_verification_operations`: assignment enumeration, source constraint checks, candidate constraint checks, witness materialization, and structural checks
- `total_accounted_cost_proxy`: explicit sum for DEVELOPMENT comparison only

Compile checks are performed in both arms. The fidelity arm does not receive a stronger solver/compiler; semantic verification is the intervention and its operations are charged rather than hidden.

## Raw endpoint reconstruction

Every candidate row stores evaluator truth separately from arm-observable receipts and records the authority decision. The validator reconstructs:

- TP: faithful + authority
- TN: wrong + reject/restrict
- FP: wrong + authority
- FN: faithful + reject/restrict
- balanced accuracy = `0.5 * (TPR + TNR)`
- wrong-formalization authority rate = `FP / wrong_count`
- faithful-formalization rejection rate = `FN / faithful_count`

An inconclusive court result counts as no authority. On a faithful candidate that therefore contributes an FN; this is intentional fail-closed safety accounting.

## Artifact and reconstruction validator

Artifact schema: `NLM-EXP-297-PAIRED-DEV-EVAL-V1`.

The validator does not trust declared aggregates. It regenerates each source/candidate set from deterministic lineage, validates canonical digests/order, recomputes compile validity, reruns the bidirectional court, verifies any witness assignment and direction, reconstructs operation counts, recomputes per-arm confusion matrices/endpoints, and checks the DEVELOPMENT aggregate.

It must reject re-hashed tampering of:

- faithful label
- trap stratum/construction provenance
- witness deletion or forged assignment
- court direction/decision
- authority decision
- compute/verification cost reduction
- `court_inconclusive -> court_accept`
- candidate reorder/drop/duplication
- source/candidate digest changes
- confirmatory/challenge flags
- hidden trap family consumption
- semantic-authority promotion

## Registry, CLI and CI

Extend the Neural Arm Registry to include `EXP-297`, while preserving `match_court = BLOCKED`. A validated DEVELOPMENT execution may reach `PAIRED_FIDELITY_COURT_DEV_READY`; that state means machinery/headroom is engineering-ready, not scientifically promoted.

Add a hardened `scripts/run_exp297_paired_dev.py` that verifies the frozen protocol digest before model work, refuses output overwrite, stages JSON atomically, validates the execution and registry payloads before publication, and exposes a CPU-safe `--tiny` mode for CI.

CI runs focused EXP-297 unit/tamper/CLI coverage and a tiny paired smoke. Package version becomes `0.15.0` only after this lane is integrated.

## Non-claims

A perfect DEVELOPMENT balanced accuracy does not scientifically solve EXP-297, validate open-language formalization, validate `FidelityCourtRegion` as a learned semantic judge, consume confirmatory-open evidence, or authorize unrestricted formal reasoning. Confirmatory sample-size freeze, hidden semantic-trap stratification, future-beacon challenge materialization, and the frozen +0.10 BA decision rule remain future evidence gates.
