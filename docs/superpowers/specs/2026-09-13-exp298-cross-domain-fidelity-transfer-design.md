# EXP-298 Cross-Domain Fidelity Transfer Court Design

## Authority

EXP-298 / `H-XDOMAIN-01` is authorized only by EXP-297's scoped `EV-E3 / PROMOTE_TO_NEXT_STAGE` result. EXP-297 kept `semantic_authority_promoted=false`; EXP-290 is closed negative; EXP-291..296 remain blocked. Base: `main@150252b63f7268157e66024a4f9e55ad61088e35`. This lane MUST NOT modify `protocols/stage_a_v1.json` or its SHA sidecar.

## Question

Does the EXP-297 fidelity/verification fabric retain material value beyond SAT/CSP in **code invariants, causal diagnosis, and controlled language ambiguity**?

V1 tests cross-domain verification transfer, not unrestricted language understanding, arbitrary code verification, causal discovery, EXP-290 clause transfer, or integrated V0.16 superiority.

## Common court

Use one fail-closed `BehavioralFidelityCourt` with three domain adapters. Each adapter exposes a canonical finite probe sequence, exact evaluation, and canonical digest. `adjudicate(source, candidate, semantics)` returns:

- `court_accept`: complete frozen probe space exhausted with no difference;
- `court_reject`: directional witness found (`source_to_candidate` or `candidate_to_source`);
- `court_inconclusive`: exact comparison cannot complete within `max_exact_probes`.

No witness is not proof unless the adapter certifies exhaustive completion. `court_inconclusive` grants no authority.

## Domains

### Code invariants

Use a bounded deterministic transition DSL, not arbitrary Python. Problems contain finite integer/Boolean state, deterministic branches/transitions, invariant, and bounded pre-state domain. Probes enumerate all pre-states and one transition step. Frozen strata: `faithful_equivalent`, `off_by_one_boundary`, `stale_variable_reference`, `branch_omission`, `variable_binding_swap`, `predicate_polarity_flip`, `invariant_strengthen`, `invariant_weaken`, `incorrect_post_state_reference`.

### Causal diagnosis

Use finite acyclic deterministic SCMs. Probes are exogenous settings × allowed single-variable interventions. Outcomes are endogenous assignments plus diagnosis target. Frozen strata: `faithful_equivalent`, `edge_reversal`, `mediator_omission`, `wrong_intervention_target`, `effect_polarity_flip`, `exogenous_binding_swap`, `observationally_equivalent_interventionally_wrong`, `spurious_direct_edge`. Every replicate includes an intervention-only-disambiguated wrong candidate.

### Grounded language ambiguity

Use finite micro-worlds plus controlled sentences with exact finite semantics. Probes enumerate grounded contexts. Frozen strata: `faithful_equivalent`, `quantifier_scope_swap`, `negation_scope_flip`, `relation_direction_swap`, `referent_binding_swap`, `attachment_swap`, `conjunction_disjunction_flip`, `existential_universal_flip`. This is controlled grounded language only.

## Candidate and leakage boundary

For each `(root, domain, replicate)`, source plus ordered candidate tuple is frozen before arms execute. Each domain uses 16 executable candidates per replicate, balanced faithful/wrong. `is_faithful` and `stratum` are evaluator-only and MUST NOT enter arm causal inputs. Receipts prove byte-identical order/digests and evaluator-truth/stratum exclusion.

## Matched arms

Arms: `compile_only_control` and `cross_domain_fidelity_fabric`. Both have identical trainable geometry, initialization, source/candidate encoders, recurrent comparison core, authority/verifier heads. Only fidelity arm receives the public behavioral receipt; control gets a canonical null receipt. Fidelity authority requires executable candidate and exact `court_accept`; reject/inconclusive remain restricted. Neural scores cannot override exact court results.

## Frozen V1 geometry

- roots `[0,1,2,3]`
- domains `code_invariant`, `causal_diagnosis`, `grounded_language_ambiguity`
- 32 evaluation replicates/domain/root
- 16 candidates/replicate
- 1536 candidate evaluations/root; 6144 total
- `d_model=64`, `hidden_size=64`, `target_parameters=500000`
- `max_exact_probes=4096`
- per-domain BA-gain MESI `>=0.10`
- per-domain wrong-authority ceiling `<=0.05`
- per-domain faithful-rejection ceiling `<=0.10`

No threshold sweep, root replacement, post-result domain removal, geometry/cardinality change, or sample escalation after one-shot release.

## Decisions

For domain `d`, `BA_gain[d] = BA(fidelity,d) - BA(control,d)`. Root primary endpoint is `min_d BA_gain[d]`.

Root returns `CROSS_DOMAIN_FIDELITY_TRANSFER_ESTABLISHED` only when all three domains meet all frozen effect/safety gates, pairing/leakage receipts reconstruct, model geometry/init matches, raw rows reconstruct all aggregates, and all non-claim flags remain false. Otherwise: `CROSS_DOMAIN_FIDELITY_TRANSFER_NOT_ESTABLISHED`.

Cross reducer consumes exactly four canonical roots. Only 4/4 established roots may return `CROSS_DOMAIN_FIDELITY_TRANSFER_RECURRENT` with `successor_design_authorized=true` and `authorization_scope=DESIGN_EXP299_SCAFFOLD_REMOVAL_COURT_ONLY`. Every other valid result sets authorization false/NONE. EXP-298 cannot authorize EXP-300 directly.

## Cost and artifacts

Per candidate publish `neural_accounted_flops`, `compile_validation_operations`, `semantic_probe_evaluations`, `semantic_domain_operations`, `total_accounted_cost_proxy`, and `hardware_profiler_flops_claimed=false`. Cost-normalized diagnostics are secondary and cannot replace the frozen primary endpoint.

Root schema: `NLM-EXP-298-CROSS-DOMAIN-FIDELITY-ROOT-V1`. Cross schema: `NLM-EXP-298-CROSS-DOMAIN-FIDELITY-CROSS-V1`. Validators regenerate deterministic worlds/candidates, rerun exact semantics/court, verify witnesses/order/digests/leakage/cost/model match, and recompute metrics/decisions. Canonical JSON receipts use SHA256 sidecars and internal canonical `artifact_digest` excluding that field.

## Evidence boundary

V1 is DEVELOPMENT / EV-E2 only. Root and cross receipts must keep false: `scientific_evidence_eligible`, `confirmatory_data_consumed`, `challenge_materialized`, `promotion_claimed`, `unrestricted_semantic_authority_claimed`, `open_language_understanding_claimed`, `causal_discovery_claimed`, `general_code_reasoning_claimed`, `exp290_authority_inherited`, `exp291_296_authority_inherited`.

Release discipline: design/plan → TDD → dedicated+generic exact-head CI → zero-run freeze guard → marker-only release → exactly one four-root DEVELOPMENT run → cross reducer → duplicate-run guard → independent artifact audit → close experiment PR unmerged → integrate only docs/closure/matrix to main unless separately authorized.

A positive V1 authorizes only design of EXP-299 scaffold-removal court. A negative V1 closes this frozen hypothesis and must narrow the program rather than be repaired post-result.