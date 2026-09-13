# NLM V0.16.1 Program Closure Matrix — Design

## Purpose

This document turns the V0.16.1 EXP-277..EXP-300 roadmap into an evidence-aware dependency authority. It does not rewrite the original research specification and does not reinterpret negative results as success. Its job is to prevent two failure modes: continuing a downstream experiment after its parent hypothesis has been killed, and treating DEVELOPMENT diagnostics as confirmatory scientific closure.

Base authority for this matrix is `main@e7a034365e542895b772701a247f6ad9b931bfc9` plus preserved GitHub experiment evidence. The frozen Stage-A protocol digest remains `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`.

## Canonical status vocabulary

Every EXP-277..EXP-300 row has exactly one program status:

- `SCIENTIFICALLY_CLOSED_PROMOTE` — a frozen scientific court reached a promotion disposition for the scoped hypothesis.
- `SCIENTIFICALLY_CLOSED_KILL` — a frozen scientific court killed the scoped hypothesis. This is not a blanket refutation of neighboring hypotheses.
- `DEVELOPMENT_CLOSED_NEGATIVE` — a preregistered DEVELOPMENT court closed negatively but cannot be promoted to EV-E3 scientific closure.
- `DEVELOPMENT_OPEN` — development machinery/results exist but the experiment family has no final scientific disposition.
- `AUTHORIZED_NEXT_STAGE` — upstream evidence explicitly supports opening this experiment as the next learned/integrated court.
- `DEPENDENCY_REVIEW_REQUIRED` — the experiment is not forbidden, but an upstream kill or unresolved parent invalidates the original straight-line justification; a new hypothesis must be preregistered before implementation.
- `BLOCKED_ON_PARENT` — this experiment must not run until the named parent experiment passes its required gate.
- `NOT_STARTED` — no current implementation/evidence authority was found and no stronger dependency status applies.

`SCIENTIFICALLY_CLOSED_*` refers only to the frozen experiment scope. `PROMOTE` never means integrated NLM success. `KILL` never silently expands beyond the frozen hypothesis.

## Evidence anchors already established

### EXP-277 — oracle structure headroom

Real EV-E3 Gate A/B evidence closed `KILL_SUBSYSTEM` for the frozen synthetic structure-dense oracle-headroom comparison. Observed oracle-CBRF verified-utility-per-accounted-FLOP relative gain was about `+0.0573`, below frozen MESI `+0.10`, with the protected solution-rate guard passing. Program status: `SCIENTIFICALLY_CLOSED_KILL`.

Consequence: EXP-278 may not proceed as the old claim “learned compiler recovers material oracle headroom” because the required material oracle headroom was not established. Reopening compiler work requires a genuinely different hypothesis, not a same-protocol rerun.

### EXP-279 — propagation / branch / hybrid

The main branch contains the matched routing substrate and later merged causal diagnostics. A long DEVELOPMENT lineage then tested routing/remediation courts through V10. The latest sealed V10 CRIC disposition is `REPRESENTATION_SIGNAL_NOT_ESTABLISHED`; all preregistered representation views were cross-budget not identifiable, `successor_design_authorized=false`, and the only permitted continuation is new representation/objective research. This evidence is DEVELOPMENT-only and V10 remained closed unmerged. Program status: `DEVELOPMENT_CLOSED_NEGATIVE` for the current routing/representation line, not a scientific kill of every propagation/search hybrid.

Consequence: no threshold tuning, V10 rerun, sample escalation, or laundering into Stage-A promotion. EXP-280/281 require independent dependency review rather than inheriting a positive EXP-279 result.

### EXP-282 — explicit belief state

Real EV-E3 confirmatory-open ceremony closed `KILL_SUBSYSTEM` for the frozen `explicit_belief` versus `recurrent_hidden` partial-observability comparison. Program status: `SCIENTIFICALLY_CLOSED_KILL`.

Consequence: EXP-283/284/285 are not automatically scientifically dead, because belief-deviation recovery, falsifier seeking, and residual typing can be posed independently. They do, however, lose the original straight-line justification from a winning explicit-belief subsystem and therefore require new preregistered hypotheses.

### EXP-286 — oracle conflict-core value

Real EV-E3 Gate-B court closed `PROMOTE_TO_NEXT_STAGE` for the frozen oracle conflict-core comparison. Program status: `SCIENTIFICALLY_CLOSED_PROMOTE`.

Consequence: EXP-286 demonstrated scoped oracle conflict-core headroom and authorized the learned-localization question tested by EXP-287. Its positive result remains intact after EXP-287's negative learned-localization result.

### EXP-287 — learned conflict localization

The preregistered EXP-287 V1 DEVELOPMENT court executed once across four canonical roots. All roots reproduced positive oracle headroom, positive learned economic headroom, and verified-solution rate `1.0`; learned oracle-value capture was approximately `0.719..0.758`. However, top-2 true-core precision was only `0.2207..0.2617`, with cross-root mean `0.2392578125`, near the frozen raw core prevalence `0.25`. All four roots therefore failed the preregistered `top2_core_precision > 0.50` localization predicate.

Sealed cross decision: `LOCALIZATION_VALUE_NOT_ESTABLISHED`. `successor_design_authorized=false`, `authorization_scope=NONE`, and `mechanism_successor_authorized=false`. Evidence level remained EV-E2 / DEVELOPMENT. Program status: `DEVELOPMENT_CLOSED_NEGATIVE`.

Consequence: EXP-288 remains blocked under the current hypothesis. EXP-287 V1 may not be repaired by post-result threshold changes, top-k changes, sample escalation, replacement roots, or rerun. A future return to conflict localization requires a scientifically distinct representation/objective hypothesis.

### EXP-289 — episode-local nogoods

Real EV-E3 Gate-B court closed `PROMOTE_TO_NEXT_STAGE` for the frozen episode-local nogood comparison. Program status: `SCIENTIFICALLY_CLOSED_PROMOTE`.

Consequence: EXP-289 validly authorized EXP-290 as a learned-clause transfer/generalization research seam. EXP-290 has now executed and closed separately; the positive EXP-289 result remains intact.

### EXP-290 — structural clause transfer

EXP-290 V1 was a preregistered DEVELOPMENT / EV-E2 court asking whether a supervised learned source→target clause translator could reuse observed one-literal nogoods across mandatory non-identity variable surface permutations while preserving EXP-289 safety floors.

The exact pre-data head passed dedicated EXP-290 CI, generic core 3.11/core 3.13/model-smoke, Stage-A/geometry identity checks, and a zero-run freeze guard before a marker-only release. Exactly one authoritative DEVELOPMENT run (`34743509423`, attempt 1) executed four canonical roots and one cross reducer; the post-release freeze guard observed that run only and no duplicates.

All four roots reproduced strong positive fresh oracle transfer headroom (`0.673828125..0.6875`), so the denominator for learned transfer value was valid. However, learned oracle-value capture was only about `0.1282..0.1834`, far below the frozen `0.50` majority-of-oracle threshold. Learned valid-state over-prune was about `0.6839..0.8106`, far above the frozen `0.005` safety ceiling, and learned verified-solution rate was only `0.36328125..0.5234375` versus control `1.0` on every root.

All four roots therefore closed `LEARNED_STRUCTURAL_TRANSFER_NOT_ESTABLISHED`. The sealed cross decision is `STRUCTURAL_CLAUSE_TRANSFER_NOT_ESTABLISHED`, with `successor_design_authorized=false`, `authorization_scope=NONE`, and `mechanism_successor_authorized=false`. Evidence remained EV-E2 / DEVELOPMENT. Program status: `DEVELOPMENT_CLOSED_NEGATIVE`.

Consequence: EXP-291 remains blocked on this dependency path. EXP-290 V1 may not be repaired by post-result calibration, top-k/cardinality changes, threshold changes, sample escalation, replacement roots, geometry changes, or rerun. This negative result closes only the frozen V1 representation/objective; it does not erase EXP-289 or prove that all structural-transfer representations are impossible.

### EXP-297 — Encoding Fidelity Court

Real EV-E3 Gate-B evidence closed `PROMOTE_TO_NEXT_STAGE` for the frozen synthetic semantic-formalization challenge family while explicitly keeping unrestricted semantic authority false. Program status: `SCIENTIFICALLY_CLOSED_PROMOTE`.

Consequence: EXP-298 is independently authorized as a cross-domain transfer court, but no open-domain semantic claim may be inherited from EXP-297.

## EXP-277..EXP-300 matrix

| ID | Original question | Program status | Dependency / next authority |
|---|---|---|---|
| EXP-277 | Oracle constraint/factor headroom over ARCS | `SCIENTIFICALLY_CLOSED_KILL` | Frozen H-CBRF-01 killed; no same-hypothesis rerun |
| EXP-278 | Learned compiler recovers oracle headroom | `DEPENDENCY_REVIEW_REQUIRED` | Old justification blocked by EXP-277 kill; requires new hypothesis |
| EXP-279 | Propagation vs branch vs hybrid | `DEVELOPMENT_CLOSED_NEGATIVE` | Current routing/representation line closed through V10; redesign-only |
| EXP-280 | Energy/soft-constraint equilibration beyond message passing | `DEPENDENCY_REVIEW_REQUIRED` | Must not assume EXP-279 success; independent preregistration required |
| EXP-281 | Detect/recover loopy oscillation/local minima | `DEPENDENCY_REVIEW_REQUIRED` | Requires a viable propagation/equilibration substrate definition |
| EXP-282 | Explicit belief vs recurrent hidden state | `SCIENTIFICALLY_CLOSED_KILL` | Frozen explicit-belief comparison killed |
| EXP-283 | Recovery from induced belief deviation | `DEPENDENCY_REVIEW_REQUIRED` | New hypothesis required after EXP-282 kill |
| EXP-284 | Active falsifier seeking against self-locking | `DEPENDENCY_REVIEW_REQUIRED` | May be independent but cannot inherit explicit-belief authority |
| EXP-285 | Residual typing: conflict/unknown/search-incomplete/model-mismatch | `DEPENDENCY_REVIEW_REQUIRED` | New standalone residual-typing hypothesis required |
| EXP-286 | Oracle conflict-core value | `SCIENTIFICALLY_CLOSED_PROMOTE` | Positive oracle headroom preserved; learned seam tested by EXP-287 |
| EXP-287 | Learned conflict localization approaches oracle value | `DEVELOPMENT_CLOSED_NEGATIVE` | V1 closed `LOCALIZATION_VALUE_NOT_ESTABLISHED`; no same-hypothesis rerun |
| EXP-288 | Dependency-directed backjump vs chronological rollback | `BLOCKED_ON_PARENT` | EXP-287 did not authorize successor design |
| EXP-289 | Episode-local nogoods | `SCIENTIFICALLY_CLOSED_PROMOTE` | Positive scoped local-memory result preserved; its authorized EXP-290 seam has now closed negative |
| EXP-290 | Learned clauses transfer across surface randomization | `DEVELOPMENT_CLOSED_NEGATIVE` | V1 closed `STRUCTURAL_CLAUSE_TRANSFER_NOT_ESTABLISHED`; no same-hypothesis rerun |
| EXP-291 | Detect spurious counterexamples from bad encoding/modeling | `BLOCKED_ON_PARENT` | EXP-290 did not authorize successor design; new independent hypothesis required to reopen |
| EXP-292 | Counterexample-guided invariant/lemma generation | `BLOCKED_ON_PARENT` | Requires trustworthy counterexample/encoding boundary, normally EXP-291 |
| EXP-293 | Lemma-precondition eligibility checks | `BLOCKED_ON_PARENT` | Requires a viable lemma generator/library from EXP-292 |
| EXP-294 | Utility-governed Lemma Economy | `BLOCKED_ON_PARENT` | Requires EXP-292/293 viable lemma artifacts |
| EXP-295 | Clause/lemma garbage collection | `BLOCKED_ON_PARENT` | Requires a live library and utility semantics from EXP-294 |
| EXP-296 | Truth-maintenance dependency tracking | `BLOCKED_ON_PARENT` | Requires reusable clauses/lemmas with provenance dependencies |
| EXP-297 | Encoding Fidelity Court | `SCIENTIFICALLY_CLOSED_PROMOTE` | Authorizes scoped EXP-298 transfer research |
| EXP-298 | Transfer beyond SAT/CSP | `AUTHORIZED_NEXT_STAGE` | Highest-priority currently authorized surviving court; must preserve EXP-297 semantic-authority limits |
| EXP-299 | Gains after scaffold/cache/retrieval removal | `BLOCKED_ON_PARENT` | Requires integrated gains in domains from EXP-298 and earlier surviving chains |
| EXP-300 | Full V0.16 vs strongest 100M rivals | `BLOCKED_ON_PARENT` | Requires an integrated candidate assembled only from surviving mechanisms |

## Priority rule after EXP-290 closure

The next active research priority is EXP-298 because:

1. EXP-290 closed validly negative, so EXP-291 remains blocked and cannot be opened by reinterpretation.
2. EXP-292..EXP-296 consequently remain blocked on their counterexample/lemma dependency chain unless a scientifically distinct replacement hypothesis is preregistered.
3. EXP-297 already produced real EV-E3 positive evidence for the scoped encoding-fidelity question and explicitly authorized EXP-298 as the next cross-domain transfer court.
4. EXP-298 is independent of the failed EXP-290 mechanism, so pursuing it does not launder EXP-290's negative result into a successor authorization.
5. EXP-287 V1 and EXP-290 V1 remain immutable negative DEVELOPMENT courts; neither may be tuned or rerun merely to keep the roadmap moving.
6. EXP-279's current routing/representation line remains redesign-only, while EXP-278 and EXP-280..285 require independent dependency review rather than inheriting positive authority.

## Closure discipline

A program row changes state only with an immutable evidence record that names the experiment, frozen hypothesis, evidence level, decision, exact code/protocol identity, and scope boundary. Negative results reduce the active architecture. They are not converted into “partial success” merely to preserve a roadmap.

A downstream experiment may be skipped permanently if its parent is killed or fails its authorization gate and no scientifically distinct replacement hypothesis is preregistered. Completion of V0.16.1 therefore means closing the dependency graph honestly, not mechanically executing every numeric experiment.
