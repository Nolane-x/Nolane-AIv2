# NLM V0.16.1 Program Closure Matrix — Design

## Purpose

This document turns the V0.16.1 EXP-277..EXP-300 roadmap into an evidence-aware dependency authority. It does not rewrite the original research specification and does not reinterpret negative results as success. Its job is to prevent two failure modes: continuing a downstream experiment after its parent hypothesis has been killed, and treating DEVELOPMENT diagnostics as confirmatory scientific closure.

Base authority for this matrix is `main@b9b9f63980e31313a1f86ccb1f41fce78596b553` plus preserved GitHub experiment evidence. The frozen Stage-A protocol digest remains `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`.

## Canonical status vocabulary

Every EXP-277..EXP-300 row has exactly one program status:

- `SCIENTIFICALLY_CLOSED_PROMOTE` — a frozen scientific court reached a promotion disposition for the scoped hypothesis.
- `SCIENTIFICALLY_CLOSED_KILL` — a frozen scientific court killed the scoped hypothesis. This is not a blanket refutation of neighboring hypotheses.
- `DEVELOPMENT_CLOSED_POSITIVE` — a preregistered DEVELOPMENT / EV-E2 court passed its frozen scoped decision rule. This is not EV-E3 scientific promotion; it may grant only the explicit successor authority encoded by that DEVELOPMENT court.
- `DEVELOPMENT_CLOSED_NEGATIVE` — a preregistered DEVELOPMENT court closed negatively but cannot be promoted to EV-E3 scientific closure.
- `DEVELOPMENT_OPEN` — development machinery/results exist but the experiment family has no final scientific disposition.
- `AUTHORIZED_NEXT_STAGE` — upstream evidence explicitly supports opening this experiment as the next learned/integrated court.
- `AUTHORIZED_DESIGN_ONLY` — upstream evidence authorizes preregistration/design of the named successor court only. It does not authorize implementation-time data consumption or execution before that successor has its own frozen protocol and pre-data gate.
- `DEPENDENCY_REVIEW_REQUIRED` — the experiment is not forbidden, but an upstream kill or unresolved parent invalidates the original straight-line justification; a new hypothesis must be preregistered before implementation.
- `BLOCKED_ON_PARENT` — this experiment must not run until the named parent experiment passes its required gate.
- `NOT_STARTED` — no current implementation/evidence authority was found and no stronger dependency status applies.

`SCIENTIFICALLY_CLOSED_*` refers only to the frozen experiment scope. `PROMOTE` never means integrated NLM success. `KILL` never silently expands beyond the frozen hypothesis. `DEVELOPMENT_CLOSED_POSITIVE` remains DEVELOPMENT evidence and must not be rewritten as confirmatory promotion.

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

Consequence: EXP-298 was independently authorized as a cross-domain transfer court, but no open-domain semantic claim could be inherited from EXP-297.

### EXP-298 — cross-domain fidelity transfer

EXP-298 V1 executed as a preregistered DEVELOPMENT / EV-E2 court across bounded code invariants, finite intervention-based causal diagnosis, and controlled grounded-language ambiguity. The exact pre-data head passed dedicated EXP-298 CI, generic core 3.11/core 3.13/model-smoke, Stage-A/geometry identity checks, compile checks, and a zero-run freeze guard before a marker-only release.

The sole authoritative scientific execution is DEVELOPMENT run `34751159358` at marker head `262e786e5bea59e78a380c9d8a552259b10e7908`. All four canonical root jobs completed successfully, validated their receipts before upload, and sealed their GitHub Actions artifacts. Every root/domain cell contained 512 candidates (256 faithful / 256 wrong). Compile-only balanced accuracy was `0.5`; fidelity-fabric balanced accuracy was `1.0`; the gain was therefore `+0.5` in every one of the 12 root/domain cells versus frozen MESI `+0.10`. Fidelity wrong-formalization authority and faithful-formalization rejection were both `0.0` everywhere. All four roots closed `CROSS_DOMAIN_FIDELITY_TRANSFER_ESTABLISHED`.

The workflow cross job later failed because full root reconstruction in a new process reported `EXP-298 root reconstruction mismatch`. The producing jobs had already passed the same root validator in-process. Post-run forensic verification of the sealed root artifacts independently revalidated receipt byte sidecars, internal artifact digests, frozen identities, causal authority decisions, domain metrics, model-state immutability, matched-arm boundaries, and all four established decisions without re-executing the model. The failure is archived as `CROSS_PROCESS_FULL_PAYLOAD_RECONSTRUCTION_PORTABILITY_FAILURE`; no hardware-specific cause is asserted as proven.

Applying the already-frozen four-of-four cross rule to the four sealed authoritative root decisions yields `CROSS_DOMAIN_FIDELITY_TRANSFER_RECURRENT`. The recovered disposition explicitly records the original cross-job failure and authorizes only `DESIGN_EXP299_SCAFFOLD_REMOVAL_COURT_ONLY`; EXP-300 remains unauthorized. Evidence remains DEVELOPMENT / EV-E2. Program status: `DEVELOPMENT_CLOSED_POSITIVE`.

A later GitHub workflow object (`34751503700`, run #2) was created by cumulative PR path semantics after a test-only commit, but its first-authority preflight failed before checkout/root/model execution. It is not a second scientific execution. PR #70 was closed unmerged and the experiment branch sealed.

Consequence: EXP-299 was validly authorized for design/preregistration as the next scaffold-removal court. EXP-299 has now executed and closed separately; the positive EXP-298 result remains intact within its bounded fidelity-fabric scope and does not itself establish native scaffold-free gains.

### EXP-299 — native fidelity scaffold removal

EXP-299 V1 / `H-NATIVE-01` was a preregistered DEVELOPMENT / EV-E2 court asking whether fit-time auxiliary exact-court teacher supervision leaves a material neural fidelity/authority advantage after the exact court is completely removed from held-out causal inference.

Before release, exact-head dedicated EXP-299 contracts, structural leakage checks, fresh-process validator portability, generic core Python 3.11 and 3.13, generic model-smoke, Stage-A/geometry identity checks, compileall, and the zero-run freeze guard were all green. The marker-only release bound pre-marker head `564aa26b9e4dc0c9189aaa824dc9999c824c92f1`, source-tree digest `422d0c43ee1cf3d3384c0b2c59cee7ade594ce9477946880ea57e2b89c11b4c8`, geometry digest `7210e64dca010c6c1c8ab2cf20c7f03f7d7f263a88f9c8ae162d43151bb9932e`, structural-encoder contract digest `843069e7081309842f8e594f1ab8833f3ad23b2b3d49c975351268e10f157a23`, and frozen Stage-A digest.

Exactly one authoritative DEVELOPMENT run (`34753873059`, run #1, attempt 1) executed all four canonical roots and the cross reducer at marker head `2ab16177fd2bda5e53bc5353b9aa9ec01590dd81`. Each root used 4,608 fit candidates plus 1,536 held-out candidates across the three frozen domain families. Root receipts reported zero inconclusive cases, disjoint fit/evaluation identities, prediction commitments before evaluator-court materialization, and unchanged model-state digests across held-out evaluation.

All four roots closed `NATIVE_FIDELITY_NOT_ESTABLISHED`. The native teacher arm showed isolated positive cells, but the advantage was not recurrent or conjunctively safe across the frozen domains and roots. Examples include root 0 code-invariant gain `+0.4375` with wrong-authority `0.125`; root 2 grounded-language gain `+0.125` with zero wrong authority while code-invariant gain was `-0.35546875` with wrong-authority `0.765625`; and root 3 causal/language gains were only `+0.0625`, below the frozen `+0.10` MESI, while code invariant rejected all faithful cases.

The sealed cross decision is `NATIVE_FIDELITY_NOT_ESTABLISHED_RECURRENT`, cross artifact digest `40a4259f912402e171a01a99adc74093749f756fba049428d00764d5a4210520`, with `established_root_count=0`, `successor_design_authorized=false`, `authorization_scope=NONE`, and `exp300_execution_authorized=false`. Evidence remains EV-E2 / DEVELOPMENT. Program status: `DEVELOPMENT_CLOSED_NEGATIVE`.

Independent post-run byte audit of all four root artifacts and the cross artifact revalidated receipt sidecars, internal artifact digests, partition identities, raw metric reconstruction, prediction commitments, model-state immutability, root-to-cross digest linkage, and protected evidence flags. No audit mismatch was found. PR #72 was then closed unmerged and the experiment branch sealed.

Consequence: EXP-300 remains blocked on this lineage. EXP-299 V1 may not be repaired by post-result threshold changes, auxiliary-weight tuning, representation changes, sample escalation, root replacement, domain removal, or rerun. Any future return to scaffold-free native fidelity requires a scientifically distinct preregistered hypothesis; it cannot inherit successor authority from EXP-299 V1.

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
| EXP-297 | Encoding Fidelity Court | `SCIENTIFICALLY_CLOSED_PROMOTE` | Scoped positive fidelity court preserved; EXP-298 transfer seam closed positive in DEVELOPMENT |
| EXP-298 | Transfer beyond SAT/CSP | `DEVELOPMENT_CLOSED_POSITIVE` | Frozen controlled-domain V1 closed 4/4 recurrent; authorized EXP-299 design only, now consumed |
| EXP-299 | Gains after scaffold/cache/retrieval removal | `DEVELOPMENT_CLOSED_NEGATIVE` | V1 closed `NATIVE_FIDELITY_NOT_ESTABLISHED_RECURRENT`; no successor authority and no same-hypothesis rerun |
| EXP-300 | Full V0.16 vs strongest 100M rivals | `BLOCKED_ON_PARENT` | EXP-299 did not establish required native scaffold-free gains or authorize successor design/execution |

## Priority rule after EXP-299 closure

The original straight-line EXP-297 → EXP-298 → EXP-299 → EXP-300 dependency path has no currently authorized successor after EXP-299 V1, because:

1. EXP-298 remains a valid positive DEVELOPMENT / EV-E2 result for its frozen fidelity-fabric transfer question, but its scoped successor authority was consumed by EXP-299 design/preregistration.
2. EXP-299 executed exactly once under its own frozen court and closed `NATIVE_FIDELITY_NOT_ESTABLISHED_RECURRENT` with `0/4` established roots.
3. EXP-299 grants neither successor-design nor EXP-300 execution authority. EXP-300 therefore remains blocked rather than being mechanically opened to keep the roadmap moving.
4. EXP-299 V1 is immutable after held-out visibility. Threshold tuning, teacher-weight tuning, representation edits, root replacement, domain removal, sample escalation, or same-hypothesis rerun are not permitted.
5. A future attempt at model-native scaffold-free fidelity must start from a scientifically distinct hypothesis—such as a different representation, objective, learning mechanism, or internalization mechanism—with a new preregistration and new pre-data gate. Merely renaming or widening EXP-299 V1 does not reopen it.
6. A future integrated V0.16 comparison requires an independently justified viable native candidate plus its own VCPF/safety authority. Neither EXP-297, EXP-298, nor EXP-299 establishes integrated superiority.
7. EXP-290 remains closed validly negative, so EXP-291 remains blocked and EXP-292..EXP-296 remain blocked on that dependency chain unless a scientifically distinct replacement hypothesis is preregistered.
8. EXP-287 V1 and EXP-290 V1 remain immutable negative DEVELOPMENT courts; neither may be tuned or rerun merely to keep the roadmap moving.
9. EXP-279's current routing/representation line remains redesign-only, while EXP-278 and EXP-280..285 require independent dependency review rather than inheriting positive authority.

The next legitimate research action is therefore **hypothesis generation and preregistration**, not execution of EXP-300 and not repair/rerun of EXP-299 V1.

## Closure discipline

A program row changes state only with an immutable evidence record that names the experiment, frozen hypothesis, evidence level, decision, exact code/protocol identity, and scope boundary. Negative results reduce the active architecture. Positive DEVELOPMENT results grant no more authority than their preregistered successor scope.

A downstream experiment may be skipped permanently if its parent is killed or fails its authorization gate and no scientifically distinct replacement hypothesis is preregistered. Completion of V0.16.1 therefore means closing the dependency graph honestly, not mechanically executing every numeric experiment.
