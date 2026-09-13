# NLM V0.17 — 10M Native Recursive Substrate

**Status:** `PRE-IMPLEMENTATION / REVIEW-FROZEN / EMPIRICALLY-UNVERIFIED`  
**Parent authority:** `main@a7d14a97ae74ace9a0f4be4e72fade434cbaa96b`  
**Parent terminal state:** `V0161_PROGRAM_EXECUTION_CYCLE_CLOSED`  
**Immediate scientific action:** `EXP-301` only  
**Full W5 review artifact SHA-256:** `56074bb42c788ac721249d833a6967a7bfe9b58b43d4ab9d2ae8aedbe44d7b8c`  
**EXP-301 preregistration digest:** `sha256:110701d1054fe0743bac2b65d16faa602182a1d90ab85bdc7d3d426de0212253`

## 1. Purpose

V0.17 is a scientifically distinct 10M successor lineage. It is not a repaired rerun of V0.16.1 and does not reinterpret any V0.16.1 result. The core question is whether a small shared recurrent latent substrate can obtain reproducible verified-generalization headroom without relying on brittle explicit fabrics or inference-time privileged scaffolds.

The governing rule is:

> No optional cognitive mechanism enters the integrated architecture merely because it is plausible. It must beat the strongest simpler 10M rival under matched resident parameters, matched training data, accounted compute, and protected safety/fidelity floors.

Negative evidence may delete architecture. A positive result authorizes only the next named dependency.

## 2. Binding V0.16.1 evidence

The redesign inherits the following terminal evidence rather than resetting history:

- EXP-277: `SCIENTIFICALLY_CLOSED_KILL`; frozen oracle-CBRF material headroom was not established.
- EXP-279: `DEVELOPMENT_CLOSED_NEGATIVE`; the current propagation/routing representation line did not establish a useful signal.
- EXP-282: `SCIENTIFICALLY_CLOSED_KILL`; the frozen explicit-belief subsystem did not beat recurrent hidden state.
- EXP-286: `SCIENTIFICALLY_CLOSED_PROMOTE`; oracle conflict-core information has scoped value.
- EXP-287: `DEVELOPMENT_CLOSED_NEGATIVE`; learned top-k conflict localization did not reliably recover the true core.
- EXP-289: `SCIENTIFICALLY_CLOSED_PROMOTE`; episode-local nogood information has scoped value.
- EXP-290: `DEVELOPMENT_CLOSED_NEGATIVE`; learned structural transfer captured little oracle value and dangerously over-pruned valid states.
- EXP-297: `SCIENTIFICALLY_CLOSED_PROMOTE`; bounded Encoding Fidelity Court has scoped value.
- EXP-298: `DEVELOPMENT_CLOSED_POSITIVE`; scaffolded bounded cross-domain fidelity transfer recurred.
- EXP-299: `DEVELOPMENT_CLOSED_NEGATIVE`; native scaffold-free fidelity did not recur; EXP-300 remained unauthorized.

A V0.17 hypothesis may revisit one of these questions only if it declares a material change in representation, objective, available information, intervention, causal question, safety mechanism, or evaluation population before data. Threshold, top-k, sample-count, seed, optimizer, hidden-width, or post-hoc calibration changes are not a new hypothesis.

## 3. Architecture thesis

The default architecture is a recurrent monolith, not an explicit cognitive fabric. The model spends parameters on a reusable latent transition operator and spends variable compute by repeatedly applying that operator. Adaptive halting, sparse/event routing, uncertainty state, conflict credit, memory, structural transfer, and native fidelity are optional sidecars that must each earn their capacity.

Three approaches were retained during W5 hypothesis ecology:

1. Pure Recursive 10M — required Occam rival.
2. Dynamic Hypergraph NLM — rejected as the default because it risks recreating CBRF under new vocabulary.
3. Native Recursive Substrate — selected: recurrent core first, evidence-driven optional mechanisms second.

The integrated V0.17 model is the survivor set, not the union of all proposed modules.

## 4. Exact 10M resource constitution

The base geometry is frozen as:

- vocabulary: 4,608 tokens;
- `d_model = 448`;
- seven attention heads × 64 dimensions;
- three shared recurrent layers;
- `d_ff = 1152`;
- RMSNorm;
- RoPE;
- SwiGLU;
- tied token embedding / language-model head.

Parameter ledger:

| Component | Formula | Parameters |
|---|---:|---:|
| token embedding / tied LM head | `4608*448` | 2,064,384 |
| attention per shared layer | `4*448*448` | 802,816 |
| SwiGLU per shared layer | `3*448*1152` | 1,548,288 |
| RMSNorm per shared layer | `2*448` | 896 |
| one shared layer | sum | 2,352,000 |
| three shared layers | `3*2,352,000` | 7,056,000 |
| final RMSNorm | `448` | 448 |
| **frozen base** | | **9,120,832** |
| **Capacity Exchange Envelope** | `10,000,000 - 9,120,832` | **879,168** |
| **hard resident ceiling** | | **10,000,000** |

The 879,168-parameter Capacity Exchange Envelope (CEE) is not free sidecar capacity. In control arms it is spent on useful generic recurrent capacity. Any specialized component must replace equal active generic capacity. Dead padding is forbidden.

## 5. Core recurrence contract

The same three-layer block is repeatedly applied in latent space. Training loop support is `{1,2,4,8}` and challenge loop budgets are `{1,2,4,8,12,16}`. Depths 12 and 16 are extrapolation/stability probes rather than assumed useful compute.

Recurrence must be audited for representation-rank collapse, state-change concentration, residual-norm pathology, output stagnation, step-contribution skew, and compute-curve saturation. More loops are not credited merely because they exist.

Adaptive halting is not part of the initial mandatory core. It becomes eligible only after recurrent-depth headroom is established.

## 6. No learned hard pruning

V0.17 constitutionally forbids a learned score from deleting a candidate state on confidence alone. Learned conflict/memory/transfer mechanisms may produce reversible soft penalties, priorities, energy terms, or requests for more evidence. A hard exclusion requires an independently valid deterministic verifier/formal rule.

This is a direct repair of the EXP-290 failure mode, where learned transfer substantially over-pruned valid states.

## 7. Optional research branches

The following mechanisms are hypotheses, not promised architecture:

- adaptive halting / input-dependent loop allocation;
- event-driven or sparse relational workspace;
- native predictive uncertainty and recovery from induced state corruption;
- active information seeking against self-locking;
- oracle-conflict value replication on the new core;
- counterfactual/intervention-based conflict credit;
- relational or hypergraph conflict representation only if simpler credit leaves headroom;
- soft episode anti-pattern memory;
- reversible structural conflict transfer;
- student-reachable privileged-information testing;
- joint privileged-information distillation;
- progressive scaffold dropout/removal;
- fresh-process cross-domain native fidelity.

A failed optional branch returns its capacity to generic recurrent compute.

## 8. Scaffold/native boundary

EXP-299 showed that supervised auxiliary teacher information did not reliably survive scaffold removal. V0.17 therefore changes the causal question.

Before distillation, EXP-314 asks whether the teacher advantage is statistically reachable from student-visible information. A target classified as `UNREACHABLE_PRIVILEGE` is not treated as a valid internalization target. For reachable targets, EXP-315 compares joint privileged-information distillation against both student-only and naïve auxiliary supervision. EXP-316 progressively removes the scaffold, ending in a student-only training segment, then evaluates in a fresh challenge process.

Native means no challenge-time privileged teacher, exact court, solver trace, oracle embedding, hidden retrieval store, privileged label column, or cached scaffold output. Any such path invalidates the native court.

## 9. Task families and split discipline

V0.17 uses bounded families chosen to expose different causal properties:

- iterative grid/maze reasoning;
- algorithmic sequence transformation;
- generator-heldout abstract transformation;
- partial observability/recovery;
- conflict/constraint worlds;
- bounded fidelity: code invariants, finite intervention-based causal diagnosis, grounded-language ambiguity;
- small language/sequence control.

Splits are generator-level rather than random-example-level. Challenge generators/materializations are created after the pre-data freeze. Generator identities, solver traces, latent parameters, and privileged annotations may not leak through training artifacts.

## 10. Primary evaluation geometry

`VSR(B)` is verified success at accounted compute budget `B`.

EXP-301 uses `RCG`, recurrent compute gain, defined from the normalized difference between paired verified-success-versus-`log2(accounted FLOPs)` curves over common budgets. A model does not win by receiving more compute without producing verified utility.

Later integrated decisions use Pareto operating points rather than a tunable free-form scalar: verified success, accounted compute, resident state, safety/fidelity errors, and wall-time are all reported.

Statistical unit is the held-out generator instance paired across arms; training root is the replication stratum. Default gates use roots `[0,1,2,3]`; EXP-318 uses eight roots. Primary uncertainty is a paired, root-stratified hierarchical bootstrap. Token counts are not treated as independent replicates.

Practical equivalence selects the simpler architecture.

## 11. Experiment dependency graph

The V0.17 namespace is:

- EXP-301 recurrent-depth headroom — `PREREGISTERED_NEXT`.
- EXP-302 recurrence stability/depth extrapolation — blocked on EXP-301.
- EXP-303 adaptive halting — optional, blocked on recurrence survivor evidence.
- EXP-304 event-driven routing — optional, blocked on recurrence survivor evidence.
- EXP-305 router generalization — blocked on EXP-304.
- EXP-306 native predictive uncertainty — optional, blocked on recurrence survivor evidence.
- EXP-307 induced-state deviation recovery — blocked on EXP-306.
- EXP-308 active information/self-locking — optional, blocked on EXP-307 or an independently preregistered uncertainty survivor.
- EXP-309 oracle conflict-value replication — blocked on recurrent-core survival.
- EXP-310 counterfactual conflict credit — blocked on positive EXP-309 headroom.
- EXP-311 relational/hypergraph conflict representation — only if EXP-310 leaves material unexplained headroom.
- EXP-312 soft episode anti-pattern memory — blocked on a safe conflict representation.
- EXP-313 reversible structural conflict transfer — blocked on safe local memory.
- EXP-314 student-reachable privileged information — blocked on survivor core.
- EXP-315 joint PI distillation versus naïve auxiliary — blocked on reachable privilege.
- EXP-316 scaffold dropout and fresh-process removal — blocked on EXP-315.
- EXP-317 cross-domain native fidelity transfer — blocked on native scaffold removal.
- EXP-318 integrated survivor 10M court — blocked until all included modules have positive scoped authority.

No code for EXP-302..318 is needed to answer EXP-301 and should be built preemptively.

## 12. EXP-301 frozen court

Question: at a matched 10M-class resident-parameter budget and matched accounted inference-compute budgets, does a shared recurrent latent core provide reproducible verified-generalization headroom over the strongest fixed-depth and simple recurrent rivals without protected-domain regression?

Arms:

- `A_FIXED`: strongest fixed-depth 10M Transformer rival;
- `B_LOOP_SIMPLE`: simple weight-tied fixed-loop recurrent rival;
- `C_NRS_CORE`: V0.17 shared recurrent core with loop conditioning but no optional sidecars.

Frozen roots: `[0,1,2,3]`.

Frozen training loops: `[1,2,4,8]`.

Frozen challenge loops: `[1,2,4,8,12,16]`.

Primary decision rule requires all of:

1. all protected floors pass;
2. aggregate `RCG >= +0.05` versus `A_FIXED`;
3. at least two reasoning families improve verified success by at least +5 percentage points at one common compute budget;
4. language/sequence control drops by no more than 2 percentage points;
5. directionally positive result on at least 3 of 4 roots;
6. 95% paired hierarchical-bootstrap CI for aggregate RCG excludes zero.

If the valid court fails this conjunction, `H-RD-01` is killed. It may not be rescued by changing loop schedule, thresholds, roots, task weights, sample count, or compute accounting after outcome.

The complete machine-readable registration is `protocols/v017/exp301_preregistration_v1.json`.

## 13. Scale constitution

V0.17 cannot authorize a direct 100M build.

If EXP-301 fails, the central current V0.17 lineage stops before sidecar construction. If recurrence survives but every optional module fails, a pure recurrent 10M model remains a legitimate survivor architecture. If optional mechanisms survive, EXP-318 integrates only those survivors and compares them against the strongest simpler 10M rivals under matched resources.

A positive EXP-318 may grant at most:

`AUTHORIZE_30M_DESIGN_ONLY`

It does not authorize a 30M run automatically and never authorizes 100M directly.

## 14. W5 process boundary

Nolane World 0.12.0 was used as a research-process constraint, not as capability authority. The W5/depth-5 Research MicroVM compiled a 22-stage program. At freeze:

```text
r1 question-certificate       COMPLETE
r2 evidence-map               COMPLETE
r3 counterexample-retrieval   COMPLETE
r4 assumption-stress          COMPLETE
r5 rival-hypotheses           COMPLETE
r6 falsification              COMPLETE
r7 preregistration            COMPLETE
r8 decisive-experiment        CURRENT / NOT RUN
r9..r22                       PENDING ON EMPIRICAL EVIDENCE
```

Advancing r8..r22 without real EXP-301 evidence is forbidden process theater. Later claim-provenance, replication, nondeterminism, benchmark-validity, external-validity, adversarial-review and closure stages must consume actual experiment artifacts.

## 15. Current epistemic status

What is established now:

- V0.17 is a scientifically distinct response to V0.16.1 evidence;
- the 10M geometry and parameter ledger are concrete;
- optional mechanisms face a generic-capacity rival and explicit kill paths;
- the experiment graph is dependency-aware;
- EXP-301 is preregistered before outcome;
- W5 correctly stops at the decisive-experiment boundary.

What is not established:

- recurrence beats Transformers;
- NLM V0.17 is a complete general-purpose AI;
- any optional sidecar helps;
- conflict transfer is safe;
- uncertainty is calibrated/useful;
- privileged information can be internalized;
- scaffold-free fidelity will pass;
- scaling to 30M or 100M is justified.

The correct status is:

`V0.17 DESIGN FROZEN FOR REVIEW / EXP-301 PREREGISTERED / ARCHITECTURE PERFORMANCE UNVERIFIED`

## 16. Review-to-implementation gate

After human review/approval of this design and the full W5 specification, the only next artifact is an EXP-301 implementation plan. That plan must freeze the exact parameter compiler, active CEE control, baselines, optimizer/search budget, generator manifests, split rules, metric implementation, FLOP accounting, challenge freeze ceremony, code/data identities, root list, rerun policy and analysis digest.

Only after that plan is reviewed may test-driven implementation of EXP-301 begin.
