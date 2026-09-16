# EXP-319 — Learnability Foundation Diagnostic

**Status:** `DESIGN / PRE-IMPLEMENTATION / HUMAN-REVIEW-REQUIRED`  
**Lineage:** scientifically distinct post-EXP-301 diagnostic  
**Design branch parent:** `cf994bc3cd95d581fb575815615fd99a2e5f3b01`  
**EXP-301 final scientific disposition:** `KILL_H_RD_01`  
**Scale authorization:** `false`  
**EXP-302 implementation authorization:** `false`  
**EXP-320 implementation authorization:** `false`

## 1. Why EXP-319 exists

EXP-301 was a valid negative court. Its final frozen reducer returned `KILL_H_RD_01` with eligible evidence, aggregate recurrent-compute gain `0.0`, bootstrap 95% CI `[0.0, 0.0]`, zero positive roots, and zero reasoning families clearing the preregistered gain threshold. Independent artifact audit also found zero verified successes for `A_FIXED`, `B_LOOP_SIMPLE`, and `C_NRS_CORE` at the primary efforts across all four roots.

That result kills the specific hypothesis that the frozen EXP-301 recurrent-depth setup established useful recurrent-compute headroom. It does **not** answer a more basic causal question: did the 10M learners acquire enough supervised task competence for an architecture comparison to be informative?

EXP-301 trained each trial for exactly 512 optimizer steps over 512 examples, one pass through the training set, using answer-only byte-level cross-entropy. Its scientific outcome remains binding; EXP-319 does not reinterpret or rescue it. EXP-319 changes the causal question from **architecture superiority** to **learnability diagnosis**.

The governing question is:

> Under a preregistered, bounded increase in supervised optimization exposure, can the frozen 10M learning stack first fit a tiny sanity distribution, then acquire nontrivial in-distribution competence, then transfer any of that competence to a fresh generator-heldout split?

Only after this foundation is demonstrated is it scientifically meaningful to spend another court on the long-term Nolane AI thesis of an always-active persistent neural substrate.

## 2. What EXP-319 is not

EXP-319 is not:

- a rerun of EXP-301;
- a rescue of `H-RD-01`;
- a recurrence-versus-Transformer winner court;
- a post-hoc loop-schedule search;
- a scale experiment;
- authorization for 30M or 100M;
- authorization for EXP-302 implementation;
- authorization to build continual plasticity, tool use, or public thought/status channels.

A positive EXP-319 can authorize **only the design/preregistration** of a new experiment, provisionally named `EXP-320 Always-Active Persistent-State Microcourt`. It cannot authorize EXP-320 implementation by itself.

## 3. Long-term architecture boundary

The intended Nolane AI architecture remains broader than EXP-319:

1. an always-active neural core that continues state transitions after boot even when no external prompt arrives;
2. persistent internal state so new events enter an already-evolving process instead of starting a fresh stateless response;
3. later, controlled online plasticity where selected neural-core parameters may change during operation;
4. later, separate semantic emission modes for private/internal cognition, human-visible `STATUS`/`FINAL`, and structured `TOOL.ACTION` outputs.

EXP-319 deliberately tests none of those properties. It establishes whether the present neural substrate and supervised stack can learn a bounded task at all before those additional variables are introduced.

## 4. Frozen model family for the diagnostic

The resident-parameter ceiling stays exactly 10,000,000 trainable parameters per arm. EXP-319 reuses the existing compiled 10M arms without changing their architecture:

- `A_FIXED`: fixed-depth Transformer control;
- `B_LOOP_SIMPLE`: simple weight-tied recurrent diagnostic control;
- `C_NRS_CORE`: current Native Recursive Substrate core.

The byte tokenizer, answer-only loss, AdamW optimizer family, weight decay `0.01`, gradient clipping `1.0`, and model parameter geometry remain unchanged unless the implementation plan later discovers a literal correctness bug. A correctness bug must be repaired in a separate provenance lineage and may not be silently folded into the scientific court.

`B_LOOP_SIMPLE` is diagnostic-only in the main decision tree. The minimum foundation gate is decided by `A_FIXED` and `C_NRS_CORE`; this avoids spending a third full confirmatory budget merely to rank recurrence variants.

## 5. Three-stage diagnostic geometry

EXP-319 has three sequential stages. Later stages cannot be used to retune earlier choices.

### Stage A — memorization / optimization sanity

Purpose: establish that the implementation, objective, optimizer, tokenizer, and model can fit a tiny supervised distribution.

Geometry:

- one tuning root: `root=0`;
- four task families;
- 8 examples per family, 32 examples total;
- deterministic repetition of those same 32 examples;
- all three arms;
- inherited learning-rate candidates `{1e-4, 3e-4}`;
- checkpoints at optimizer steps `{128, 512, 1024}`;
- training effort schedule remains the existing frozen `{1,2,4,8}` cycle for recurrent arms and matched fixed-arm accounting.

Metrics recorded at every checkpoint:

- answer-only cross-entropy;
- answer-token accuracy;
- exact-match rate on the 32 sanity examples;
- EOS correctness rate;
- invalid-token/output rate;
- gradient norm before clipping;
- parameter-update norm ratio;
- NaN/Inf count.

An arm passes Stage A at step 1024 only if all are true:

- exact-match `>= 0.90`;
- answer-token accuracy `>= 0.99`;
- EOS correctness `>= 0.95`;
- final answer-only loss `<= 0.25 * initial measured loss`;
- no NaN or Inf event;
- invalid-output rate `<= 0.01`.

For each arm, the Stage A learning rate used in later stages is the candidate with the higher exact-match at step 1024; ties use higher answer-token accuracy, then lower loss, then the lower learning rate. Stage A is explicitly a tuning/sanity population and never contributes confirmatory generalization evidence.

If `A_FIXED` fails Stage A at both learning rates, the scientific interpretation is `TRAINING_STACK_NOT_LEARNABLE`; no later stage runs. If `A_FIXED` passes but `C_NRS_CORE` fails, the interpretation is `NRS_LOCAL_TRAINABILITY_FAIL`; no always-active design is authorized.

### Stage B — in-distribution capability floor

Purpose: determine whether successful tiny-set fitting turns into useful supervised competence on a fresh nontrivial distribution.

Stage B starts from fresh model initialization. Stage A weights are never reused.

Geometry:

- confirmatory roots `{1,2,3,4}`;
- primary arms `A_FIXED` and `C_NRS_CORE`;
- 128 training examples per family, 512 total per root;
- 64 IID-development examples per family, 256 total per root;
- selected per-arm learning rate frozen by Stage A before any Stage B development score is read;
- optimizer checkpoints at cumulative steps `{512,1024,2048}`;
- deterministic cycling over the 512 training examples after the first pass;
- no augmentation selected from development outcomes.

At each checkpoint, record the full Stage A metric set plus train exact-match and IID-development metrics.

A root meets the Stage B capability floor for an arm only if, at one checkpoint selected by the frozen rule below:

- training exact-match `>= 0.90`;
- IID answer-token accuracy `>= 0.95`;
- IID family-balanced exact-match `>= 0.25`;
- at least 3 of 4 families have exact-match `>= 0.10`;
- EOS correctness `>= 0.95`;
- invalid-output rate `<= 0.01`.

Checkpoint selection is deterministic: choose the earliest checkpoint satisfying all Stage B floors; if none satisfies them, choose the checkpoint with highest IID family-balanced exact-match, then higher IID token accuracy, then lower loss, then earlier step. This rule is frozen before Stage B runs.

An arm passes Stage B cross-root only if at least 3 of 4 confirmatory roots meet the root-level floor.

If both primary arms pass Stage A but `A_FIXED` fails Stage B cross-root, the experiment returns `SUPERVISED_FOUNDATION_UNDERTRAINED` and stops. If `A_FIXED` passes while `C_NRS_CORE` fails, it returns `NRS_IID_GENERALIZATION_FLOOR_FAIL` and stops.

### Stage C — fresh generator-heldout transfer

Purpose: determine whether competence survives a generator/template shift rather than only repeated exposure to the Stage B training generator family.

Stage C uses the already-selected Stage B checkpoints; no further gradient update is allowed.

Geometry:

- same confirmatory roots `{1,2,3,4}`;
- primary arms `A_FIXED` and `C_NRS_CORE`;
- 128 heldout examples per family, 512 total per root;
- heldout generator/template identities must differ from Stage B train and IID-development identities;
- heldout materialization happens only after Stage B checkpoint selections are sealed;
- a run-bound challenge beacon deterministically fixes the heldout content;
- no model, optimizer, learning-rate, step-budget, tokenizer, prompt format, or scoring change is permitted after heldout materialization.

A root meets the Stage C transfer floor for an arm only if:

- heldout answer-token accuracy `>= 0.85`;
- heldout family-balanced exact-match `>= 0.05`;
- at least 2 of 4 families have nonzero exact-match;
- EOS correctness `>= 0.90`;
- invalid-output rate `<= 0.01`.

An arm passes Stage C cross-root only if at least 3 of 4 confirmatory roots meet this floor.

The threshold is intentionally a **foundation floor**, not a claim of useful general intelligence. Five-percent exact verified success on fresh heldout generators is sufficient only to show that the stack has moved materially beyond the all-zero regime that made EXP-301 uninformative about learnability.

## 6. Task-family complexity ladder

EXP-319 retains the four broad task families so the diagnostic remains relevant to the earlier court, but introduces an explicit complexity ladder.

### Family 1 — iterative grid/state transition

- Stage A: short deterministic move strings and small state ranges;
- Stage B: longer compositions drawn from the same generator grammar;
- Stage C: heldout composition lengths and generator templates, not merely new random seeds.

### Family 2 — algorithmic sequence transform

- Stage A: short sequences with one transformation;
- Stage B: short compositions of two or three operations;
- Stage C: unseen operation-order templates and heldout composition identities.

### Family 3 — abstract rule transform

- Stage A: directly inferable small mappings with repeated rule family;
- Stage B: multiple instances from the same rule grammar;
- Stage C: generator-heldout rule templates with the same observable interface.

### Family 4 — language/sequence control

- Stage A: short sort/reverse/control sequences;
- Stage B: longer sequences and disjoint content tokens;
- Stage C: heldout templates and vocabulary instances under the same deterministic verifier.

The implementation plan must freeze the exact generator parameters, complexity ranges, prompt grammar, and verifier before any scientific Stage B/C run.

## 7. Fresh-data and leakage rules

EXP-319 must not reuse EXP-301 challenge worlds, challenge nonces, prediction commitments, or scientific outcomes as training examples.

Allowed reuse:

- source code implementing unchanged arm geometry;
- tokenizer implementation;
- deterministic verifier logic when semantics are unchanged;
- task-family concepts;
- historical EXP-301 evidence as motivation only.

Required freshness:

- Stage A tuning examples use a new EXP-319 generator namespace;
- Stage B train/IID examples use roots and salts not used by Stage A;
- Stage C heldout templates are inaccessible to training and are materialized only after Stage B selections;
- all Stage C content IDs bind the run beacon, generator identity, root, family, and index.

## 8. Why token metrics are mandatory

EXP-301's primary verifier was exact success, and all arms ended at zero verified success. Exact-match alone cannot distinguish at least four failure modes:

1. the model learned essentially nothing;
2. it learned token-level regularities but could not close full answers;
3. it fit the training distribution but failed IID development;
4. it learned IID structure but failed generator-heldout transfer.

EXP-319 therefore records token accuracy, loss, EOS behavior, and exact-match at every diagnostic rung. These metrics are diagnostic evidence only; they are not retroactively substituted for EXP-301's frozen success criterion.

## 9. Decision tree

The EXP-319 final reducer may return only one of these dispositions:

- `INVALID_DIAGNOSTIC` — provenance, generator, execution, artifact, or scoring contract is invalid;
- `TRAINING_STACK_NOT_LEARNABLE` — `A_FIXED` cannot pass Stage A under either preregistered LR;
- `NRS_LOCAL_TRAINABILITY_FAIL` — control can memorize but `C_NRS_CORE` cannot;
- `SUPERVISED_FOUNDATION_UNDERTRAINED` — Stage A is valid but `A_FIXED` fails the Stage B cross-root capability floor;
- `NRS_IID_GENERALIZATION_FLOOR_FAIL` — `A_FIXED` passes Stage B while `C_NRS_CORE` does not;
- `NRS_HELDOUT_GENERALIZATION_FLOOR_FAIL` — `C_NRS_CORE` passes Stage B but fails Stage C cross-root;
- `FOUNDATION_READY_FOR_EXP320_DESIGN_ONLY` — both `A_FIXED` and `C_NRS_CORE` pass Stages A, B, and C under the frozen rules.

No disposition ranks the architectures. `FOUNDATION_READY_FOR_EXP320_DESIGN_ONLY` means only that the NRS core is sufficiently trainable to justify designing a separate always-active persistent-state experiment.

Every disposition must include:

- `exp302_implementation_authorized=false`;
- `exp320_implementation_authorized=false`;
- `scale_authorized=false`;
- `30m_authorized=false`;
- `100m_authorized=false`.

## 10. Standard-runner execution constitution

All initial execution remains on standard GitHub-hosted `ubuntu-latest` CPU runners. No larger-runner or Enterprise-specific compute is required by the design.

To avoid repeating the monolithic timeout failure, training must be resumable in bounded immutable chunks. The implementation plan must ensure:

- Stage A chunks are no larger than 256 optimizer steps;
- Stage B chunks are no larger than 512 optimizer steps;
- every continuation artifact binds model state, optimizer state, cumulative step, data cursor/order digest, RNG state digest, arm, root, LR, code identity, and parent artifact digest;
- a continuation can only append to its exact parent chain;
- no checkpoint from a different root, arm, LR, code identity, or run can be spliced into a chain;
- scoring jobs never mutate checkpoints;
- reruns cannot silently create a second accepted artifact for the same chain position.

These are procedural requirements, not implementation authorization.

## 11. Stage-gating and human visibility

Stage A is explicitly inspectable because it is a tuning/sanity stage.

Once Stage B begins, only operational metadata may be inspected before Stage C materialization. Human inspection of Stage B development scores may not be used to alter thresholds, generators, budgets, LRs, or prompts.

Once Stage C is materialized, its predictions and partial scores may not be inspected for tuning. The reducer consumes all required root evidence before the scientific disposition is treated as final.

## 12. Provenance requirements

The implementation must create a new EXP-319 identity namespace. It must never reuse an EXP-301 execution-identity schema as though this were the same court.

Required bindings include:

- parent historical EXP-301 cross-root artifact digest;
- parent repair provenance digest;
- EXP-319 source tree digest;
- exact generator-manifest digest;
- exact training/selection/scoring contract digests;
- standard-runner workflow digest;
- Stage A tuning receipt;
- Stage B checkpoint-chain receipts;
- Stage C heldout manifest and prediction commitments;
- four root evidence artifacts;
- one cross-root reducer artifact.

A provenance failure invalidates the diagnostic rather than becoming a negative capability result.

## 13. Relationship to the always-active roadmap

A positive EXP-319 authorizes only a written design/preregistration for `EXP-320 Always-Active Persistent-State Microcourt`.

That future court should ask a different question:

> With weights frozen during evaluation, can a persistent neural state continue meaningful controlled evolution during intervals with no external input, preserve task-relevant state across long tick horizons, and integrate a later event using the evolved state rather than reconstructing it from a fresh prompt?

EXP-320 should test persistent state before any online weight plasticity. If persistent-state evidence survives, a later separate experiment may introduce bounded plastic weights. Only after persistent state and plasticity each have independent evidence should a communication/action router for `THINK`, `HUMAN.STATUS`, `HUMAN.FINAL`, and `TOOL.ACTION` be tested.

This sequencing keeps the user's intended architecture intact while preserving causal interpretability:

`LEARNABILITY -> PERSISTENT ALWAYS-ACTIVE STATE -> BOUNDED ONLINE PLASTICITY -> HUMAN/TOOL EMISSION ROUTING -> INTEGRATED AGENT`.

## 14. Kill rules and anti-rescue rules

The following are forbidden after Stage B scientific execution begins:

- adding epochs or optimizer steps beyond the frozen 2048-step ceiling;
- introducing a third LR;
- changing tokenizer vocabulary or output grammar;
- changing exact-match normalization;
- relaxing EOS/invalid-output rules;
- swapping task-family weights;
- dropping a failing root or family;
- changing the Stage B/C floors;
- moving heldout examples into training;
- claiming architecture superiority from diagnostic token metrics;
- calling a failed EXP-319 a continuation of H-RD-01.

A failed disposition can motivate a **new materially different hypothesis**, but it cannot be rescued by local threshold or budget edits after outcome.

## 15. Success criteria for this design phase

The design phase is complete when human review agrees that EXP-319 correctly asks the foundational question and preserves the scientific boundary around the negative EXP-301 result.

Only after that review may an implementation plan be written. The implementation plan must enumerate exact files, tests, generator manifests, JSON schemas, chunk/continuation contracts, CLI surfaces, workflow stages, freeze ceremony, and reducer verification. No EXP-319 scientific code should be implemented before that plan is reviewed.

## 16. Current authorization state

At this document's creation:

- EXP-301 remains `KILL_H_RD_01`;
- H-RD-01 remains dead;
- EXP-302 implementation remains unauthorized;
- EXP-319 is design-only;
- EXP-320 is concept-only and unauthorized for implementation;
- 30M and 100M scaling remain unauthorized.

The only immediate next step after approval of this document is to write the EXP-319 implementation plan.