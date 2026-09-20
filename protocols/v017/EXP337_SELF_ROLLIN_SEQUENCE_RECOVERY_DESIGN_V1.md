# EXP-337 — C_NRS Self-Roll-In Sequence-Recovery Court

**Program:** Nolane-AIv2 V0.17 / 10M Native Recursive Substrate  
**Status:** DESIGN / PREREGISTRATION ONLY  
**Implementation authority:** **FALSE**  
**Stage-B authority:** **FALSE**

## 1. Why EXP-337 exists

EXP-336 is closed with the independently verified disposition:

`CNRS_FULL32_STAGE_A_NOT_ESTABLISHED`

The frozen transition selected by EXP-336 authorizes only:

`NEW_CAUSAL_HYPOTHESIS_PREREGISTRATION_ONLY`

Therefore EXP-337 must not reinterpret EXP-336, retune EXP-336 after the fact, open Stage B, or change model scale. It may only define a distinct falsifiable mechanism and freeze its test before implementation.

EXP-336 nevertheless gives a useful descriptive failure surface. Under the exact inherited projector, the final PROJECT arm reached:

- iterative-grid-and-maze: `7/8`
- algorithmic-sequence-transform: `2/8`
- generator-heldout-abstract-transformation: `8/8`
- language-sequence-control: `1/8`

The generator family is fully solved and iterative is nearly solved. The dominant remaining failure surface is the two sequence-output families. In language, the final PROJECT mean teacher-forced answer-token accuracy is `0.9340277777777778` while exact world success remains only `1/8`. Multiple failed language worlds have teacher-forced token accuracy in the `0.9167–0.9583` range but full-answer exact `0`.

This pattern is consistent with, but does not prove, an autoregressive prefix-state failure: the model is often locally close under a correct prefix but free-running sequence emission is brittle.

EXP-337 tests that causal hypothesis directly.

## 2. Parent authority

All EXP-337 arms must start from the exact EXP-336 final PROJECT checkpoint, not from a fresh initialization and not from the CONTROL arm.

Frozen parent:

- EXP-336 authoritative run: `35481336946`
- EXP-336 final decision: `CNRS_FULL32_STAGE_A_NOT_ESTABLISHED`
- EXP-336 evidence digest: `c89f70f3555ad95892794821eddbcc158a2e7e9d7c4ef4fa02082a00acd6ccf7`
- parent artifact: `exp336-chunk-7-35481336946`
- artifact ID: `10601286249`
- raw ZIP SHA-256: `5a12a2e59e399bec993ee1597ddbf4f85546e1d5de0b26635cb1d7def5a9666d`
- chunk bundle digest: `c5c72c4919077b0d1647939d4335cea79f579090af27fb2ff0a5e82e91d8cae8`
- PROJECT checkpoint member SHA-256: `a9e638f3029e4af05a685dd4cfba096da6107221ca3a31cb26e5422b4d394208`
- PROJECT receipt digest: `6831373eb293c6c32281e0c94059610af62f11e1da85cf7f80950d1cb415c04f`
- PROJECT model-state digest: `e122331caa2e6924c13d4c479574dc8d3e4e1de823adfc02604158ec3ad7fc38`
- PROJECT optimizer-state digest: `44db727abdf9edca155ebd1a3d9e697a4d0b0e1b82e70371e3a77768b72932a7`
- PROJECT RNG-state digest: `3d2d8e928c4e09f880efbd2edaac3df47c88bace04c0e270c4ec0639e65bdb95`
- starting cumulative step: `2048`

All arms load exactly the same model, optimizer, and RNG bytes.

## 3. Causal hypothesis

### H337 — Self-roll-in prefix recovery

After the exact EXP-336 projector has constrained cross-world destructive interference, the dominant remaining Stage-A failure is **train-test answer-prefix mismatch**:

- training uses canonical/gold answer prefixes;
- greedy inference conditions on the model's own previous tokens;
- one wrong byte can move later predictions onto a prefix state the model was not trained to recover from.

The intervention is parameter-neutral: expose the model during training to its own generated answer-prefix states and train it to recover toward the canonical target.

### What this hypothesis does **not** claim

EXP-337 does not claim that the problem is:

- insufficient resident parameter count;
- learning-rate retuning",
- a need for a new projector;
- family-specific hard-coded rules;
- a Stage-B capability failure;
- an argument for scaling to 30M or 100M.

## 4. Arms

Exactly three arms are frozen:

1. `CONTROL_PROJECT_GOLD_PREFIX`
2. `SHAM_SELF_ROLLIN_MEASURE_PROJECT`
3. `SELF_ROLLIN_RECOVERY_PROJECT`

All three start from the exact same EXP-336 PROJECT checkpoint and continue with the exact same inherited projector.

### CONTROL

Source loss is the unchanged gold-prefix answer-only loss.

### SHAM

The self-roll-in path and its gradient are fully computed, but that gradient is discarded. The applied source gradient is exactly CONTROL's gold-prefix source gradient.

CONTROL and SHAM must be exactly equivalent at every frozen boundary. Any mismatch invalidates capability interpretation.

### SELF_ROLLIN

Source loss is:

`0.5 * gold_loss + 0.5 * self_rollin_loss`

No extra optimizer step is allowed.

The active source gradient is projected using the same inherited EXP-336 subspace-projection rule as every other arm.

## 5. Exact self-roll-in construction

For one source world:

1. Encode the canonical prompt exactly as before:
   `[BOS, prompt bytes, SEP]`.
2. Let the canonical answer contain `n` byte tokens before EOS.
3. Using the **current pre-update model snapshot** and current exposure effort, greedily generate exactly `n` answer-prefix tokens with no gradient.
4. Do **not** stop early on EOS.
5. If the model emits EOS or any non-byte special token inside the roll-in, append that raw predicted token ID and continue. Positional alignment must remain fixed.
6. Form a recovery input conditioned on the generated prefix.
7. Target the canonical answer byte token at each aligned answer position and canonical EOS after the last answer byte.
8. Compute answer-only cross entropy on those canonical targets.

The roll-in mechanism applies to every world. No branch may special-case algorithmic or language examples.

This is a scheduled-sampling-style causal intervention, but its exact rule is frozen here rather than tuned from results.

## 6. Common projector

Every arm retains the exact EXP-336 projector:

`g' = g - T pinv(T^T T) T^T g`

Frozen geometry:

- Gram dtype: float64
- `pinv rtol = 1e-12`
- target norm² floor: `1e-24`
- targets: all other 31 worlds
- target order: frozen Stage-A world order excluding source
- target gradients: **gold-prefix answer-only gradients for every arm**
- all target gradients measured at the same pre-update model/optimizer/RNG state

Only the source loss differs between CONTROL and SELF_ROLLIN.

No post-hoc half-space, no-new-conflict, target filtering, or alternative projector is allowed.

## 7. Continuation budget

Resident parameters remain exactly:

`10,000,000`

No architecture or geometry change is permitted in EXP-337.

Continuation from parent cumulative step `2048`:

- additional exposures/world: `32`
- source updates: `1024`
- chunks: `8`
- exposures/world/chunk: `4`
- source updates/chunk: `128`
- final cumulative step: `3072`
- effort cycle: `1 / 2 / 4 / 8`
- optimizer: exact continued AdamW state
- LR: `3e-4`
- weight decay: `0.01`
- gradient clip: `1.0`

Frozen world schedule:

`for exposure 0..31 → family in frozen family order → world index 0..7`

No LR sweep or extension chosen after seeing results is permitted.

## 8. Evaluation gate — unchanged

EXP-337 is not allowed to weaken EXP-336's Stage-A definition.

### Full-32 teacher-forced floor

At final effort `4`, every one of the 32 worlds must simultaneously satisfy:

- answer-token accuracy `>= 0.99`
- full-answer exact `>= 0.90`
- nonfinite events `= 0`

### Aggregate EXP-319 Stage-A floor

Also required:

- greedy exact `>= 0.90`
- answer-token accuracy `>= 0.99`
- EOS correctness `>= 0.95`
- final loss fraction of original EXP-319 initial loss `<= 0.25`
- invalid output rate `<= 0.01`
- nonfinite events `= 0`

An arm passes only if **both** floors pass.

## 9. Diagnostic metrics that are not reducer input

The following may be recorded but cannot change the reducer or successor:

- per-family pass counts;
- per-family mean teacher-forced token accuracy;
- teacher-forced versus greedy-exact gap;
- self-roll-in prefix divergence rate;
- gold loss versus self-roll-in loss;
- first-error-position distribution;
- intermediate rescue/regression vectors.

Partial improvement is explicitly not positive reducer evidence.

## 10. Final reducer

Frozen reducer order:

1. `INVALID_EXP337_SELF_ROLLIN_COURT`
2. `EXP337_PARENT_AUTHORITY_MISMATCH`
3. `EXP337_SHAM_MEASUREMENT_MISMATCH`
4. `EXP337_STAGE_A_ESTABLISHED_BOTH`
5. `EXP337_STAGE_A_ESTABLISHED_CONTROL_ONLY_RECOVERY_REGRESSION`
6. `EXP337_STAGE_A_RESCUED_BY_SELF_ROLLIN_NO_REGRESSION`
7. `EXP337_SELF_ROLLIN_NO_CAUSAL_RESCUE`

### Positive causal rescue

The self-roll-in mechanism is causally established only if:

- CONTROL fails the complete final Stage-A arm rule;
- SELF_ROLLIN passes the complete final Stage-A arm rule;
- at least one CONTROL-failed world is rescued;
- no CONTROL-passing world regresses.

Anything weaker remains a negative Stage-A result.

## 11. Pre-result successor lock

Before implementation/result visibility:

- INVALID → validity/provenance review only
- parent mismatch → parent-authority reconciliation only
- SHAM mismatch → measurement-nonperturbation localization only
- both CONTROL and SELF_ROLLIN pass → Stage-B **design/preregistration only**, using the simpler CONTROL continuation
- CONTROL passes and SELF_ROLLIN fails → Stage-B **design/preregistration only**, CONTROL foundation; self-roll-in rejected
- only SELF_ROLLIN passes with no regression → Stage-B **design/preregistration only**, exact self-roll-in repair
- all other valid outcomes → another new causal-hypothesis preregistration only

No branch of this reducer authorizes Stage-B implementation directly.

## 12. Authorization guards

Still false:

- EXP-337 implementation
- Stage-B implementation
- Stage-C implementation
- EXP-302 implementation
- EXP-320 implementation
- scaling
- 30M
- 100M

The only thing this design authorizes is review/freeze of the preregistration itself.

## 13. Falsification discipline

EXP-337 must be considered a failed hypothesis if SELF_ROLLIN does not satisfy the exact positive reducer condition.

It is forbidden to rescue the claim by:

- changing the 0.5/0.5 loss mixture after results;
- changing the rollout length;
- stopping roll-in early on EOS after results;
- targeting only algorithmic/language families after results;
- changing the projector;
- extending training because an intermediate boundary looks promising;
- relaxing the 32/32 or aggregate Stage-A thresholds;
- calling an increase such as `18/32 → 25/32` a Stage-A success.

Negative results remain first-class evidence.

## 14. Implementation boundary

This document and `exp337_preregistration_v1.json` are **not implementation authorization**.

No EXP-337 runtime, workflow, scientific dispatch, checkpoint mutation, or scale transition may be added until:

1. this preregistration is committed;
2. its canonical digest and raw bytes are frozen;
3. CI validates internal consistency;
4. a separate explicit operator transition authorizes implementation of exactly this frozen court.
