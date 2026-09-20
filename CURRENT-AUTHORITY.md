# Nolane-AIv2 V0.17 — Current Authority Snapshot

> **Repository-process snapshot only.** This file records the current evidence/authorization state. It is not part of any sealed scientific source.

## Active lineage

The active research lineage remains **V0.17 / 10M Native Recursive Substrate**.

Resident scale remains exactly **10,000,000 trainable parameters**. No scale transition is authorized.

## EXP-335 — CLOSED

Authoritative EXP-335 run `35445927525` closed with:

`AFIXED_FULL32_FOUNDATION_REENTRY_RESCUED_NO_REGRESSION`

Independent frozen audit run `35477337239` succeeded.

EXP-335 authorized only the next C_NRS_CORE full-32 Stage-A design/preregistration transition, which became EXP-336.

Machine-readable closure:

`protocols/v017/exp335_final_closure_v1.json`

## EXP-336 — CLOSED

EXP-336 tested whether the exact EXP-319 selected **C_NRS_CORE** 10M checkpoint could establish the full 32-world Stage-A foundation under continued training, with the exact EXP-335 inherited projector evaluated prospectively.

### Frozen authority

- authoritative run: `35481336946`
- event: `workflow_dispatch`
- run attempt: `1`
- run conclusion: **SUCCESS**
- sealed head: `08e2caebca4c1b7e4b197f6d2190ef90e94468fd`
- scientific source: `5220e50adec7f6f048510aac44a077a4ddfdaf44`
- source-tree digest: `6485fdc131dbf0358f8e6548b3ec510ce96bb80b8fbaf2858a97d5bade62a911`
- preregistration digest: `491899f5f613b3d80b36689e7882e8c72875fb268a6886e1a40542eb09dc33ce`
- execution digest: `f81f382846d1c128934be40717e6eee33c48fe00dde00fd736a2ad2fdb15d5db`
- scientific workflow SHA-256: `b29710b206fb028f9064040f593ffd79ef4cdf6082fd820b3869d624b447de97`

Exactly one authoritative workflow-dispatch exists on the sealed head.

### Exact parent authority

EXP-336 starts from the exact EXP-319 selected C_NRS step-1024 checkpoint:

- EXP-319 run: `35311529822`
- selected artifact: `10534076546`
- parent ZIP SHA-256: `d88a9d85584476ec5226321479560a1255f0d015122827d4be75f67a6f337755`
- checkpoint SHA-256: `bd58607a0e49bb32f45124689d11880a13040afa702bf0524cce115d947f650b`
- model-state digest: `01c2a0f16b3f84dfee1b6db749821cf09897de05c6ceb15e42274abcd5926a76`
- optimizer-state digest: `d18439109cc9e020153b33fa6c39d38d4d0e05f2262c6a9ee3598df208f1ddf9`
- RNG-state digest: `3d2d8e928c4e09f880efbd2edaac3df47c88bace04c0e270c4ec0639e65bdb95`

The model is exactly **10,000,000 trainable parameters**.

### Frozen court

Arms:

- `CONTROL_CNRS_FULL32`
- `SHAM_MEASURE_CNRS_FULL32`
- `SUBSPACE_PROJECT_CNRS_FULL32`

Continuation:

- start step: `1024`
- additional source updates: `1024`
- final step: `2048`
- exposures/world: `32`
- immutable chunks: `8`
- effort cycle: `1/2/4/8`
- LR: `3e-4`

PROJECT used exactly:

`g' = g - T pinv(T^T T) T^T g`

with float64 Gram, `pinv rtol=1e-12`, target norm² floor `1e-24`, and all 31 target gradients measured at the identical pre-update state.

## Final scientific result

Authoritative final artifact:

- artifact ID: `10601451258`
- raw ZIP SHA-256: `3ed55f98e028034b8ee80d447b11dab44466013fb360910fe3e163f00022b5bf`
- final JSON SHA-256: `e04b24a0fef1f00c35f8ad8c6549538d18642a249f955f2067294111f8288fcc`
- evidence digest: `c89f70f3555ad95892794821eddbcc158a2e7e9d7c4ef4fa02082a00acd6ccf7`

Final preregistered reducer disposition:

`CNRS_FULL32_STAGE_A_NOT_ESTABLISHED`

Therefore **C_NRS Stage A is not established by EXP-336**.

Final per-world pass counts:

- CONTROL: `14/32`
- SHAM: `14/32`
- PROJECT: `18/32`

Final aggregate Stage-A gates:

- CONTROL: **false**
- PROJECT: **false**

Final aggregate metrics:

- CONTROL token accuracy: `0.8732394366197183`
- CONTROL greedy exact: `0.4375`
- CONTROL EOS correctness: `1.0`
- CONTROL invalid output: `0.0`
- CONTROL loss fraction: `0.004028852176910027`
- PROJECT token accuracy: `0.9119718309859155`
- PROJECT greedy exact: `0.5625`
- PROJECT EOS correctness: `1.0`
- PROJECT invalid output: `0.0`
- PROJECT loss fraction: `0.003099444664217025`

PROJECT rescued five CONTROL failures:

- `iterative-grid-and-maze:3`
- `iterative-grid-and-maze:5`
- `iterative-grid-and-maze:7`
- `generator-heldout-abstract-transformation:0`
- `generator-heldout-abstract-transformation:2`

PROJECT regressed one CONTROL-passing world:

- `language-sequence-control:6`

PROJECT geometry:

- projection updates: `1010`
- projected targets: `10945`

The projector improved the final world-pass count, but it did **not** satisfy the preregistered 32/32 + aggregate Stage-A gate.

## Independent verification — COMPLETE

Frozen independent auditor:

`208be21e035a5064eb9d60a9269154768717f5b1`

### Live immutable 8/8 chain audit

- run: `35493516557`, attempt `3`
- result: **SUCCESS**
- artifact ID: `10601572905`
- artifact ZIP SHA-256: `bb1ba966c81d08e2d032630f80fd898ab9ba5eb0e3259bac8d1c90d5574c3303`
- live chain report SHA-256: `5a1bde40d053941a56529cd8cbb046e622ddf77c3dcd9db626c3e39315da442b`
- verified chunks: `8/8`
- complete chain: `true`
- final raw chunk7 ZIP SHA-256: `5a12a2e59e399bec993ee1597ddbf4f85546e1d5de0b26635cb1d7def5a9666d`

### Frozen final audit

The pre-registered default-branch `workflow_run` invocation did not instantiate after authoritative completion. The locked verifier bytes were **not modified**.

A post-result **process-only recovery invocation harness** was created solely to invoke the already-frozen auditor over the exact immutable authoritative artifacts. The harness has no scientific authority and did not change source, thresholds, projector geometry, reducer logic, preregistration, or transition rules.

Recovery audit:

- run: `35499263542`
- result: **SUCCESS**
- frozen auditor SHA: `208be21e035a5064eb9d60a9269154768717f5b1`
- immutable audit artifact ID: `10600869072`
- audit artifact ZIP SHA-256: `3df746fb3173f0dffd19109b18bb68d8ac107a2fc8afe24690516845d4f2eff0`
- independent final audit report SHA-256: `c184be9e6bfc24e26695447521a6f29dc79817c54b505102c9903cff264f913c`
- independent audit receipt SHA-256: `f3f1e71e440e4c72ed932aa455a50b1e0a825701f07765692f8ea23a8e27d55e`
- independently recomputed disposition: `CNRS_FULL32_STAGE_A_NOT_ESTABLISHED`
- independently recomputed evidence digest: `c89f70f3555ad95892794821eddbcc158a2e7e9d7c4ef4fa02082a00acd6ccf7`
- CONTROL/SHAM exact at all 8 chunk boundaries: true
- complete 8-chunk chain: true
- all downstream authorization flags: false

The independent auditor also verifies:

- every raw artifact ZIP against GitHub metadata;
- parent ZIP chain from selected C_NRS → chunk0 → ... → chunk7;
- bundle/checkpoint/receipt digests;
- final boundaries exactly equal independently audited chunk7 boundaries;
- final chunk artifact/bundle/parent arrays;
- preregistered reducer vectors;
- canonical final evidence digest.

## Final closure

Machine-readable closure:

`protocols/v017/exp336_final_closure_v1.json`

Repository closure commit:

`818fe44bccb1b8177bea4ba622dc6306775521ed`

Closure digest:

`c291b3f8f0af0952ce8bbbda8fd107e29d8bd916e4656f2d13cc2e9b70a53d48`

Closure file raw SHA-256:

`1081551dd1f68baa32b8228cbad3636dcfff02e26cfd694f8d409c51d61ba644`

Closure Git blob SHA:

`615f5d53ecb4f36d59f6a75515f670c7a8d08c4c`

## Frozen transition selected

The already-frozen pre-result transition matrix maps:

`CNRS_FULL32_STAGE_A_NOT_ESTABLISHED`

to:

`NEW_CAUSAL_HYPOTHESIS_PREREGISTRATION_ONLY`

Frozen rule:

> Preserve the negative result. Do not open Stage B; only a distinct preregistered causal hypothesis may follow.

Therefore:

- Stage A established: **false**
- Stage-B design authorized: **false**
- Stage-B implementation authorized: **false**
- Stage-C implementation authorized: **false**
- EXP-302 implementation authorized: **false**
- EXP-320 implementation authorized: **false**
- scale authorized: **false**
- 30M authorized: **false**
- 100M authorized: **false**

## Current repository authority

EXP-336 is scientifically and evidentially **closed**.

The previous hypothesis — that exact continuation of the EXP-319 C_NRS foundation plus the inherited EXP-335 projector would establish full Stage A — did not pass the preregistered gate.

No result should be reinterpreted as Stage-A success merely because PROJECT improved from CONTROL `14/32` to `18/32`.

## Next repository action

The only authorized next scientific action is:

**design and preregister a distinct causal hypothesis.**

That successor must:

1. preserve EXP-336 as a negative result;
2. explain the remaining failure surface, especially the algorithmic and language families;
3. not retune EXP-336 post hoc;
4. define a falsifiable mechanism before implementation;
5. freeze thresholds, causal predictions, controls and transition rules before execution;
6. remain at the current 10M scale unless a future frozen gate explicitly authorizes scaling.

Do **not** open Stage B, EXP-302, EXP-320, 30M or 100M from EXP-336.
