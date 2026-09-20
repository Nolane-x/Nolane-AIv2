# Nolane-AIv2 V0.17 — Current Authority Snapshot

> **Repository-process snapshot only.** This file is maintained on the active V0.17 lineage for continuity. It is not part of any sealed scientific source.

## Active lineage

The active research lineage remains **V0.17 / 10M Native Recursive Substrate**.

Resident scale remains exactly **10,000,000 trainable parameters**. No scale transition is authorized.

## EXP-335 — CLOSED

Authoritative EXP-335 run `35445927525` closed successfully with disposition:

`AFIXED_FULL32_FOUNDATION_REENTRY_RESCUED_NO_REGRESSION`

Independent frozen audit run `35477337239` also succeeded and recomputed the complete 8-chunk chain, final reducer and evidence digest.

EXP-335 therefore authorized only the next **C_NRS_CORE full-32 Stage-A design/preregistration** transition.

Machine-readable closure:

`protocols/v017/exp335_final_closure_v1.json`

## EXP-336 — ACTIVE AUTHORITATIVE SCIENTIFIC EXECUTION

EXP-336 asks whether the selected **C_NRS_CORE** 10M checkpoint from EXP-319 can establish the full 32-world Stage-A foundation when continued from its exact step-1024 state, with the exact EXP-335 registered projector tested prospectively.

Frozen preregistration:

- preregistration head: `fe457d51ab8c01dc0c5d1831d9c83ce13e627c3c`
- canonical preregistration digest: `491899f5f613b3d80b36689e7882e8c72875fb268a6886e1a40542eb09dc33ce`
- exact preregistration JSON SHA-256: `6090e7423332756998b8199cc77d56b7e40a296b401f63dbcc6cabecf97feeab`

Separate post-freeze operator authorization was recorded before implementation:

`protocols/v017/exp336_operator_authorization_attestation_v1.json`

Frozen scientific source:

- source commit: `5220e50adec7f6f048510aac44a077a4ddfdaf44`
- source-tree digest: `6485fdc131dbf0358f8e6548b3ec510ce96bb80b8fbaf2858a97d5bade62a911`
- scientific workflow SHA-256: `b29710b206fb028f9064040f593ffd79ef4cdf6082fd820b3869d624b447de97`
- sealed execution digest: `f81f382846d1c128934be40717e6eee33c48fe00dde00fd736a2ad2fdb15d5db`
- sealed marker: `08e2caebca4c1b7e4b197f6d2190ef90e94468fd`

Sole authoritative scientific run:

- run: `35481336946`
- event: `workflow_dispatch`
- run attempt: `1`
- sealed head: `08e2caebca4c1b7e4b197f6d2190ef90e94468fd`
- status: **IN PROGRESS**
- authority lock: `protocols/v017/exp336_authoritative_run_lock_v1.json`

There are no observed duplicate EXP-336 workflow-dispatch runs. Any future duplicate is non-authoritative by the frozen authority rule.

## Exact C_NRS parent authority

EXP-336 starts from the exact EXP-319 selected C_NRS step-1024 checkpoint, not from A_FIXED and not from a fresh initialization.

- EXP-319 run: `35311529822`
- selected LR: `3e-4`
- selected artifact: `10534076546`
- artifact ZIP SHA-256: `d88a9d85584476ec5226321479560a1255f0d015122827d4be75f67a6f337755`
- checkpoint SHA-256: `bd58607a0e49bb32f45124689d11880a13040afa702bf0524cce115d947f650b`
- model-state digest: `01c2a0f16b3f84dfee1b6db749821cf09897de05c6ceb15e42274abcd5926a76`
- optimizer-state digest: `d18439109cc9e020153b33fa6c39d38d4d0e05f2262c6a9ee3598df208f1ddf9`
- RNG-state digest: `3d2d8e928c4e09f880efbd2edaac3df47c88bace04c0e270c4ec0639e65bdb95`

The real immutable parent artifact has been downloaded and reconstructed successfully in CI as exactly **10,000,000 trainable parameters**.

Parent Stage-A aggregate metrics at step 1024:

- answer-only loss: `2.149582388769822`
- answer-token accuracy: `0.7007042253521126`
- greedy exact: `0.25`
- EOS correctness: `1.0`
- invalid output: `0.0`
- nonfinite events: `0`

## Frozen EXP-336 court

Arms:

- `CONTROL_CNRS_FULL32`
- `SHAM_MEASURE_CNRS_FULL32`
- `SUBSPACE_PROJECT_CNRS_FULL32`

Training continuation:

- start step: `1024`
- additional source updates: `1024`
- final step: `2048`
- 32 additional exposures/world
- 8 immutable chunks
- 4 exposures/world/chunk
- 128 source updates/chunk
- effort cycle: `1/2/4/8`
- learning rate: `3e-4`
- weight decay: `0.01`
- gradient clip: `1.0`

PROJECT inherits exactly:

`g' = g - T pinv(T^T T) T^T g`

with float64 Gram, `pinv rtol=1e-12`, target norm² floor `1e-24`, all other 31 gradients measured at the identical pre-update state, and the same current exposure effort.

Teacher-forced evaluation effort is exactly `4`. Aggregate Stage-A gating also uses effort `4`.

Final arm success requires both:

1. every one of the 32 worlds simultaneously passes teacher-forced token >= `0.99` and full-answer exact >= `0.90`;
2. aggregate Stage-A floor passes greedy exact >= `0.90`, token >= `0.99`, EOS >= `0.95`, loss fraction <= `0.25`, invalid <= `0.01`, nonfinite = `0`.

## Verified authoritative progress

Identity artifact:

- artifact: `10595334487`
- ZIP SHA-256: `fcf481c6850c83d5f730bd0c2bc3cb2cc6ffcce55abfc15eec527a72dc41b5f7`
- exact identity JSON SHA-256: `a8f81a0494ba00658d0e8eeea4c0112f1a0547e436783a557b1e70ec9091a190`
- execution digest independently recomputed: true

Chunk 0:

- artifact: `10596866141`
- raw ZIP SHA-256: `176eedc34450c7e843ff4a21c218c82dea7fbc3fc15239559d4941e60bf69dbf`
- exposure/world: `4`
- cumulative training step: `1152`
- CONTROL/SHAM exact: true
- nonfinite: 0
- interim CONTROL world-pass count: `9/32`
- interim PROJECT world-pass count: `8/32`

Chunk 1:

- artifact: `10597253461`
- raw ZIP SHA-256: `7fd5d9536438109896eca9922ff80704483aee452dc1061f28c58dede88a6a36`
- parent digest equals actual raw chunk0 ZIP SHA: true
- exposure/world: `8`
- cumulative training step: `1280`
- CONTROL/SHAM exact: true
- nonfinite: 0
- interim CONTROL world-pass count: `10/32`
- interim PROJECT world-pass count: `11/32`
- PROJECT cumulative projection updates: `256/256`
- PROJECT cumulative projected targets: `2917`

Chunk 2:

- artifact: `10597562011`
- raw ZIP SHA-256: `8063277a4f0a99c34d8c4491eb57606717f68ee93f46212eea582f21c696c784`
- parent digest equals actual raw chunk1 ZIP SHA: true
- exposure/world: `12`
- cumulative training step: `1408`
- CONTROL/SHAM exact: true
- nonfinite: 0
- interim CONTROL world-pass count: `9/32`
- interim PROJECT world-pass count: `14/32`
- PROJECT interim rescues: iterative `:5,:7`; generator `:0,:2,:3`
- PROJECT interim regressions: none
- CONTROL aggregate token / greedy exact: `0.78169 / 0.28125`
- PROJECT aggregate token / greedy exact: `0.75352 / 0.4375`
- aggregate Stage-A pass: false in all arms
- PROJECT cumulative projection updates: `382/384`
- PROJECT cumulative projected targets: `4362`

Frozen independent audit through chunk2:

- workflow run `35486348709`, attempt `2`: **SUCCESS**
- artifact `10598291494`
- artifact ZIP SHA-256: `ddc7c708d8a078046bb33ef16e7103ec8a3ac33b818b6b83f2c3e6af01906176`
- report JSON SHA-256: `10cc46e0d5dd826fafd1415b913a81217b8b89ded62842614141cc269c52a3d8`
- verified consecutive chunks: `3`
- complete chain: false (correct; run is active)
- independent CONTROL/PROJECT pass trajectory: `9/8 → 10/11 → 9/14`

Chunk2 post-hoc geometry:

`protocols/v017/exp336_chunk2_exploratory_geometry_v1.json`

This diagnostic remains `reducer_input=false`.

Current job:

- chunk3 job: `106014853802`
- status: **IN PROGRESS**
- active step: `Execute EXP-336 chunk 3 court`

**These chunk metrics are non-decisional. They are not reducer input for selecting a repository transition.**

Machine-readable progress:

`protocols/v017/exp336_authoritative_progress_v1.json`

Post-hoc projector geometry diagnostics:

`protocols/v017/exp336_chunk0_1_exploratory_geometry_v1.json`

Those diagnostics have `reducer_input=false`.

## Independent final verification — ARMED

Frozen independent auditor:

- PR: `#114`
- auditor SHA: `208be21e035a5064eb9d60a9269154768717f5b1`
- auditor source-tree digest: `12163069da0a73400a0752f38926652f1bcc076971ee3c695bb9ae2983b4744c`
- dedicated EXP-336 contract: **SUCCESS**
- dedicated tests: **19 passed**
- real parent-smoke: **SUCCESS**
- generic core 3.11: **SUCCESS**
- generic core 3.13: **SUCCESS**
- generic model-smoke: **SUCCESS**

Frozen-auditor lock:

`protocols/v017/exp336_independent_audit_lock_v1.json`

Real independent live audit through chunk1:

- frozen auditor actually executed: `208be21e035a5064eb9d60a9269154768717f5b1`
- workflow run: `35486210011`
- audit artifact: `10597830566`
- audit artifact ZIP SHA-256: `699115d89a91c2431b216addf10555f803f49e3026a75d9991adb1dd63fcd717`
- audit report JSON SHA-256: `87dcaf3fb25ddcffa1420bb3c9a6e5d0d2184be91c002fbb6f5934f5cf324668`
- verified chunks: `2`
- complete chain: false (correct; authoritative run is still active)
- selected C_NRS parent → chunk0 → chunk1 raw ZIP chain: verified
- CONTROL/SHAM exact: true for both completed chunks
- independent pass counts:
  - exposure 4: CONTROL `9`, SHAM `9`, PROJECT `8`
  - exposure 8: CONTROL `10`, SHAM `10`, PROJECT `11`
- aggregate Stage-A pass: false in all arms at both intermediate boundaries

Default-branch post-run verifier:

`.github/workflows/exp336-postrun-independent-audit.yml`

Registration commit:

`4f6d29e5da76815fbbb02f58edc0d9823e6829a1`

It is hard-bound to authoritative run `35481336946`, sealed marker `08e2caeb...`, run attempt 1, and frozen auditor SHA `208be21e...`.

On authoritative success it must independently re-download/hash all 8 raw chunk ZIPs + identity + final, rebuild the chain, bind final boundaries to audited chunk7, rerun the preregistered reducer, and recompute the final evidence digest.

On authoritative non-success it may only archive a fail-closed provenance receipt.

## Pre-result transition lock

The successor action for every possible final reducer disposition was frozen **before scientific result visibility**:

`protocols/v017/exp336_post_result_transition_lock_v1.json`

Intermediate chunks cannot choose a successor.

A positive Stage-A result may authorize only a future **Stage-B DESIGN/PREREGISTRATION** transition according to that already-frozen matrix. It does not authorize Stage-B implementation.

## Current authorization guards

The following remain **false**:

- Stage-B implementation
- Stage-C implementation
- EXP-302 implementation
- EXP-320 implementation
- scale authorization
- 30M authorization
- 100M authorization

Persistent always-active state remains blocked until all required foundation / Stage-B / Stage-C evidence gates close positively.

## Next repository action

Do **not** change or rerun the sealed EXP-336 science.

Continue only by:

1. monitoring authoritative run `35481336946`;
2. as each immutable chunk appears, independently verify the actual raw ZIP digest, parent chain, receipts/checkpoint member SHA values, CONTROL/SHAM exactness, frozen identity, and authorization flags; the frozen auditor has already performed this successfully through chunk1;
3. treat all intermediate metrics as non-decisional;
4. after the complete 8-chunk chain and final artifact exist, require the frozen independent post-run auditor to recompute the final reducer;
5. apply only the already-frozen EXP-336 post-result transition matrix.

Do not open Stage B implementation, EXP-302, EXP-320 or any scale transition.
