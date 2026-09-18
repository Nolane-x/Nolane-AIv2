# EXP-319R Stage-A Finalization Recovery

Status: procedural recovery / no scientific retuning

## Authority

This recovery is bound to the already-computed authoritative EXP-319 run:

- source run: `35311529822`
- failed job: `105499052525` (`stage-a-gate`)
- repaired marker: `124584061616ab0864355219645ed00c1298c575`
- frozen scientific source: `2002a42322c7b919c3c9dc3da7d9cb0f546d4431`
- source-tree digest: `ca4970a96b9eb9cb2a9b895683b8f623160b1d134e25e0e91ace6172c38b18a6`
- frozen workflow SHA-256: `72b84f2fe32a7ded194b98620ac7e7701e0d18fdbb683aa4b3759145c268e237`
- training-contract digest: `cd2997f1899e21d3eee6f59e968bb4313db1d439258b6575d32357fe6c53a4e1`
- scoring-contract digest: `13c6d3520330457135bd167a7f8c14806bc27ef3d445e1511c4adbeb0be4b0c1`
- execution digest: `675655dfec9499ac8c49f3429771d1c5456e7ee652a3d7f309117bbc8691c9ba`
- Stage-A selection artifact id: `10533822077`
- Stage-A selection artifact digest: `sha256:ee65f07a5f63c5b7f472eba5aea5ab17a624697138c23266d515810024bfb559`

## Observed procedural defect

The frozen `stage-a-gate` step wrote `continue_stage_b` to `$GITHUB_OUTPUT` and then consumed
`${{ steps.gate.outputs.continue_stage_b }}` later inside the same shell step. GitHub evaluates that expression
before the step finishes, so the expression materialized as the empty string. The frozen root-evidence construction
did execute, but the finalizer command was skipped. The subsequent artifact upload correctly observed the completed
step output (`false`) and failed because `stage-a-stop/final.json` had never been created.

The Stage-B negative gate contained the analogous latent defect for `continue_stage_c`.

## Recovery rule

Do not rerun training. Do not rerun LR selection. Do not change thresholds, model geometry, data, optimizer budget,
scoring, reducer semantics, or authorization rules.

The recovery workflow must:

1. verify the sealed marker and frozen workflow bytes;
2. query the exact Stage-A selection artifact from source run `35311529822` and verify its immutable GitHub artifact digest;
3. download only that sealed selection artifact;
4. reconstruct exactly the four Stage-A-stop root evidence objects using the frozen `_seal_root_evidence` implementation;
5. invoke the unchanged frozen `scripts/exp319_finalize.py`;
6. independently verify final/root canonical digests, provenance fields, Stage-B/C absence, and all authorization flags;
7. emit a recovery receipt and a distinct recovered-final artifact;
8. report the recovery run/artifact identities back to draft PR #87.

The recovered artifact is post-processing of immutable evidence from run `35311529822`; it is not a second scientific run.

## Scientific boundary

EXP-301 remains `KILL_H_RD_01`.

This recovery cannot authorize:

- EXP-302 implementation;
- EXP-320 implementation;
- 30M;
- 100M;
- any scale-up.

The final EXP-319 disposition is authoritative only after the unchanged frozen finalizer emits `final.json`.
