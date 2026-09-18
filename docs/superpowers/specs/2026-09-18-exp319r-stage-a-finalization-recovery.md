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


## Sealed recovery result

The preferred seal-only recovery succeeded without rerunning training, inference, LR selection, or scientific selection.

### TDD evidence

- RED regression commit: `e0be38416252036c6fbc00a8131bb0b04b5a9565`
- RED helper run: `35314011526`
- RED Python 3.13 job: `105501689539`
- RED result: `1 failed, 302 passed, 122 skipped, 18 deselected`
- GREEN workflow-only fix: `1989d4bcb7c8ec92a474db7ca17e8dcbf6225ada`
- GREEN verification run: `35314122021`
- GREEN Python 3.11 job `105503298236`: `303 passed, 122 skipped, 18 deselected`
- GREEN Python 3.13 job `105503298263`: `303 passed, 122 skipped, 18 deselected`

### Recovery execution

- helper head: `592c913a2e04a159f9dd371dfe8cd887c73ee3f1`
- recovery run: `35315014390`
- recovery job: `105504743035` — SUCCESS
- PR execution SHA recorded by GitHub: `48534eccecf4e8e8b070038084fcf26a8a3056ad`
- recovered artifact: `exp319-final-recovered-run-35311529822`
- artifact id: `10534634312`
- artifact digest: `sha256:f98ba9dedd2d4d952fbc161c94497c86aed4d9e7726bd749e5bbae9c04e7c04a`

The source selection artifact was downloaded from authoritative run `35311529822` and GitHub independently verified its digest as
`sha256:ee65f07a5f63c5b7f472eba5aea5ab17a624697138c23266d515810024bfb559`.

### Frozen finalizer disposition

The unchanged frozen `scripts/exp319_finalize.py` emitted:

```text
TRAINING_STACK_NOT_LEARNABLE
```

Sealed output identities:

- final evidence digest: `4be2850f6c1d633af4cb6687e379c0011aaf8281075bea79821d0f2e038ffb07`
- `final.json` SHA-256: `97dd39ec093a23431b360a192171b06e626d88270775f7eada7ff7584e9533e3`
- recovery receipt digest: `83453cac34523871fbc605eb1d84202520f829fbc60e40cd11a19c5c73be954e`
- root-1 digest: `f826eb06962fbfd99d3a8fa0d9dcdc0f1e627406a3d13933616b75ee5c6d04cf`
- root-2 digest: `3464a6bd69e939ad328628b8cd55a0b5b88108f37d17f9ace6692e7e03a9c89c`
- root-3 digest: `ea886a165fda6bc0f683e50a7b67241bd2f8025b8ca348c14f22fa63bc7d3c6a`
- root-4 digest: `24446113ac2930e74551f7554cdcbce9263d2d92af22b5652e526d64f1460d3b`

The uploaded recovery artifact was downloaded again after the run and independently verified. Its ZIP SHA-256, final evidence digest, receipt digest, all four root digests, source-selection file digest, source run/marker/artifact binding, and authorization state all reproduced exactly.

Stage B and Stage C did not run. Their absence is preserved in the sealed evidence.

All authorizations remain false:

- `exp302_implementation_authorized=false`
- `exp320_implementation_authorized=false`
- `scale_authorized=false`
- `authorized_30m=false`
- `authorized_100m=false`

This recovery closes the procedural finalization gap only. It does not reinterpret EXP-301, rescue the V0.17 recurrent hypothesis, or authorize any scale-up or successor implementation.
