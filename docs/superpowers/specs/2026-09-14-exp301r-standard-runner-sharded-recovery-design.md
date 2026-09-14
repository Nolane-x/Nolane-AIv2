# EXP-301R Standard-Runner Sharded Recovery Design

## Status

Approved recovery design for the pre-outcome infrastructure failure of EXP-301 run `34823251660`.

## Context

EXP-301 attempt #1 executed the sealed marker commit `bac51c29c46e4c1fb3db5445a299da4674fdb6d8` on four `ubuntu-latest` root jobs. All four jobs reached the frozen scientific executor and were cancelled at the GitHub-hosted six-hour wall-clock limit. No root artifact was uploaded and the cross-root reducer never ran. Therefore this event is an infrastructure/resource failure and carries no `KILL_H_RD_01` or promotion disposition.

The recovery must not use Enterprise or larger runners. It must fit on ordinary GitHub-hosted `ubuntu-latest` jobs without weakening the experiment.

## Scientific invariants

EXP-301R MUST preserve all frozen scientific semantics from EXP-301:

- original frozen implementation digest `89ea87607b75061c0c3d426fdf07ba82cf1c0fb0ad62135c1106ccbaa5e8890c`;
- original marker commit `bac51c29c46e4c1fb3db5445a299da4674fdb6d8` and source commit `f724a18df5783d72131139cc365df0f84cccdcfb`;
- exact 10,000,000 active trainable parameters per arm;
- arms `A_FIXED`, `B_LOOP_SIMPLE`, `C_NRS_CORE`;
- roots `(0,1,2,3)`;
- two frozen LR trials per arm with the existing frozen model-init seeds;
- exactly 512 optimizer steps per trial;
- unchanged development selection rule;
- challenge size 512 worlds/root;
- efforts `(1,2,4,8,12,16)` with `(12,16)` remaining unseen-depth diagnostics;
- exactly 96 autoregressive forwards per challenge prediction;
- unchanged compute ledger and compute-match tolerance;
- unchanged protected floors, bootstrap procedure, promotion/kill thresholds and cross-root reducer;
- CPU execution (`EXP301_DEVICE=cpu`);
- no tuning flags, no sample-count reduction, no root reduction, no shorter decode budget, no altered task weights.

No existing EXP-301 scientific implementation, protocol, marker, reducer, or test file may be modified by the recovery lineage. Recovery code lives only in new `exp301r_*` files, new scripts, new tests, recovery docs, and the new recovery workflow.

## Recovery authority

EXP-301R is not represented as the original monolithic workflow. It is a resource-recovery execution envelope around the unchanged frozen scientific implementation.

Every recovery artifact binds:

- the original frozen implementation digest;
- the original failed run id `34823251660`;
- the recovery schema/version;
- the recovery beacon;
- the exact recovery workflow SHA-256;
- the relevant root/trial/shard coordinates;
- canonical artifact payload digests.

The final recovery envelope binds the unchanged frozen cross-root evidence digest and its exact reducer decision. Scientific claims MUST be reported as originating from `EXP301R-STANDARD-RUNNER-SHARDED-RECOVERY-V1` and MUST NOT claim that original run `34823251660` completed.

## Execution topology

### Stage 0: preflight

A preflight job verifies the original EXP-301 marker from Git history and verifies recovery semantic isolation. The branch diff from the marker commit may contain only recovery files and documentation. Existing EXP-301 files must be byte-identical to the marker commit.

### Stage 1: 24 independent frozen trials

Run one GitHub job for each `(root, arm, trial_index)` coordinate:

- 4 roots;
- 3 arms;
- 2 frozen LR trials.

Each job resolves its coordinate against `frozen_trial_plan(root=...)`, calls the existing `run_scientific_trial`, and emits the existing scientific checkpoint plus a recovery trial receipt. The recovery CLI selects a frozen coordinate; it does not expose scientific hyperparameters.

This stage deliberately uses one trial/job rather than one root/job so no single job inherits the six-hour failure mode. Trial independence is already frozen by exact model-init seed, root, arm, LR, 512 training steps and write-once checkpoint digest.

### Stage 2: four root selections

For each root, download all six frozen trial artifacts, validate every scientific checkpoint, rebuild the existing per-arm two-LR selection receipts, and emit the unchanged `RootSelectionManifest`.

Only the three selected checkpoints are repackaged into the root-selection artifact. The selection job performs no challenge materialization and sees no challenge outcome.

### Stage 3: four challenge manifests

For each root, materialize the complete 512-world challenge using one fresh recovery beacon derived only from GitHub workflow `run_id` and the fixed recovery schema, not `run_attempt`.

The stable run-id beacon makes retrying an infrastructure-failed shard reproduce the same challenge instead of silently changing the court. The challenge-manifest artifact contains identity/digest metadata and ordered world content IDs; it contains no model prediction or score.

### Stage 4: 64 challenge prediction shards

Use 16 contiguous canonical world shards per root, 32 worlds/shard. Each `(root, shard_index)` job:

1. validates the root selection artifact;
2. rematerializes the full challenge and matches it to the Stage-3 challenge manifest;
3. slices exactly its 32 canonical worlds;
4. loads the three selected models through the existing checkpoint validation path;
5. runs the existing `scientific_challenge_generate` for every world, arm and effort;
6. commits predictions with the existing `commit_prediction` function;
7. emits commitments only; it MUST NOT call the verifier or score challenge answers.

Each shard therefore contains exactly `32 * 3 * 6 = 576` prediction commitments, each with frozen generation-token count 96.

### Stage 5: four root sealers

For each root, download all 16 shard artifacts plus the selection and challenge-manifest artifacts. The sealer rejects missing, duplicate, overlapping or foreign shards.

Only after the complete commitment grid is proven present does the sealer invoke existing verifier scoring, build the existing runtime root identity, and build the existing `EXP301-ROOT-SCIENTIFIC-EVIDENCE-V1` artifact. The original evidence builder remains the authority for complete-grid, compute-match, challenge-rematerialization and digest checks.

The sealer additionally writes an `EXP301R-ROOT-RECOVERY-RECEIPT-V1` binding the original root-evidence digest to the recovery workflow and shard-set digests.

### Stage 6: unchanged cross-root reducer

Download all four root-evidence artifacts and audit them with the existing cross-root reducer. No recovery code may reimplement the scientific decision rule.

After the unchanged reducer emits `cross-root-evidence.json`, a recovery finalizer writes `EXP301R-CROSS-ROOT-RECOVERY-ENVELOPE-V1`, binding:

- all four recovery root receipts;
- all four root-evidence digests;
- the cross-root evidence digest;
- the exact frozen reducer decision;
- `prior_failed_run_id=34823251660`;
- `original_monolithic_run_completed=false`.

## Retry semantics

Infrastructure retry is allowed only for missing/failed recovery jobs within the same GitHub workflow run id. The recovery beacon excludes `run_attempt`, so successful artifacts and rerun shards remain in the same challenge identity.

A new workflow dispatch receives a new run id and therefore a new challenge beacon. Artifacts from different run ids/beacons may never be merged.

No retry may change scientific geometry or select only favorable results.

## Artifact schemas

New recovery-only schemas:

- `EXP301R-TRIAL-RECEIPT-V1`
- `EXP301R-SELECTION-RECEIPT-V1`
- `EXP301R-CHALLENGE-MANIFEST-V1`
- `EXP301R-PREDICTION-SHARD-V1`
- `EXP301R-ROOT-RECOVERY-RECEIPT-V1`
- `EXP301R-CROSS-ROOT-RECOVERY-ENVELOPE-V1`

Every schema uses canonical JSON SHA-256 digests and write-once output files.

## Failure policy

- Trial failure: infrastructure failure; no scientific disposition.
- Selection validation failure: invalid recovery execution; no scientific disposition.
- Challenge-manifest mismatch: invalid recovery execution; no scientific disposition.
- Missing/duplicate shard: invalid recovery execution; no scientific disposition.
- Shard generation timeout: infrastructure failure; rerun only the failed shard under the same run id when supported.
- Root evidence validation failure: invalid recovery execution; no scientific disposition.
- Reducer decision is the sole authority for `KILL_H_RD_01`, `PROMOTE_H_RD_01_TO_EXP302_DESIGN_ONLY`, `INVALID_COURT`, or recurrence-stability diagnostic disposition.

EXP-301R can never authorize EXP-302 implementation or 30M/100M scale directly.

## Verification gates

Before dispatch, all of the following must be green:

1. recovery isolation verifier;
2. recovery unit tests including shard completeness, overlap rejection, beacon stability and canonical merge order;
3. existing EXP-301 contract tests;
4. full repository CI at exact recovery head;
5. workflow static contract test confirming `ubuntu-latest`, CPU execution, 24 frozen trial coordinates, 16 shards/root and unchanged reducer invocation.

The original sealed EXP-301 branch and PR #75 remain unmodified by recovery implementation.