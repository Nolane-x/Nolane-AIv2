# EXP-323R — Materialized Reconstruction Procedural Repair

## Status

Procedural repair only. EXP-323 authoritative run `35343381108` produced no scientific evidence and no reducer disposition because both intervention jobs failed before any post-2048 optimizer update.

The common failure was:

`ValueError: EXP-323 exact step-2048 replay mismatch`

The failed run is therefore execution-invalid, not a scientific negative result.

## Evidence for the defect

The frozen EXP-323 source had already demonstrated one exact replay:

- replay-authority run: `35342079836`
- reconstructed model digest: `4d3b848d193e6e465473dab7346edaafa3ee0ffc870eac600c6c46952ca880fc`
- reconstructed optimizer digest: `9b32588f0bc63881aa974df54829c4cda1d9e4913aa0fcc40cff5d5f2cddeb79`
- reconstructed RNG digest: `e293380e9776eb9e373bd0c3d3ab6a8aa4d0ab3b6764dc582bc859efc12fca07`
- post-2048 optimizer steps: `0`

Yet the two scientific jobs in run `35343381108`, using the same frozen source, same Ubuntu image and same Torch version, both failed the same replay digest gate.

Hosted-runner metadata differed between executions. This establishes a procedural reproducibility risk: a transient replay can be bitwise exact on one hosted CPU and non-exact on another even when the code/image/package versions are unchanged.

## Repair boundary

EXP-323R MUST NOT change:

- A_FIXED architecture or 10M parameter count;
- tokenizer;
- 32-world population;
- canonical answers;
- answer-only objective;
- AdamW family or inherited moments;
- weight decay or gradient clipping;
- training effort cycle;
- intervention arms `HOLD_5E5` and `DECAY_2P5E5`;
- arm learning rates;
- checkpoints `2304 / 2560 / 3072`;
- teacher-forced floor;
- material-progress thresholds;
- reducer decision order;
- any EXP-302 / EXP-320 / 30M / 100M authorization.

The only permitted repair is to materialize the already-authoritative step-2048 DECAY state as an immutable checkpoint before scientific continuation.

## Phase A — bounded reconstruction pool

Exactly four independent standard hosted-runner attempts are launched.

Each attempt:

1. downloads the same sealed EXP-322 final evidence;
2. downloads the same immutable EXP-319 step-1024 checkpoint;
3. replays exactly global steps 1024 through 2047 at the frozen EXP-322 DECAY learning rate;
4. computes model, optimizer and RNG digests;
5. performs zero optimizer steps after 2048;
6. uploads a reconstruction checkpoint only if all three digests exactly equal the authoritative EXP-322 anchors.

A mismatch attempt is diagnostic only and is not an EXP-323 scientific observation.

The pool is bounded to four attempts. No fifth automatic attempt is permitted by this helper.

## Deterministic artifact selection

If no attempt produces an exact checkpoint, this repair remains blocked.

If one or more attempts produce exact checkpoints, selection is by the numerically lowest attempt index only. Selection cannot inspect scientific metrics because no post-2048 training or evaluation exists in Phase A.

Every exact candidate must encode the same authoritative model/optimizer/RNG digests. The selected artifact is subsequently bound by:

- helper run id;
- artifact id;
- artifact ZIP digest;
- checkpoint-file SHA-256;
- reconstruction receipt digest;
- the three authoritative state digests.

## Phase B — repaired scientific execution

The repaired scientific workflow MUST NOT replay steps 1024-2048.

Instead, both arms download the exact same selected reconstruction artifact, independently verify its bytes and receipt, load the identical model/optimizer/RNG state, and only then:

- HOLD_5E5 continues at 5e-5;
- DECAY_2P5E5 changes only AdamW param-group learning rate to 2.5e-5.

The first optimizer update executed by either repaired scientific arm is global step 2048.

## Run-1 disposition

Run `35343381108` is permanently recorded as:

`PROCEDURAL_REPLAY_REPRODUCIBILITY_FAILURE_NO_SCIENTIFIC_DISPOSITION`

It cannot be interpreted as evidence for or against either EXP-323 arm.

## Authorization boundary

All authorization remains false:

- `exp302_implementation_authorized=false`
- `exp320_implementation_authorized=false`
- `scale_authorized=false`
- `authorized_30m=false`
- `authorized_100m=false`
