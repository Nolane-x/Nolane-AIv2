# EXP-279 Prior-Corrected Sparse-Rescue Replay — DEVELOPMENT Preregistration

## Scientific question

Can rescue replay prevent rare-positive gradient erasure **without** destroying sparse-event calibration, when the router's frozen `0.5` threshold is interpreted through the sealed hybrid stop-vs-branch compute ledger?

This is a DEVELOPMENT-only intervention. It cannot promote EXP-279 and cannot consume confirmatory data.

## Frozen authority

- Protocol: `NLM-REASONING-STAGE-A-CONFIRMATORY-V1`
- Protocol status: `FROZEN_V1`
- Protocol SHA256: `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`
- Route threshold: `0.5`, unchanged
- Root seed: `20260906-exp279-paired-dev`
- Geometry: `d_model=64`, `hidden_size=48`, `target_parameters=500000`
- World geometry: batch 8, timesteps 4, variables 6, constraints 3, noise 0.05
- Optimizer: lr 0.002, weight decay 0.0
- Training replicates: 15, 60, 120
- Evaluation replicates per cell: 33

## New single-use DEVELOPMENT lineage

Primary evaluation IDs are preregistered as `40000..40032` **before any result from that lineage is observed**.

The previously consumed `30000..30032` rescue-anchor holdout must not be used for tuning, selection, or decision-making in this intervention. The legacy `20000..20032` lineage is not part of the primary decision surface and is intentionally omitted from this sweep.

## Intervention

The natural hybrid target remains exactly the canonical rescue event:

`stop_exact_failure AND forced_branch_exact_success`

The intervention stores detached routing states for prior augmentation-training rescue episodes and replays their routing scores. Unlike the falsified equal-mass replay mechanism, it does **not** treat the replay sample's class frequency as the population frequency.

For every training step:

1. Count the raw current augmentation-training rescue labels into a causal cumulative natural prevalence.
2. Form a case-control sample from the current raw batch plus replayed positive rescue states.
3. Convert the model's routing score back to an implied natural rescue posterior using the sealed compute-ledger economics transform.
4. Convert that natural posterior to the case-control sample posterior using the exact ratio between sampled odds and causal natural odds.
5. Apply ordinary BCE on the sampled labels using that corrected sample posterior.

Before the first rescue has ever been observed, the routing gradient is zero, but all raw negatives still count toward the causal natural prior. No evaluation examples or evaluation targets enter training.

## Economics transform

Let:

- `F_stop` = sealed hybrid stop accounted FLOPs per episode
- `F_branch` = sealed hybrid branch accounted FLOPs per episode
- `c = (F_branch - F_stop) / F_branch`
- `q` = natural rescue posterior

The routing score is:

`score = q / (q + c * (1 - q))`

Therefore the frozen `score > 0.5` routing rule is equivalent to:

`q > c / (1 + c)`

No calibration temperature, class weight, threshold, replay cap, decay, or hand-tuned economic multiplier is introduced. `c` is derived only from the sealed analytical compute ledger.

This economics transform is a DEVELOPMENT surrogate for branch opportunity cost; the actual decision surface remains the repository's primary verified utility-per-accounted-FLOP metric.

## Paired court

Each train cell runs canonical baseline and intervention with identical:

- initial model state
- augmentation-training batches
- evaluation batches
- pair-audit digest
- propagation-only final state and evaluation metrics
- branch-only final state and evaluation metrics

Any identity mismatch invalidates the cell before utility deltas are interpreted.

## Preregistered interpretation

The intervention is considered a surviving DEVELOPMENT candidate only if:

- at least one of train 60 or train 120 produces non-zero routing, and
- neither train 60 nor train 120 has lower mean hybrid verified utility per accounted FLOP than its paired canonical baseline.

Otherwise the branch is falsified and should close without merge. Train 15 is retained as a low-data diagnostic, not a rescue criterion.

Regardless of outcome, all artifacts remain `EV-E2`, `UNVERIFIED`, `scientific_evidence_eligible=false`, with no confirmatory consumption or promotion claim.
