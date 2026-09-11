# EXP-279 rescue-anchor replay DEVELOPMENT plan

## Question

PR #44 establishes that forced branch execution has sparse but cost-effective exact-rescue value on the current hybrid weights, while PR #43 establishes that the deployable route head selects zero episodes. PR #37 already falsified per-batch equal-class-mass BCE. Its fallback on rescue-free batches still trained the route head toward all-negative targets, so rare positive pressure could be erased between rescue-containing batches.

This DEVELOPMENT-only intervention asks whether preserving prior training rescue representations as positive anchors prevents that sparse-label collapse without changing the frozen route threshold or using evaluation targets.

## Mechanism

- Keep the exact-rescue target: `stop fails AND forced branch succeeds`.
- Keep routing gradients scoped to the routing head only.
- Maintain a deterministic in-run replay list containing detached routing-state tensors for rescue episodes observed on prior augmentation-training batches.
- On every hybrid routing update, positives are current rescue episodes plus all prior rescue anchors; negatives are current non-rescue episodes (including both-fail, both-success, and harm cases).
- Give positive and negative classes equal loss mass whenever both exist.
- Before the first rescue anchor exists, a rescue-free batch applies zero routing gradient instead of repeatedly pushing the router toward all-stop.
- Append only augmentation-training rescue representations. No evaluation examples or targets may enter replay.
- No external examples, threshold changes, protocol changes, MESI changes, sample-size changes, protected-floor changes, challenge materialization, or confirmatory consumption.

## TDD

RED first:
1. negative-only current batch + replay positive must preserve positive pressure;
2. negative-only batch before any rescue anchor must produce zero routing gradient;
3. replay storage must contain only detached rescue states;
4. routing-supervision contract must explicitly declare replay provenance and frozen threshold.

GREEN only after those tests fail for the missing implementation.

## Scientific sweep

Run train replicates `15 / 60 / 120` with the existing frozen model/world geometry and threshold `0.5`.

Use two DEVELOPMENT evaluation lineages:
- legacy-comparability: `20000..20032`;
- fresh anti-overfit DEVELOPMENT holdout: `30000..30032`.

The fresh lineage is the primary decision surface for this intervention. The legacy lineage is descriptive comparability only.

Record route fraction, hybrid solution rate, utility/FLOP, best-simple contrast, final rescue-anchor count, and frozen-boundary receipts. If the mechanism only improves the reused lineage but not the fresh DEVELOPMENT holdout, treat it as falsified for generalization.

No confirmatory gate opens from this experiment.