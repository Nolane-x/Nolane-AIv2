# Stage-A Held-out Neural Evaluation Design

## Goal

Measure whether the exact-16M neural pilot changes capability proxies on deterministic data that is not used for gradient updates, without upgrading the evidence claim beyond EV-E2.

## Separation contract

- training generation uses the named `augmentation` RNG stream;
- held-out generation uses the named `evaluation` RNG stream;
- pre/post evaluation reuses the exact same held-out batch digests;
- evaluation executes under `torch.no_grad()` and restores model train/eval mode;
- evaluator never calls the optimizer and must not mutate parameters;
- report is bound to protocol, code, config, model-init, model-state, curriculum, and held-out batch lineage;
- the outer evidence artifact independently binds optimizer hyperparameters and train/eval geometry;
- class-support counts are regenerated from the exact evaluation stream and checked against held-out batch digests before the artifact is accepted.

## Metrics

For belief state: hard decision accuracy and Brier score.
For conflict localization: balanced accuracy and Brier score.
For semantic fidelity: balanced accuracy and Brier score.

Both aggregate metrics and paired per-batch deltas are retained. No single aggregate scalar may suppress a regression in another protected metric. Class-support counts are retained so balanced accuracy cannot be interpreted without knowing whether both classes were actually represented.

## Epistemic boundary

The generator is still synthetic and structurally related to the development curriculum. Therefore this artifact is a held-out development diagnostic only. It is capped at `EV-E2 / UNVERIFIED`; it cannot satisfy frozen confirmatory MESI or claim EV-E3 by construction.
