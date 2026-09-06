# NLM 100M + Stage-A Freeze Design

## Purpose
Bootstrap `Nolane-AIv2` as an executable research repository for NLM V0.16.1 without upgrading any UNVERIFIED neural claim to fact.

## Authoritative boundaries
- V0.16.1 closure contracts outrank historical definitions.
- The current candidate totals exactly 100,000,000 parameters.
- Machine-readable experiment IDs use `EXP-###`; evidence maturity uses `EV-*`.
- Six first gates are EXP-277, EXP-279, EXP-282, EXP-286, EXP-289 and EXP-297.
- The repository must prefer a simpler rival when practical equivalence holds.
- Raw negative outcomes and architecture-caused divergence are scientific results, not disposable infrastructure noise.

## Implementation shape
1. `nolane_ai.model`: authoritative budget registry plus an executable PyTorch scaffold. Each region has a functional path and an explicit unused-capacity reserve. The reserve is deliberate: unearned complexity is not silently implemented merely to fill the 100M budget.
2. `nolane_ai.protocol`: frozen Stage-A schema, deterministic domain-separated seed derivation, protocol digest verification and Evidence Packet identity helpers.
3. `protocols/stage_a_v1.json`: machine-readable six-gate contract. Numeric MESI values and sample-size rules introduced here are V1 engineering commitments, not claims copied from V0.16.1.
4. `tests`: assert parameter accounting, canonical IDs, seed separation, digest integrity, result-state vocabulary and model shape.

## Scientific non-claims
This bootstrap does not claim the 100M model is trained, superior, conscious, lifelong-plastic, or empirically validated. It creates the artifact boundary required to begin falsification.
