# Stage-A Neural Pilot 16M Design

## Purpose

Turn the executable EV-E2 reasoning substrate into a trainable 10–30M neural wind-tunnel without changing the frozen Stage-A v1 confirmatory protocol. The pilot exists to generate neural evidence candidates and failure traces, not to promote any claim by construction.

## Scientific boundary

- Target footprint: exactly 16,000,000 parameters.
- The pilot is a new execution substrate; the authoritative 100M candidate remains unchanged.
- `protocols/stage_a_v1.json` and `protocols/stage_a_v1.sha256` remain immutable.
- Pilot/CI/training-smoke output is at most `EV-E2` until a preregistered neural experiment consumes frozen data/world/config/evaluator artifacts.
- `capacity_reserve` parameters are accounting placeholders and MUST NOT be optimized or receive functional gradients.
- Checkpoints without full provenance are development artifacts, never confirmatory evidence.

## Model

The 16M pilot keeps only mechanisms needed for the first six Stage-A gates:

1. language/problem binding;
2. recurrent deliberation;
3. constraint-belief fabric;
4. conflict-core scoring;
5. semantic fidelity court;
6. verifier/output support;
7. explicit capacity reserve.

The same specialized region implementations used by the 100M scaffold are reused. The pilot has its own exact budget and does not silently inherit unrelated V0.16 subsystems.

## Functional optimizer boundary

Training utilities expose `functional_trainable_named_parameters(model)` and construct optimizers only from parameters that:

- require gradients;
- are not named `capacity_reserve`;
- belong to functional modules.

Tests must prove reserve parameters remain byte-identical across an optimizer step and have `grad is None`.

## Curriculum

`StageACurriculum` deterministically derives batches from the existing named RNG graph. A batch contains:

- variable states and constraint incidence;
- binary belief targets;
- conflict-core targets;
- paired semantic source/candidate vectors;
- fidelity labels;
- curriculum metadata and a content digest.

The first implementation is synthetic and explicit about that scope. It is intended for optimizer/gradient/path verification before stronger generator families are frozen.

## Training runner

`StageAPilotTrainer` performs bounded steps and reports per-loss telemetry, gradient norm, parameter counts, and seed lineage. A checkpoint manifest binds:

- protocol digest;
- code/source digest;
- pilot config digest;
- curriculum digest;
- RNG bundle;
- optimizer state identity;
- training step;
- model parameter accounting;
- loss telemetry;
- evidence status (`EV-E2`, `UNVERIFIED`).

## Checkpoint format

Checkpoint payloads separate tensor state from JSON manifest. The manifest is canonical-JSON hashable. The runner must refuse to label a checkpoint neural evidence without the protocol identity and all required provenance fields.

## CI

CI runs a tiny CPU training smoke with the same APIs, verifies reserve immutability, checkpoint lineage, deterministic curriculum generation, and exact 16M meta parameter count. CI does not train the 16M pilot to capability.
