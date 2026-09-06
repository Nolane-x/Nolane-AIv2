# EXP-282 Sealed Confirmatory-Open Executor Design

## Status

Approved continuation of the EXP-282 Stage-A evidence pipeline. This subsystem executes **confirmatory-open** observations only after a valid preparation, Execution Court authorization, and Reconstruction Court authorization exist. It does not execute post-freeze challenge data and cannot promote EV-E3 by itself.

## Goal

Execute the exact frozen EXP-282 recurrent-hidden vs explicit-belief comparison on the exact reserved confirmatory replicate IDs using the exact authorized checkpoint and reconstruction geometry, while producing an append-only raw evidence artifact that cannot silently substitute development data, change the model, select replicates, or reinterpret the frozen protocol.

## Inputs and authority

The executor consumes:

1. frozen `protocols/stage_a_v1.json` plus its SHA-256 authority file;
2. a valid `NLM-EXP-282-PAIRED-DEV-EVAL-V1` artifact;
3. a valid `NLM-EXP-282-CONFIRMATORY-RECONSTRUCTION-AUTH-V1` artifact;
4. the exact paired functional-only checkpoint file whose SHA-256 is bound by Reconstruction Court;
5. the executor source-tree digest.

The frozen protocol remains immutable. The executor does not derive or modify MESI, sample size, endpoints, multiplicity family, or decision rules.

## Seed and world separation

Development model-training lineage and confirmatory-open world lineage are deliberately different concepts.

- Model weights are reconstructed from the authorized checkpoint. The development `root_seed` inside the checkpoint is provenance for how those weights were created.
- Confirmatory-open worlds use the **frozen protocol RNG root seed** (`protocol.rng.root_seed`) and only the exact reserved replicate IDs from Reconstruction Court.
- Confirmatory-open observation generation uses the protocol's existing `evaluation` stream. It does **not** use `augmentation` and does not derive challenge randomness.
- `environment`, `intervention`, and `evaluation` seeds are derived through the existing `derive_stream_seed` contract.
- The future `POST_FREEZE_CHALLENGE` lane remains unavailable until a public beacon is supplied after code/config/evaluator/analysis freeze.

No development training/evaluation replicate ID may appear in the reserved confirmatory lineage. The executor requires the complete reserved ID sequence; subsets, additions, reordering, or duplicates are invalid.

## Checkpoint reconstruction

Before any confirmatory world is materialized, the executor must:

1. validate paired development and reconstruction authorization semantics;
2. hash the checkpoint file and match `paired_checkpoint_sha256`;
3. load the checkpoint on CPU;
4. require schema `NLM-EXP-282-PAIRED-TENSORS-V1` and `state_policy=functional-only`;
5. reject any checkpoint tensor key containing `capacity_reserve`;
6. recompute `execution_contract_digest` from the checkpoint's `execution_contract`;
7. require the checkpoint contract to match the Reconstruction Court contract exactly;
8. reconstruct recurrent-hidden and explicit-belief arms from exact `d_model`, `hidden_size`, and `target_parameters`;
9. load only functional tensors, permitting missing keys only for reconstructed `capacity_reserve` parameters and permitting no unexpected keys;
10. keep both arms in evaluation mode and never construct an optimizer.

## Confirmatory execution

For each reserved replicate ID, in the exact authorized order:

1. instantiate the partial-observability batch using frozen protocol root seed, exact authorized world geometry, and `rng_stream="evaluation"`;
2. evaluate both arms under `torch.no_grad()`;
3. retain raw per-arm grounded decision accuracy and Brier score;
4. retain exact paired contrasts;
5. retain batch digest and derived seed provenance from batch metadata;
6. never update model parameters.

The generator gains an optional metadata `scope` argument. Existing development calls retain the existing default scope so all current development digests remain unchanged; confirmatory-open execution supplies `synthetic-exp282-partial-observability-confirmatory-open`.

## Output artifact

Schema: `NLM-EXP-282-CONFIRMATORY-OPEN-RAW-V1`.

Mandatory state:

- `evidence_level = EV-E2`
- `decision = UNVERIFIED`
- `status = CONFIRMATORY_OPEN_EXECUTED_UNANALYZED`
- `confirmatory_data_consumed = true`
- `seed_materialization_status = EXECUTED`
- exact reserved replicate IDs and count
- frozen protocol digest
- Reconstruction Court digest
- paired checkpoint SHA-256
- execution-contract digest
- executor code digest
- protocol RNG root value
- exact raw per-replicate outcomes
- exact per-replicate world/batch digest and seed lineage
- artifact self-hash

The raw executor does **not** apply the frozen statistical decision rule. A separate Confirmatory Analysis Court will consume this raw artifact later.

## Fail-closed rules

Execution must stop before the first confirmatory batch if any of the following holds:

- preparation was not `CONFIRMATORY_OPEN_PREPARED` upstream;
- Reconstruction Court is invalid or not `RECONSTRUCTION_AUTHORIZED_NOT_EXECUTED`;
- checkpoint SHA mismatch;
- checkpoint reconstruction-contract mismatch;
- checkpoint contains reserve tensors;
- reserved IDs are missing, reordered, duplicated, extended, or reduced;
- protocol digest/status/experiment contract is not the frozen Stage-A V1 authority;
- requested confirmatory world root differs from frozen protocol root;
- an optimizer or training mode is requested;
- output artifact already exists in the CLI path.

Infrastructure failures before the first confirmatory observation may be classified separately by the future execution ledger. Once the first confirmatory batch is materialized, failures remain scientific/raw outcomes and are not silently rerun away.

## CLI and CI

Add `scripts/run_exp282_confirmatory_open.py` requiring protocol, paired execution artifact, reconstruction authorization, checkpoint, and output path. The CLI refuses to overwrite an existing output.

CI uses a synthetic **prepared fixture/rehearsal authorization** to exercise the executor mechanically; CI output remains EV-E2 and cannot be treated as a frozen confirmatory result. Production confirmatory-open execution is never triggered automatically by CI.

## Evidence boundary

This subsystem proves execution integrity, not hypothesis success. Even a favorable raw confirmatory-open artifact remains `EV-E2 / UNVERIFIED` until the frozen analysis is applied and all protected endpoints are checked. Post-freeze challenge and independent replication remain required afterward.
