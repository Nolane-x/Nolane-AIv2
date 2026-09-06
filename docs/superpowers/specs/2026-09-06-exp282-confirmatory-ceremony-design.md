# EXP-282 Two-Phase Confirmatory Ceremony Design

## Purpose

Provide one fail-closed ceremony boundary between frozen EXP-282 preparation and confirmatory-open observation consumption. The ceremony separates an immutable **SEAL** phase from an **EXECUTE** phase so no confirmatory observation can be materialized before protocol, code, checkpoint, geometry, sample size, reserved replicate IDs, and analysis contract are bound together.

## Scientific boundary

- Frozen Stage-A V1 protocol bytes remain unchanged.
- Canonical protocol SHA-256 remains `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`.
- SEAL is `EV-E2 / UNVERIFIED` and must record `confirmatory_data_consumed=false`, `seed_materialization_status=NOT_EXECUTED`, and `challenge_materialized=false`.
- EXECUTE may consume only confirmatory-open reserved IDs. It must never materialize challenge randomness.
- Raw confirmatory evidence remains immutable `EV-E2 / UNVERIFIED`.
- Analysis may emit at most Stage-A small-model `EV-E3`. EV-E4/EV-E5 remain separate future ceremonies.
- PR/CI tests use synthetic fixtures only. This PR does not execute real confirmatory-open observations.

## Architecture

### 1. Ceremony Seal

`seal_exp282_confirmatory_ceremony(...)` validates the existing paired development artifact, confirmatory prep, execution authorization, and reconstruction authorization. It requires all lineage digests to agree, requires prep status `CONFIRMATORY_OPEN_PREPARED`, and requires a single `ceremony_code_digest` to equal the prep-frozen analysis digest and the execution authorization code digest.

The seal emits `NLM-EXP-282-CONFIRMATORY-CEREMONY-SEAL-V1` containing:

- canonical protocol digest;
- paired development artifact digest;
- prep digest;
- execution authorization digest;
- reconstruction digest;
- checkpoint SHA-256;
- execution-contract digest;
- frozen-analysis digest;
- sample-size-freeze digest;
- exact `confirmatory_n`;
- exact ordered reserved replicate IDs;
- ceremony/source-tree digest;
- `confirmatory_data_consumed=false`;
- `seed_materialization_status=NOT_EXECUTED`;
- `challenge_materialized=false`;
- self-hash `ceremony_seal_digest`.

A validator must recompute the self-hash and cross-check all semantic fields. Rehashing a tampered seal must not bypass lineage validation.

### 2. Ceremony Execution

`execute_exp282_confirmatory_ceremony(...)` accepts a valid seal plus the exact upstream artifacts, protocol, checkpoint path, and current source-tree digest. Before calling the existing sealed executor it verifies:

1. canonical protocol identity;
2. seal semantic validity;
3. current source-tree digest equals the seal's ceremony digest;
4. upstream artifact digests equal the seal;
5. checkpoint file SHA equals the sealed checkpoint SHA;
6. reserved IDs and confirmatory n remain identical;
7. challenge materialization remains forbidden.

Only after all preflight checks pass may it call `execute_exp282_confirmatory_open(...)`. It then passes the resulting raw artifact to `build_exp282_confirmatory_analysis(...)` using the same frozen source-tree digest.

The ceremony returns `NLM-EXP-282-CONFIRMATORY-CEREMONY-RESULT-V1`, an append-only bundle containing the seal, raw artifact, analysis artifact, their digests, and ceremony status. It does not rewrite either scientific artifact.

### 3. Failure semantics

- Invalid seal or lineage mismatch: abort before confirmatory inference.
- Checkpoint mismatch: abort before confirmatory inference.
- Source-tree drift after seal: abort before confirmatory inference.
- Partial execution exception: no ceremony result artifact is considered valid.
- Scientific `HOLD_UNSTABLE`, `KILL_SUBSYSTEM`, or `PROMOTE_TO_NEXT_STAGE` is carried from the Analysis Court and is distinct from infrastructure failure.
- Challenge randomness is never created by either phase.

## CLI surfaces

Two separate commands preserve the physical phase boundary:

- `scripts/seal_exp282_confirmatory_ceremony.py`
- `scripts/execute_exp282_confirmatory_ceremony.py`

Both use no-overwrite outputs. The execute CLI requires an existing seal file and refuses to run when current `source_tree_digest` differs from the seal.

## One-shot semantics

The implementation guarantees one-shot behavior within a declared artifact location through no-overwrite outputs and an immutable seal identity. It does not claim a globally replay-proof distributed lock. A future protected workflow/artifact service may add cross-host replay prevention without changing the scientific artifact schemas.

## Testing

TDD must cover:

- RED import failures before production modules exist;
- valid SEAL with no data consumption;
- not-ready prep rejected;
- seal tamper + rehash rejected;
- source-tree drift rejected before executor invocation;
- checkpoint SHA drift rejected before executor invocation;
- reserved-ID drift rejected;
- challenge flag cannot be enabled;
- synthetic EXECUTE produces raw EV-E2 and analysis EV-E3 bundle;
- analysis HOLD/KILL/PROMOTE passes through unchanged;
- CLI no-overwrite and protocol forgery rejection;
- existing neural/model-smoke suite remains green.

## Out of scope

- Real confirmatory-open execution in CI or during this PR.
- Post-freeze challenge beacon materialization.
- EV-E4 or EV-E5 claims.
- Global distributed replay locks.
- A GitHub Actions workflow that transports real checkpoints or sealed bundles. Secure artifact transport must be designed separately before such a workflow can be called scientifically sealed.
