# EXP-279 real Gate-B ceremony attempt 1 — authoritative pre-inference NOT_READY evidence

This directory preserves the authoritative evidence from GitHub Actions run `34359170130`
for ceremony arm SHA `f0faf3c47674df3933e62fe7f4e6bdd1aa79805f`, whose scientific baseline was
`2d8532371f05798d2d8be5fb19731503192363e0`.

The frozen Stage-A protocol digest is
`c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`.
The scientific source-tree digest recorded by the ceremony is
`157ec83aced2462aa64ae1b26e27ba16302931ddad2114d32998babb50fd8b5f`.

## Authoritative outcome

The preflight result is `NOT_READY_VARIANCE_EXCEEDS_MAX_N` at `EV-E2 / UNVERIFIED`.
The two frozen contrast planning requirements were:

- `hybrid_vs_branch_only`: required paired `n = 264`
- `hybrid_vs_propagation_only`: required paired `n = 286`

The frozen maximum is `n = 128`, so Gate A correctly remained unopened.

No real confirmatory challenge was consumed:

- `confirmatory_data_consumed = false`
- `challenge_materialized = false`
- `inference_started = false`
- no public drand beacon was selected
- Gate B was not executed
- no PROMOTE / HOLD / KILL scientific decision was executed

This is therefore a pre-inference readiness result, not a negative confirmatory result and
not scientific evidence for promotion or killing the subsystem.

## Integrity

`ACTIONS_BUNDLE_SHA256SUMS` reproduces the manifest from the original Actions evidence bundle.
The canonical files were independently re-hashed after download and all five matched their
recorded SHA-256. Because the GitHub connector cannot atomically commit the complete Actions
ZIP, this repository persistence keeps the canonical summary/source/registry, a compact
preflight projection bound to the canonical preflight SHA-256, and exact Actions artifact
IDs/digest in `actions-provenance.json`. The original full bundle remains the authority for
files not reproduced here.

Both the outcome artifact and evidence artifact reported the same GitHub Actions artifact
digest:

`sha256:d498671067616b4e774ba63abc40d5564c45b37e1772ab998f5e9d9bf1e34684`

`actions-provenance.json` records the workflow, CI, artifact and lineage identifiers needed
to resolve the exact ceremony later.
