# EXP-282 Confirmatory Analysis Court Design

## Purpose

Build the first deterministic statistical court that can transform an already executed EXP-282 confirmatory-open raw artifact into a bounded Stage-A neural evidence decision without changing the frozen Stage-A V1 protocol, without reusing development observations, and without materializing post-freeze challenge randomness.

The court is intentionally narrower than a general experiment framework. It exists only for the frozen EXP-282 question: whether the `explicit_belief` matched neural arm beats `recurrent_hidden` under partial observability by the preregistered absolute MESI while satisfying the Brier and accounted-FLOP protected endpoints.

## Evidence boundary

The highest evidence level this court may emit is `EV-E3`, corresponding to Stage-A small-model NLM-substrate neural evidence. It must never emit `EV-E4` or `EV-E5`. Post-freeze challenge replication and independent/clean-room replication remain open after any EV-E3 decision.

The raw confirmatory artifact itself remains immutable `EV-E2 / UNVERIFIED / CONFIRMATORY_OPEN_EXECUTED_UNANALYZED`. The analysis court creates a new self-hashed result artifact and never rewrites raw evidence.

## Inputs

`build_exp282_confirmatory_analysis(...)` consumes:

1. the canonical frozen Stage-A V1 protocol object and exact canonical SHA-256;
2. the confirmatory prep artifact;
3. the reconstruction authorization artifact;
4. the executed confirmatory-open raw artifact;
5. the exact paired development execution artifact, used only to bind resource-match/checkpoint lineage and reconstruct the analytical compute ledger, never to reuse development outcome values in the confirmatory statistic;
6. an `analysis_code_digest` supplied by the caller.

The caller must verify file-level protocol identity against the canonical frozen Stage-A V1 digest before invoking the court.

## Canonical frozen contract

The court hard-fails on drift from:

- experiment: `EXP-282`;
- primary endpoint: `grounded_decision_accuracy`, direction `higher`;
- MESI: absolute gain `0.03`;
- familywise alpha: `0.05`;
- paired analysis: `true`;
- analysis method: `paired accuracy difference with bootstrap CI plus calibration guard`;
- Brier guard: `explicit_belief <= recurrent_hidden + 0.02`;
- compute guard: relative accounted-FLOP difference `<= 0.05` unless cost-normalized, which this V1 court does not do.

## Lineage court

Before statistics, the court validates all supplied artifacts using their existing validators and then requires exact lineage closure:

- protocol digest is canonical V1 and matches prep, paired execution, reconstruction, and raw lineage;
- prep digest matches the analysis input;
- paired execution artifact digest matches prep/reconstruction/raw lineage;
- paired checkpoint SHA matches prep/reconstruction/raw lineage;
- reconstruction digest matches raw lineage;
- execution-contract digest matches reconstruction and raw lineage;
- confirmatory `n` and reserved replicate IDs are identical across prep, reconstruction, and raw;
- raw replicate order is exactly the frozen reserved order;
- raw confirmatory data is consumed, while challenge randomness remains unmaterialized;
- analysis code digest is non-empty and is recorded in the result artifact.

Any malformed or tampered input raises before a result artifact is produced. `INVALID_RUN` is reserved for a well-formed result artifact explicitly generated from an infrastructure/scientific invalidity state in a future extension; V1 does not launder validation exceptions into a statistical decision.

## Statistical method

### Primary paired effect

For each frozen confirmatory replicate `i`:

`d_i = explicit_belief_accuracy_i - recurrent_hidden_accuracy_i`

The observed effect is the arithmetic mean of the paired vector.

### Frozen deterministic bootstrap

V1 uses exactly `10_000` paired nonparametric bootstrap resamples of the replicate-level effect vector.

The bootstrap PRNG seed is derived from pre-result identity material only:

`SHA256(protocol_digest | frozen_analysis_digest | EXP-282 | paired-bootstrap-v1)`

where `frozen_analysis_digest = canonical_sha256(prep["frozen_analysis"])` and therefore existed before confirmatory observations were consumed. No data-derived seed selection is permitted.

The court records:

- observed mean paired effect;
- one-sided 95% lower bound = empirical 5th percentile of bootstrap means;
- one-sided 95% upper bound = empirical 95th percentile of bootstrap means;
- median paired effect;
- positive/zero/negative sign counts;
- exact bootstrap sample count and seed digest.

The percentile implementation is deterministic and index-based; no external numerical dependency is added.

## Protected endpoints

### Brier calibration guard

The court aggregates raw sufficient statistics rather than averaging replicate Brier means with unequal denominators:

- recurrent Brier = total recurrent `brier_sum` / total recurrent `brier_count`;
- explicit Brier = total explicit `brier_sum` / total explicit `brier_count`;
- delta = explicit - recurrent.

Pass iff delta `<= 0.02`.

### Compute guard

The court reconstructs both matched arms from the frozen execution-contract arm geometry and calls the existing `account_matched_belief_arm_pair(...)` using the exact world `timesteps` and `variables` geometry. This is the repository's analytical scalar-arithmetic accounting boundary, not a hardware-profiler FLOP claim.

Pass iff:

- primitive-operation match is true;
- accounted-FLOP match is true or relative accounted-FLOP difference `<= 0.05`;
- the reported ledger still declares `hardware_profiler_flops_claimed = false`.

The reconstructed ledger is embedded in the result artifact so the compute guard is auditable without changing raw confirmatory observations.

## Decision logic

Decision order is deterministic:

1. If the primary one-sided lower bound is `>= 0.03` and both protected guards pass: `EV-E3 / PROMOTE_TO_NEXT_STAGE`.
2. Else if the primary one-sided upper bound is `< 0.03`: `EV-E3 / KILL_SUBSYSTEM` for the explicit-belief EXP-282 subsystem hypothesis at this Stage-A scope.
3. Else: `EV-E3 / HOLD_UNSTABLE`.

A protected-endpoint failure can block promotion but does not by itself create a new kill threshold that the frozen protocol did not specify; it therefore resolves to `HOLD_UNSTABLE` unless rule 2 independently establishes failure to clear the MESI.

`PRACTICALLY_EQUIVALENT_USE_SIMPLER_RIVAL` is not emitted by V1 because the frozen EXP-282 protocol does not provide a separate equivalence margin from which a valid two-sided practical-equivalence test can be constructed. V1 must not invent one post hoc.

## Result artifact

Schema: `NLM-EXP-282-CONFIRMATORY-ANALYSIS-V1`.

Required fields include:

- `schema`, `claim_id`, `experiment_id`;
- `evidence_level`, `decision`, `status`;
- `confirmatory_n`, ordered replicate IDs;
- primary-effect summary and bootstrap contract;
- Brier protected-endpoint summary;
- compute protected-endpoint summary and full analytical ledger;
- complete lineage digests;
- `raw_per_replicate_metrics` copied exactly from the raw artifact so EV-E3 evidence remains reconstructible;
- remaining blockers for EV-E4/EV-E5;
- self-hash `analysis_digest` over canonical JSON excluding the digest field itself.

The validator rejects EV-E4/EV-E5 claims, any decision inconsistent with the frozen decision rules, missing raw replicate evidence, digest tampering, lineage drift, non-finite metrics, and protected-endpoint recomputation mismatches.

## CLI

Add `scripts/analyze_exp282_confirmatory_open.py` with explicit paths for protocol, prep, paired execution, reconstruction, raw input, and output.

The CLI must:

- refuse to overwrite an existing output;
- validate the protocol schema and byte SHA;
- enforce the canonical frozen Stage-A V1 SHA via the shared identity helper;
- validate all input artifacts before analysis;
- write exactly one immutable JSON result artifact;
- print only a compact summary of schema, evidence level, decision, effect bounds, guard states, and result digest.

## Tests

TDD must cover at least:

- RED import/build failure before the analysis module exists;
- deterministic bootstrap reproducibility and pre-result seed derivation;
- correct promote, hold, and kill decision boundaries using synthetic raw metrics;
- Brier guard aggregation from sufficient statistics;
- compute ledger reconstruction and <=5% guard;
- no EV-E4 promotion;
- tampered raw/prep/reconstruction/paired lineage rejection;
- canonical protocol digest enforcement in direct API and CLI;
- raw per-replicate metrics preserved exactly;
- self-hash tamper rejection;
- CLI no-overwrite behavior;
- full existing model-smoke regression suite and compileall.

## Non-goals

- No actual confirmatory-open observations are executed in this PR.
- No post-freeze challenge beacon is derived or materialized.
- No 100M integrated claim is promoted.
- No hardware-profiler FLOP claim is introduced.
- No frozen protocol file is modified.
