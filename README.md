# Nolane-AIv2 — NLM V0.16.1 100M Research Substrate

Nolane-AIv2 is the executable research repository for the **Nolane Living Model (NLM) V0.16.1** program. The project is intentionally built as a falsification lab: mechanisms earn complexity by surviving frozen Stage-A gates; structural correctness is never relabeled as neural capability evidence.

## Current authoritative boundary

- Candidate footprint: **exactly 100,000,000 parameters**.
- Frozen support allocation: **10,000,000**; trainable allocation: **90,000,000**.
- Canonical experiment namespace: `EXP-###`.
- Evidence maturity namespace: `EV-*`.
- Confirmatory Stage-A protocol remains frozen at `protocols/stage_a_v1.json`.
- Neural architecture claims remain **UNVERIFIED** until matched neural runs produce EV-E3 evidence.

## What is executable now

The repository now has two separate reasoning lanes.

### 1. Algorithmic Stage-A falsification lane

A dependency-light structured substrate implements:

- Canonical Problem State with finite-domain table constraints;
- generalized arc-consistency propagation;
- chronological branch search;
- propagation + branch hybrid search;
- oracle conflict-variable prioritization;
- episode-local nogood storage;
- explicit binary belief-state updates and a recurrent evidence baseline;
- bounded exact semantic-equivalence checking for compile-valid fidelity traps;
- paired open-seed proxy runners for `EXP-277`, `EXP-279`, `EXP-282`, `EXP-286`, `EXP-289`, and `EXP-297`, with every smoke arm separated from its `protocol_arm_id`;
- deterministic paired-effect/bootstrap summaries over an explicitly labeled abstract operation proxy;
- EV-E2 Evidence Packet generation locked to the frozen protocol digest and source-tree digest.

This lane measures whether cheap algorithmic proxies expose mechanism headroom. Every raw arm is marked `EV-E2_ALGORITHMIC_PROXY`; it is not a claim that the frozen protocol arm (for example full V0.15 ARCS) has been implemented. It cannot promote a neural claim by itself.

### 2. Exact-100M neural candidate lane

The 100M PyTorch candidate no longer routes every region through the same generic residual block. Four Stage-A-critical regions now expose specialized functional computation while preserving the exact V0.16 budget:

- `recurrent_deliberation_core` → recurrent GRU deliberation;
- `constraint_belief_fabric` → differentiable constraint↔variable message passing + belief logits;
- `conflict_core_backjump_clause` → learned conflict attribution scores;
- `problem_compiler_fidelity_court` → learned semantic-pair fidelity score.

The remaining unused budget is still represented as explicit `capacity_reserve`. This is deliberate: reserve is capacity, not evidence that a mechanism exists.

A model audit reports functional vs reserved parameters per region so architecture growth cannot hide behind the headline 100M count.

## Run the Stage-A engineering smoke lane

```bash
python -m pip install -e '.[dev]'
python scripts/verify_protocol.py
python scripts/run_stage_a.py --replicates 4 --output /tmp/stage-a-smoke.json
```

Run all 32 frozen open replicate indices:

```bash
python scripts/run_stage_a.py --full-open --output /tmp/stage-a-open-32.json
```

The runner always emits `EV-E2 / UNVERIFIED`. Its cost counters are explicitly labeled `ABSTRACT_OPERATIONS_NOT_HARDWARE_FLOPS`, so smoke effects cannot be interpreted as satisfying the confirmatory FLOP-based MESI. It is intentionally incapable of declaring the neural thesis verified.

## Audit the exact 100M candidate

```bash
python -m pip install -e '.[dev,model]'
python scripts/audit_model.py
```

The audit instantiates the authoritative candidate on PyTorch's `meta` device, so exact parameter accounting can be checked without allocating 100M real parameter values.

## Tests

```bash
pytest -q
python -m compileall -q src scripts
```

CI has a dependency-light core lane plus a separate PyTorch model lane. The model lane verifies exact 100M accounting, tiny forward execution, structured reasoning shapes, Stage-A loss backpropagation and reserve isolation.

## Evidence discipline

- A protocol hash establishes committed bytes, not scientific truth.
- Open-smoke results are development/process evidence, not EV-E3 neural evidence.
- Architecture-caused divergence remains a scientific outcome.
- Practical equivalence selects the simpler rival.
- Post-freeze challenge randomness remains unavailable until code/config/evaluator/analysis freeze.
- A negative gate is allowed to delete a subsystem.
- `capacity_reserve` must not be counted as implemented capability.

## Stage-A neural pilot (16M wind tunnel)

The next execution layer is an exact **16,000,000 parameter** neural pilot for the first six Stage-A gates. It deliberately keeps only problem binding, recurrent deliberation, constraint-belief reasoning, conflict scoring, semantic fidelity, verifier support and explicit reserve capacity.

Audit the real 16M parameter topology without allocating storage:

```bash
python scripts/audit_stage_a_pilot.py
```

Run a tiny CPU training smoke that exercises the same optimizer/curriculum/checkpoint contracts:

```bash
python scripts/train_stage_a_pilot.py \
  --tiny \
  --steps 2 \
  --batch-size 4 \
  --variables 4 \
  --constraints 3 \
  --output-dir /tmp/nlm-stage-a-pilot
```

The trainer excludes every `capacity_reserve` tensor from optimizer groups. Checkpoint manifests bind protocol, source tree, model config, curriculum, RNG lineage and tensor-file SHA-256. Training-smoke checkpoints are structurally capped at `EV-E2 / UNVERIFIED`; they cannot self-promote to neural evidence.

The default non-`--tiny` runner instantiates the exact 16M pilot on CPU. This is an execution substrate, not a capability claim or a substitute for frozen EV-E3 experiments.

## EXP-277 matched neural DEVELOPMENT lane

Package `0.11.0` adds an executable matched-neural development lane for the frozen `EXP-277` oracle-structure headroom gate. It intentionally remains **EV-E2 / UNVERIFIED**: the lane measures development behavior and closes execution provenance, but it does not consume confirmatory-open data or execute the frozen scientific promotion rule.

Run the CPU-safe development smoke:

```bash
python scripts/run_exp277_paired_dev.py \
  --tiny \
  --train-replicates 2 \
  --eval-replicates 2 \
  --batch-size 2 \
  --timesteps 3 \
  --variables 4 \
  --constraints 2 \
  --output /tmp/nlm-exp277-paired.json \
  --registry-output /tmp/nlm-exp277-registry.json
```

The two arms share functional initialization and exact parameter count. `arcs_branch` receives only the surface event/variable representation; `oracle_cbrf` additionally receives the ground-truth constraint↔variable incidence artifact. Both are charged by an explicit analytical accounted-FLOP ledger under one declared per-episode ceiling and run on the exact same paired world lineage. The CLI requires the canonical frozen Stage-A protocol digest and refuses to overwrite either output before any experiment execution.

The resulting registry state is `PAIRED_STRUCTURE_DENSE_DEV_READY` with `match_court=BLOCKED`. Remaining confirmatory sample-size/analysis freeze, confirmatory-open execution, and post-freeze challenge evidence are deliberately not manufactured by this development lane.
