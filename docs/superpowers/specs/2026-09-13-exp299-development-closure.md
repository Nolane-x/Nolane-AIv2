# EXP-299 Native Fidelity Scaffold-Removal — DEVELOPMENT Closure

## Disposition

EXP-299 V1 / `H-NATIVE-01` is closed as a **negative DEVELOPMENT / EV-E2 court** for the frozen scaffold-removal hypothesis.

The sole authoritative four-root cross disposition is:

`NATIVE_FIDELITY_NOT_ESTABLISHED_RECURRENT`

with:

- `established_root_count=0`
- `required_established_root_count=4`
- `successor_design_authorized=false`
- `authorization_scope=NONE`
- `exp300_execution_authorized=false`

This is a scoped negative result. It does not refute every possible representation, objective, teacher signal, or internalization mechanism. It does close EXP-299 V1 exactly as frozen and prohibits same-hypothesis tuning or rerun after held-out visibility.

## Frozen question

EXP-299 asked whether a matched neural model trained with an auxiliary exact behavioral-fidelity court teacher would retain a material cross-domain fidelity/authority advantage after that court was removed completely from held-out causal inference.

The causal contrast was:

1. `binary_supervision_control` — structural pair features plus the primary binary fidelity target only;
2. `court_teacher_then_native` — the same structural pair features, architecture, initialization, primary supervision, examples, batch order, optimizer family, parameter budget, and inference path, plus a fixed fit-time auxiliary court-teacher loss.

Neither arm received a court receipt, witness, evaluator truth, faithful label, or stratum during held-out native inference.

## Pre-data geometry and identities

The experiment was frozen before DEVELOPMENT data execution with:

- canonical roots: `[0,1,2,3]`
- domains: `code_invariant`, `causal_diagnosis`, `grounded_language_ambiguity`
- fit replicates per domain/root: `96`
- held-out evaluation replicates per domain/root: `32`
- candidates per replicate: `16`
- fit candidates per root: `4,608`
- held-out candidates per root: `1,536`
- total held-out candidates across four roots: `6,144`
- structural feature width: `128`
- pair feature width: `512`
- matched neural target parameters: `500,000`
- `d_model=64`
- `hidden_size=64`
- batch size: `32`
- AdamW learning rate: `3e-4`
- weight decay: `1e-4`
- gradient clip norm: `1.0`
- fit passes: `1`
- control auxiliary loss weight: `0.0`
- teacher auxiliary loss weight: `0.25`
- authority threshold: `0.5`
- exact-probe bound: `4096`

Frozen per-domain success gates were conjunctive:

- teacher native balanced accuracy `>= 0.70`;
- teacher-minus-control balanced-accuracy gain `>= 0.10`;
- teacher wrong-authority rate `<= 0.05`;
- teacher faithful-rejection rate `<= 0.10`.

Exact identities:

- Stage-A protocol digest: `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`
- geometry digest: `7210e64dca010c6c1c8ab2cf20c7f03f7d7f263a88f9c8ae162d43151bb9932e`
- structural-encoder contract digest: `843069e7081309842f8e594f1ab8833f3ad23b2b3d49c975351268e10f157a23`
- source-tree digest: `422d0c43ee1cf3d3384c0b2c59cee7ade594ce9477946880ea57e2b89c11b4c8`
- pre-marker frozen head: `564aa26b9e4dc0c9189aaa824dc9999c824c92f1`
- marker-only release head: `2ab16177fd2bda5e53bc5353b9aa9ec01590dd81`

Before release, the exact pre-marker head passed dedicated EXP-299 contracts, structural leakage checks, fresh-process validator portability, generic core Python 3.11 and 3.13, generic model-smoke, frozen Stage-A/geometry checks, compileall, and the zero-run freeze guard.

## One-shot DEVELOPMENT execution

Release marker head `2ab16177fd2bda5e53bc5353b9aa9ec01590dd81` produced exactly one authoritative DEVELOPMENT run:

- workflow run: `34753873059`
- run number: `1`
- attempt: `1`

The preflight proved first-run authority, exact PR-head checkout, marker-only release diff, frozen Stage-A identity, and frozen EXP-299 geometry before any root execution.

All four root jobs completed, validated their receipts and SHA-256 sidecars before upload, and sealed artifacts retained for independent audit.

## Root results

Every canonical root closed `NATIVE_FIDELITY_NOT_ESTABLISHED`.

### Root 0

Artifact digest: `40e9bd0e797c6423c10fd1485902f6a0246fe24ef2102d1dc0e28303b38d3a32`

| Domain | Control BA | Teacher-native BA | Native gain | Teacher wrong authority | Teacher faithful rejection | Pass |
|---|---:|---:|---:|---:|---:|---|
| code invariant | 0.500000 | 0.937500 | +0.437500 | 0.125000 | 0.000000 | no |
| causal diagnosis | 0.875000 | 0.875000 | +0.000000 | 0.250000 | 0.000000 | no |
| grounded language ambiguity | 0.875000 | 0.875000 | +0.000000 | 0.250000 | 0.000000 | no |

Worst-domain native gain: `0.0`.

The code-invariant teacher arm achieved high balanced accuracy and large gain, but violated the frozen wrong-authority ceiling. The other two domains had no native gain and also violated the wrong-authority ceiling.

### Root 1

Artifact digest: `a05b74acf003d61747b1e4ff311baac8a67647381fe215f274126bd6ff75efa1`

| Domain | Control BA | Teacher-native BA | Native gain | Teacher wrong authority | Teacher faithful rejection | Pass |
|---|---:|---:|---:|---:|---:|---|
| code invariant | 0.92578125 | 0.962890625 | +0.037109375 | 0.07421875 | 0.000000 | no |
| causal diagnosis | 1.000000 | 1.000000 | +0.000000 | 0.000000 | 0.000000 | no |
| grounded language ambiguity | 1.000000 | 0.875000 | -0.125000 | 0.250000 | 0.000000 | no |

Worst-domain native gain: `-0.125`.

No domain met the full conjunction. Grounded-language performance regressed relative to the control and violated the wrong-authority ceiling.

### Root 2

Artifact digest: `1329310e468d9e3c80abcb14047e50fc3075b571242791e90ac0313ff4489585`

| Domain | Control BA | Teacher-native BA | Native gain | Teacher wrong authority | Teacher faithful rejection | Pass |
|---|---:|---:|---:|---:|---:|---|
| code invariant | 0.97265625 | 0.61718750 | -0.35546875 | 0.765625 | 0.000000 | no |
| causal diagnosis | 1.000000 | 0.875000 | -0.125000 | 0.250000 | 0.000000 | no |
| grounded language ambiguity | 0.875000 | 1.000000 | +0.125000 | 0.000000 | 0.000000 | yes |

Worst-domain native gain: `-0.35546875`.

This root demonstrates why the frozen decision is conjunctive: grounded-language ambiguity passed all V1 gates, while code invariant and causal diagnosis regressed materially and code invariant had a very high wrong-authority rate.

### Root 3

Artifact digest: `866b4592b8a527f82e0fc0369e9e0c1be7ec2c23d0b4603e51568bcc55478238`

| Domain | Control BA | Teacher-native BA | Native gain | Teacher wrong authority | Teacher faithful rejection | Pass |
|---|---:|---:|---:|---:|---:|---|
| code invariant | 0.500000 | 0.500000 | +0.000000 | 0.000000 | 1.000000 | no |
| causal diagnosis | 0.937500 | 1.000000 | +0.062500 | 0.000000 | 0.000000 | no |
| grounded language ambiguity | 0.937500 | 1.000000 | +0.062500 | 0.000000 | 0.000000 | no |

Worst-domain native gain: `0.0`.

The latter two domains were safe and accurate but missed the frozen `+0.10` gain MESI. Code invariant rejected every faithful candidate and therefore failed the protected faithful-rejection ceiling maximally.

## Cross-root disposition

The cross reducer consumed exactly the four canonical roots above and sealed:

- cross artifact digest: `40a4259f912402e171a01a99adc74093749f756fba049428d00764d5a4210520`
- source root decisions: four × `NATIVE_FIDELITY_NOT_ESTABLISHED`
- established roots: `0/4`
- decision: `NATIVE_FIDELITY_NOT_ESTABLISHED_RECURRENT`
- successor design authorized: `false`
- authorization scope: `NONE`
- EXP-300 execution authorized: `false`

The result does not support the frozen claim that fit-time court-teacher supervision produces a recurrent material native fidelity advantage after scaffold removal across all three controlled domain families.

## Independent artifact audit

After the authoritative run completed, all five GitHub Actions ZIP artifacts (four roots plus cross receipt) were independently downloaded and audited from their bytes rather than inferred from workflow status alone.

For every root, the audit rechecked:

- receipt byte SHA-256 against `receipt.json.sha256`;
- internal `artifact_digest` under canonical JSON with the digest field excluded;
- fit and evaluation partition digests;
- zero fit/evaluation identity overlap;
- `fit.inconclusive_count=0` and `evaluation.inconclusive_count=0`;
- exactly `4,608` fit and `1,536` held-out candidates;
- raw held-out digest;
- every prediction commitment digest;
- prediction committed before evaluator court materialization;
- held-out model causal-input key boundary;
- `teacher_scaffold_consumed=false` on held-out predictions;
- model-state digest unchanged across held-out evaluation;
- TP/TN/FP/FN, balanced accuracy, wrong-authority rate, faithful-rejection rate, native gain, and per-domain pass reconstruction from the raw rows;
- protected evidence-boundary flags all false.

The cross audit rechecked its byte sidecar, internal digest, exact source-root artifact-digest linkage, exact source-root decisions, established-root count, and negative authorization boundary. No audit mismatch was found.

All root and cross execution identities bind the same:

- repository head `2ab16177fd2bda5e53bc5353b9aa9ec01590dd81`;
- source-tree digest `422d0c43ee1cf3d3384c0b2c59cee7ade594ce9477946880ea57e2b89c11b4c8`;
- Stage-A protocol digest `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`;
- geometry digest `7210e64dca010c6c1c8ab2cf20c7f03f7d7f263a88f9c8ae162d43151bb9932e`;
- authoritative workflow run `34753873059`.

## Evidence boundary

EXP-299 remains DEVELOPMENT / EV-E2 only. The sealed cross receipt keeps false:

- `scientific_evidence_eligible`;
- `confirmatory_data_consumed`;
- `challenge_materialized`;
- `promotion_claimed`;
- `unrestricted_semantic_authority_claimed`;
- `open_language_understanding_claimed`;
- `causal_discovery_claimed`;
- `general_code_reasoning_claimed`;
- `durable_lifelong_internalization_claimed`;
- `exp300_execution_authorized`.

No conclusion from EXP-299 should be rewritten as unrestricted semantic understanding, arbitrary code verification, causal discovery, open-language competence, durable lifelong consolidation, or integrated V0.16 superiority.

## Program consequence

Program status for EXP-299 becomes `DEVELOPMENT_CLOSED_NEGATIVE`.

EXP-300 remains `BLOCKED_ON_PARENT`. The original straight-line justification for EXP-300 required model-native gains surviving EXP-299 scaffold removal. EXP-299 V1 did not establish those gains and granted no successor-design authority.

Therefore:

1. no EXP-300 design or execution is authorized by this lineage;
2. EXP-299 V1 may not be rerun, retuned, recalibrated, widened, or repaired after result visibility;
3. a future return to model-native scaffold-free fidelity requires a scientifically distinct hypothesis with a new representation/objective/teacher mechanism and a new preregistration;
4. any future integrated V0.16 comparison needs an independently justified viable native candidate rather than inheriting authority from EXP-299 V1;
5. the positive EXP-298 DEVELOPMENT result remains intact within its own bounded fidelity-fabric scope; EXP-299's negative result shows that the frozen V1 teacher/internalization mechanism did not reliably preserve that advantage once the scaffold was removed.

## Branch closure

Experiment PR #72 was closed unmerged after the authoritative DEVELOPMENT run and independent artifact audit. Its runtime, workflows, tests, release marker, and experimental implementation remain outside `main`.

Only this documentation closure and the corresponding program-authority update are candidates for integration into `main`.
