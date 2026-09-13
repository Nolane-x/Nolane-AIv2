# EXP-298 Cross-Domain Fidelity Transfer — DEVELOPMENT Closure

## Disposition

EXP-298 V1 is closed as a **positive DEVELOPMENT / EV-E2 court** for the frozen controlled-domain hypothesis. The four authoritative canonical roots all reached `CROSS_DOMAIN_FIDELITY_TRANSFER_ESTABLISHED`. Applying the preregistered four-of-four cross-root rule to those sealed root receipts yields:

- decision: `CROSS_DOMAIN_FIDELITY_TRANSFER_RECURRENT`
- established roots: `4 / 4`
- successor authority: `DESIGN_EXP299_SCAFFOLD_REMOVAL_COURT_ONLY`
- EXP-300 authority: `false`

This is not EV-E3 confirmatory promotion and does not grant unrestricted semantic authority.

## Frozen identity

The court was opened only after the exact pre-data head passed dedicated EXP-298 CI, generic Python 3.11/3.13 core CI, model-smoke, frozen Stage-A verification, compile checks, and a zero-run freeze guard.

- PR: `#70`, closed unmerged
- pre-marker head: `68e1205fd56e5b1acb8f02cea56fc9af1a881734`
- marker-only scientific head: `262e786e5bea59e78a380c9d8a552259b10e7908`
- authoritative DEVELOPMENT run: `34751159358` (run #1)
- Stage-A digest: `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`
- geometry digest: `4ba5a9a569e006d03d4bdf9a596797418c5852f555a8e0ed08314dda43dedc61`
- source-tree digest: `4139f764f037439a375864fb1fe85897663e7b1836bee7e75fc1fe549931749b`

The marker commit changed exactly one file: `protocols/exp298-cross-domain-fidelity-release.lock`.

## Frozen geometry

Each canonical root used the same preregistered geometry:

- roots `[0,1,2,3]`
- domains: bounded code invariants, finite intervention-based causal diagnosis, and controlled grounded-language ambiguity
- 32 evaluation replicates per domain/root
- 16 compile-valid candidates per replicate
- 8 faithful and 8 wrong candidates per replicate
- `d_model=64`
- `hidden_size=64`
- `target_parameters=500000`
- exact semantic probe ceiling `4096`
- per-domain balanced-accuracy gain MESI `+0.10`
- wrong-formalization authority ceiling `0.05`
- faithful-formalization rejection ceiling `0.10`

No threshold or geometry was changed after data.

## Sealed root result

All four root jobs completed successfully, validated their own root receipts before upload, and uploaded sealed GitHub Actions artifacts. Every root contains exactly `1,536` candidate evaluations; each root/domain cell contains `512` candidates, split `256` faithful / `256` wrong.

For all 12 root × domain cells:

| Metric | Compile-only control | Fidelity fabric |
|---|---:|---:|
| semantic-fidelity balanced accuracy | `0.5` | `1.0` |
| wrong-formalization authority rate | `1.0` | `0.0` |
| faithful-formalization rejection rate | `0.0` | `0.0` |

The fidelity arm therefore achieved a balanced-accuracy gain of `+0.5` in every root/domain cell, five times the frozen `+0.10` MESI. The worst-domain gain was `+0.5` on every root. All root model-state before/after digests matched, matched-arm resource checks passed, evaluator truth and trap-family labels remained outside the causal path, and fidelity authority agreed with the public court receipt.

Maximum observed exact probes were `18` for code invariants, `20` for causal diagnosis, and `256` for grounded-language ambiguity, all below the frozen `4096` ceiling.

Root artifact digests are preserved in `evidence/exp298/2026-09-13-development-run-34751159358/artifact-manifest.json`.

## Cross-job failure and recovery classification

The original cross job (`103708027495`) downloaded all four sealed root artifacts successfully, including GitHub ZIP-digest verification, but failed while loading root 0. `validate_exp298_root()` reconstructs the entire root payload by re-running `run_exp298_root()` in a new process and then requires byte-for-byte payload equality. The producing root job had already passed this validator in-process; the later cross process reported `EXP-298 root reconstruction mismatch`.

This is recorded as `CROSS_PROCESS_FULL_PAYLOAD_RECONSTRUCTION_PORTABILITY_FAILURE`. The archive does **not** claim a proven hardware-specific root cause. The important bounded fact is that the failed cross process did not alter any sealed root receipt, and independent post-run verification confirmed each receipt sidecar, each internal artifact digest, each frozen identity, all causal authority decisions, all domain metrics, and the four root decisions.

No model was re-executed during recovery. The forensic reducer simply applies the already-frozen cross rule to the four sealed root decisions. The recovered cross receipt is stored at `evidence/exp298/2026-09-13-development-run-34751159358/cross-recovery.json` and explicitly records that the workflow cross job failed and emitted no cross artifact.

## Duplicate workflow object after release

A later test-only commit on PR #70 caused GitHub's pull-request path filtering to create DEVELOPMENT run #2 (`34751503700`) because the cumulative PR diff still contained the release marker. The first-authority preflight rejected run #2 before checkout, root creation, or model execution. Therefore no second scientific execution occurred; run `34751159358` remains the sole authoritative DEVELOPMENT execution.

PR #70 was then closed unmerged and the experiment branch was sealed. This closure archive is intentionally maintained on a separate branch from `main` so the tested DEVELOPMENT implementation is not silently promoted into the integrated architecture.

## Program authority

EXP-298 V1 establishes recurrent **controlled-domain fidelity-gating transfer** under the frozen synthetic/bounded semantics. It authorizes only the design of EXP-299's scaffold/cache/retrieval-removal court.

It does **not** establish:

- open-language semantic understanding;
- general program verification;
- general causal discovery;
- learned zero-shot semantic representation transfer in unrestricted domains;
- model-native retention after scaffolding removal;
- integrated V0.16 VCPF superiority;
- EXP-300 execution authority;
- unrestricted semantic authority;
- success of EXP-290 or EXP-291..296.

EXP-299 must therefore preregister its own frozen question, arms, endpoints, MESI, resource match, scaffold-removal geometry, failure classification, and decision rule before any implementation or data execution. EXP-300 remains blocked until the integrated candidate is supported by surviving mechanisms and the required VCPF/safety gates.

## Archived evidence

- `artifact-manifest.json` — exact Actions artifact IDs, ZIP digests, receipt byte digests, internal artifact digests, run/head identities, and duplicate-run classification.
- `forensic-audit.json` — independent post-run recomputation of root/domain metrics and trust-boundary checks without model re-execution.
- `cross-recovery.json` — transparent application of the frozen four-of-four reducer rule to the sealed roots, including the original cross-job failure metadata.

The three archive digests are:

- artifact manifest: `eb88ecfbe5b8bad25d1aec79643dd44a9b433c847b01b7480304ce4acf55232d`
- forensic audit: `295911d99875622ca729eab67c0927b70f0f374fb72573c80613272f64334c87`
- recovered cross receipt: `0cc37c9d2e4eb240fff5418d5f3dfe18923117f27c1ff91c25f92e7491da6b37`
