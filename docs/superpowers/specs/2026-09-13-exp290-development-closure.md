# EXP-290 V1 DEVELOPMENT Closure Record

## Disposition

EXP-290 V1 is closed as a valid negative DEVELOPMENT / EV-E2 court.

Final cross decision: `STRUCTURAL_CLAUSE_TRANSFER_NOT_ESTABLISHED`.

Authorization remains closed:

- `successor_design_authorized = false`
- `authorization_scope = NONE`
- `mechanism_successor_authorized = false`

EXP-291 therefore remains `BLOCKED_ON_PARENT` under the frozen EXP-290 dependency path.

This result does not erase EXP-289's positive scoped episode-local nogood result and does not establish that structural clause transfer is impossible under every representation or objective. It closes only the frozen EXP-290 V1 hypothesis.

## Frozen identity

- base authority: `main@e7a034365e542895b772701a247f6ad9b931bfc9`
- exact pre-data head: `2492f079dc842fe846087d73cc5d8c5dbcca502d`
- release marker head: `144406ecd27bc142c0cde3e5073604432a8f1fb6`
- Actions PR merge SHA: `4f53bd7b702667497d65fba60cc804eb9d483666`
- authoritative DEVELOPMENT run: `34743509423`
- authoritative run attempt: `1`
- Stage-A protocol digest: `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`
- EXP-290 geometry digest: `bffb1f01d54a6867f3af6acd0c2339a83add69f6df4bc6fd8d5504aa9eb4d67a`
- source-tree code digest: `6c8402f16ec1b94d763e1b79776d852b2fff31aa4701e03ff0074dd27f9f3667`

The release commit changed exactly one file, `protocols/exp290-structural-clause-transfer-release.lock`, and bound `pre_marker_head=2492f079dc842fe846087d73cc5d8c5dbcca502d`.

## Pre-data verification

Before the release marker existed, exact pre-data head `2492f079dc842fe846087d73cc5d8c5dbcca502d` passed:

- dedicated EXP-290 CI run `34743266974` (#26) — SUCCESS
  - focused EXP-290 contract: `40 passed`
  - frozen Stage-A digest verification — SUCCESS
  - source/scripts compile — SUCCESS
- generic CI run `34743266975` (#856) — SUCCESS
  - core 3.11 — SUCCESS
  - core 3.13 — SUCCESS
  - model-smoke — SUCCESS
- freeze guard run `34743266977` (#7) — SUCCESS
  - `authoritative_development_run_id = null`
  - `observed_development_run_ids = []`
  - `duplicate_development_run_ids = []`
- release marker absent at the exact pre-data head.

The final pre-data integrity pass closed evidence-publication gaps before any held-out execution. Production receipts now publish `evaluation_episode_count`, `heldout_nonidentity_episode_count`, and the frozen top-level non-claim flags required by the validator. The validator recomputes the evidence chain from per-episode primitives through mode aggregates and root metrics to the frozen decision.

## One-shot execution

Release marker head `144406ecd27bc142c0cde3e5073604432a8f1fb6` produced exactly one authoritative DEVELOPMENT run, `34743509423`, attempt 1.

Its preflight verified:

- this was the first authoritative EXP-290 DEVELOPMENT run for PR #67;
- observed DEVELOPMENT run IDs were exactly `[34743509423]`;
- marker-only release at the exact PR head;
- marker content bound the exact pre-marker head and frozen geometry;
- Stage-A V1 remained frozen;
- EXP-290 geometry identity remained frozen.

All four canonical root jobs completed successfully, validated their canonical receipts and sidecars, and uploaded their artifacts. The cross job then downloaded exactly four root receipts, recomputed the frozen cross-root court, validated the sealed cross receipt, and uploaded the final cross artifact.

Freeze guard run `34743509466` (#8) independently recorded:

- authoritative run `34743509423`
- observed run IDs `[34743509423]`
- duplicate run IDs `[]`

No rerun, replacement root, sample escalation, threshold change, geometry change, or post-result policy change occurred.

## Four-root metrics

Every root reproduced positive fresh oracle source-to-target transfer headroom. Therefore this is not an oracle-replication failure. Every root nevertheless failed the frozen learned-transfer success criteria.

| root | control dead-end rate | learned dead-end rate | oracle dead-end rate | oracle headroom | learned capture | learned over-prune | learned solution rate | root decision |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.685546875 | 0.59765625 | 0.0 | 0.685546875 | 0.1282051282 | 0.8106235566 | 0.4609375 | `LEARNED_STRUCTURAL_TRANSFER_NOT_ESTABLISHED` |
| 1 | 0.6875 | 0.5625 | 0.0 | 0.6875 | 0.1818181818 | 0.6972789116 | 0.36328125 | `LEARNED_STRUCTURAL_TRANSFER_NOT_ESTABLISHED` |
| 2 | 0.673828125 | 0.58203125 | 0.0 | 0.673828125 | 0.1362318841 | 0.8032407407 | 0.46875 | `LEARNED_STRUCTURAL_TRANSFER_NOT_ESTABLISHED` |
| 3 | 0.681640625 | 0.556640625 | 0.0 | 0.681640625 | 0.1833810888 | 0.6838565022 | 0.5234375 | `LEARNED_STRUCTURAL_TRANSFER_NOT_ESTABLISHED` |

The frozen root success gate required all of the following simultaneously: positive oracle and learned headroom, `capture >= 0.50`, learned valid-state over-prune `<= 0.005`, learned verified-solution rate at least control minus `0.01`, positive transferred-prune count, non-identity held-out permutations, and no evaluator/oracle mapping leakage.

The learned mechanism recovered only about `12.8%..18.3%` of fresh oracle transfer value and violated the inherited safety floors by a very large margin. Control verified-solution rate was `1.0` on every root while learned verified-solution rate was only `0.36328125..0.5234375`. Held-out non-identity permutation rate remained exactly `1.0` on every root, and learned mapping/oracle leakage flags remained false.

The cross reducer therefore sealed `STRUCTURAL_CLAUSE_TRANSFER_NOT_ESTABLISHED`: zero roots established learned structural transfer while all four reproduced positive oracle headroom.

## Artifact ledger

Cross artifact:

- artifact ID: `10312779053`
- name: `exp290-cross-4f53bd7b702667497d65fba60cc804eb9d483666`
- ZIP SHA-256: `2034b38e3aa6b29e161e2dc465cdf23f3526506a1acff1405dfdb14373d50714`
- receipt SHA-256: `7896b0c5474b35b8930929069997df218858e0e76431fe081c8c1bcc7d3305eb`
- canonical artifact digest: `443161427b960f3f108fec289b71b0db065d243f33a8ac3b86ccf7083d103049`

Root artifacts:

| root | artifact ID | ZIP SHA-256 | receipt SHA-256 | canonical artifact digest |
|---:|---:|---|---|---|
| 0 | `10312883768` | `e357e6ddd07118ec038a2a16e278689c8b21e0b1df57331b1b94a3bc4f98e217` | `134c57c1134537b02d302c6ec8c1b31599ceeb2984ff68343fc0d4f3efefff60` | `5c00a90b793bcf443716b92c2f1bd25121c7bf97d554a22c6df11e00c8cccf2b` |
| 1 | `10312933898` | `dbe9d57d050bf4d9854177246694973e9c0c1b2220bf45154e83f59ba19f9d8f` | `3d7ad7c4f6de7ae39d7140a18068520a0ca06d879dcfbe17cd0f3518bf25b538` | `0f28f6090c660272e2fb153eb38f81730ab6412a77fd445ccd66800fb1615989` |
| 2 | `10313058459` | `826b4c8d8119a79844060d0061267866860a9973a6a2fcf9e2eab87ce6f767d7` | `04cb3a051ac10df5a99ef4f98b242a530e44ffe5a8fe6e2de97c1d9719f1aa07` | `10499c8ab3536e694638299ad41f22cf045e17d95f791743d11308612c0de1a7` |
| 3 | `10312918740` | `e26d41623736197d87dfe9abf1ff708902c3da0b38dffc7a81628377fb73f48e` | `da296af769cc0441daf06b66a43ecd7ce9ad460cf5c8bf1672dc6c5661b98597` | `40d9f7d7bffb891205ac9625777dc4e38077bfd9c2733c08f96fecc12364042a` |

Independent audit downloaded all five artifacts and verified receipt sidecars, canonical serialize/parse/serialize stability, canonical artifact-digest reconstruction, root-to-cross linkage, exact repository/code/protocol/geometry identity, root decisions, cross decision, model-state equality across evaluation, and evidence-boundary flags.

## Evidence boundary

The sealed root/cross receipts preserved:

- `evidence_level = EV-E2`
- `scientific_evidence_eligible = false`
- `confirmatory_data_consumed = false`
- `challenge_materialized = false`
- `promotion_claimed = false`
- `stage_a_protocol_modified = false`
- `evaluation_mapping_used_for_training = false`
- `evaluation_mapping_used_by_learned_mode = false`
- `oracle_mapping_used_by_learned_mode = false`
- `oracle_transfer_mode_deployable = false`
- `arbitrary_value_symbol_remapping_claimed = false`
- `cross_domain_transfer_claimed = false`
- `lemma_generation_claimed = false`

## Post-release CI note

The marker-head dedicated and generic core suites contain a prerelease-only sentinel requiring the release marker to be absent. Once the marker existed, that sentinel failed by construction. Dedicated EXP-290 CI #27 reported `39 passed, 1 failed`, with only `test_release_marker_is_absent_before_predata_green` failing. Generic core 3.13 likewise reported `180 passed, 1 failed` (plus expected no-model skips) with the same sentinel as the only failure. Model-smoke on the marker head remained successful.

V1 was intentionally not modified after held-out result visibility merely to make a prerelease sentinel green.

## Repository closure

PR #67 was closed without merge at release head `144406ecd27bc142c0cde3e5073604432a8f1fb6`. The experiment branch remains preserved for audit provenance. `main` remained `e7a034365e542895b772701a247f6ad9b931bfc9` at experiment closure.

The reusable frozen design, implementation record, this closure record, and the evidence-aware V0.16.1 program matrix may be integrated separately as documentation-only history. EXP-290 runtime code, workflows, tests, geometry files, and release marker from the negative court are not promoted into `main` by that documentation closure.

## Program consequence

EXP-291 is not authorized by EXP-290 V1 and remains blocked on this dependency path. EXP-292 through EXP-296 consequently remain blocked on their upstream counterexample/lemma chain unless a scientifically distinct replacement hypothesis is preregistered.

The highest-priority currently authorized surviving court is EXP-298, independently authorized by EXP-297's scoped EV-E3 promotion. EXP-298 must preserve EXP-297's semantic-authority limits and is not a continuation of the failed EXP-290 mechanism.
