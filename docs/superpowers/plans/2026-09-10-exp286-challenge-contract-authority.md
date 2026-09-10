# EXP-286 Challenge-Contract Authority Binding Plan

**Goal:** Bind immutable post-freeze challenge semantics into EXP-286 execution authorization before any public beacon can be consumed.

**Authority:** `protocols/stage_a_v1.json` EXP-286 frozen contract and `docs/superpowers/specs/2026-09-07-exp286-oracle-conflict-headroom-design.md`.

**Constraints:** No edits to frozen Stage-A protocol or digest; no beacon fetch; no challenge seed materialization; no confirmatory observations; no EV-E3 upgrade; preserve MESI 0.15, paired n 32..128, power 0.90, lower-is-better accounted reasoning FLOPs.

## Task 1 — Freeze challenge contract in code

- Add `src/nolane_ai/experiments/exp286_challenge_worlds.py` containing only `challenge_contract()` and `challenge_contract_digest()` at this stage.
- Contract must name EXP-286, POST_FREEZE_CHALLENGE, chronological_failure vs oracle_conflict_core, the exact primary endpoint/direction, oracle-label visibility/cost boundary, paired design, and future-beacon seed authority.
- Do not generate worlds yet.

## Task 2 — Bind digest into authorization

- TDD first: add a failing authority test requiring `lineage.challenge_contract_digest` and preflight `challenge_contract_bound`.
- Update `exp286_confirmatory_authorization.py` minimally so authorizations compute/bind the current challenge-contract digest and validator requires it.
- Existing pre-beacon behavior remains unchanged: no seed, beacon, or challenge fields.

## Task 3 — Verify exact head

- Focused EXP-286 workflow must pass all contract/Gate-A/authorization checks.
- Normal CI Python 3.11, Python 3.13, model-smoke, protocol verification must pass.
- EXP-277 regression must remain green.
