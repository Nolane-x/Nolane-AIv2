# EXP-286 Immutable Gate-A Seal Implementation Plan

**Goal:** Close the last pre-beacon lineage gap by sealing the authoritative EXP-286 Gate-A state to an explicit freeze commit SHA/timestamp before any future public beacon may be consumed.

**Architecture:** Add an EXP-286-specific immutable seal over the already-validated DEVELOPMENT execution, Gate-A prep, execution authorization, current code-tree digest, challenge-contract digest, frozen protocol digest, geometry digest, confirmatory n, and freeze commit metadata. The seal remains EV-E2 / UNVERIFIED and contains no beacon, seed, challenge worlds, confirmatory observations, or decision output.

**Authority:** `protocols/stage_a_v1.json` EXP-286 contract; `protocols/exp286_development_geometry_v1.json`; `src/nolane_ai/experiments/exp286_confirmatory_authorization.py`; `src/nolane_ai/experiments/exp286_challenge_worlds.py`.

## Invariants

- Schema: `NLM-EXP-286-CONFIRMATORY-GATE-A-SEAL-V1`.
- Status: `CONFIRMATORY_GATE_A_SEALED`.
- Evidence stays `EV-E2`; decision stays `UNVERIFIED`.
- `confirmatory_ready_scope` is only `FROZEN_MACHINERY_READY_FOR_FUTURE_BEACON_ONLY`.
- `confirmatory_data_consumed=False`, `seed_materialization_status=NOT_EXECUTED`, `challenge_materialized=False`, `decision_rule_executed=False`.
- Freeze SHA must be 40 or 64 hex; freeze timestamp must be timezone-aware UTC.
- Current code-tree digest must be 64 hex and must match DEVELOPMENT execution code digest, prep analysis code digest, and authorization execution-code lineage.
- Challenge-contract digest in authorization must equal the current `challenge_contract_digest()`.
- Frozen protocol and authoritative DEVELOPMENT geometry lineage must remain unchanged.
- Seal must expose canonical `pre_beacon_binding_digest` and `seal_digest`; tampering with freeze/code/contract/n/lineage invalidates it.
- No function in this task fetches entropy or upgrades scientific evidence.

## TDD sequence

1. Add `tests/test_exp286_confirmatory_ceremony.py` first and observe RED because the seal module is absent.
2. Add `src/nolane_ai/experiments/exp286_confirmatory_ceremony.py` with the minimal builder + validator.
3. Verify focused tests GREEN and preserve all existing EXP-286 tests.
4. In a separate TDD cycle, add a CLI/workflow rehearsal using synthetic freeze metadata only; mark/restrict it so CI cannot create scientific evidence.
5. After exact-head focused + normal + EXP-277 regression GREEN, real future beacon work may start only from an externally recorded beacon strictly later than the real freeze metadata.
