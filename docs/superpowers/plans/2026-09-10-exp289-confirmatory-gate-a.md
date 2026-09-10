# EXP-289 Confirmatory Gate-A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a fail-closed EXP-289 Stage-A confirmatory path from DEVELOPMENT RDER evidence through immutable pre-beacon authority, while preserving the frozen protocol and EV-E2/UNVERIFIED boundary until a valid real Gate B.

**Architecture:** Keep EXP-289-specific statistics separate from EXP-286 log-cost machinery. Gate A derives paired relative RDER reductions only from reconstruction-valid DEVELOPMENT rows, freezes sample size before challenge randomness, and refuses zero/invalid baseline denominators. Later authority layers bind the exact trained state, code tree, analysis contract, challenge contract, and post-freeze beacon without allowing CI rehearsal to promote evidence.

**Tech Stack:** Python 3.11/3.13, PyTorch, pytest, GitHub Actions, canonical SHA-256 evidence contracts.

**Spec:** `docs/superpowers/specs/2026-09-10-exp289-confirmatory-gate-a-design.md`

## Global Constraints

- Frozen protocol file and digest must not change: `protocols/stage_a_v1.json` / canonical digest `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`.
- EXP-289 primary endpoint remains `repeat_dead_end_rate`, lower is better.
- MESI remains 0.25 relative reduction.
- Confirmatory paired n remains 32..128; power target remains 0.90; familywise alpha remains 0.05.
- Protected over-prune ceiling remains 0.005; protected solution floor remains `local_nogood >= no_nogood - 0.01`.
- Zero-opportunity episodes remain in raw evidence/safety accounting but are excluded from the primary RDER denominator.
- No epsilon denominator rescue.
- DEVELOPMENT observations are never reclassified as confirmatory observations.
- No real beacon/challenge materialization before immutable Gate-A authority is complete.
- TEST-ONLY ceremony paths remain EV-E2 / UNVERIFIED.

## Evidence ledger

- Task 1: RED `762b6893b2c52e8f4110da260f1428d14a2c1f64`; RDER Gate-A prep implemented; canonical protected endpoint parser fixed without changing 0.005.
- Task 2: transactional pre-beacon CLI + focused workflow implemented; tiny rehearsal remains EV-E2 / UNVERIFIED.
- Task 3: authoritative non-tiny DEVELOPMENT geometry exact-head GREEN at `d0c4fd0a0e8ca4cac5dc328418ab1fc128593446`. Geometry digest `948347a9f06313cc1a86ccc3efc0aed4298a3c6b3cd7b35803e1710778ea3499`; scientific execution digest `017f3bbcba5f45fe82809ebb958cbf98d03d10900e08731ef08d0066df2eaf23`; outer execution digest `44e451e1557e56fe3a30fb2ce3f92b03cef548143b899c79d97830978f928283`. DEVELOPMENT planning n=32: baseline RDER=1.0, local RDER=0.0, relative reduction=1.0, paired SD=0.0, both verified-solution rates=1.0, local over-prune=0.0; Gate A `CONFIRMATORY_GATE_A_PREPARED`, confirmatory_n=32. No confirmatory inference is claimed.
- Task 4: RED `442430eb9522ef3a6071756059a2338778059646` = 3 missing-checkpoint-module failures with 55 other focused tests passing. Production checkpoint/replay court added at `76beb3c977ff70331482b67e61dc6d0cbf1219ba`. Current exact branch head includes documentation authority updates after that code; exact-head focused verification remains required before Task 5.

## Task state

- [x] Task 1 — RDER-specific Gate-A preparation.
- [x] Task 2 — transactional Gate-A CLI + focused CI.
- [x] Task 3 — authoritative DEVELOPMENT geometry + Gate-A evidence.
- [ ] Task 4 — exact trained-state checkpoint court: implementation committed; exact-head focused GREEN still required.
- [ ] Task 5 — execution authorization + immutable pre-beacon seal. Authorization must bind protocol/code/geometry/scientific+outer execution/prep plus checkpoint receipt digest, checkpoint scientific identity digest, checkpoint file SHA-256 and checkpoint execution-contract digest. Seal must transitively bind that authority and contain no beacon/seed/challenge observations.
- [ ] Task 6 — TEST-ONLY post-freeze challenge + Gate-B firewall.
- [ ] Task 7 — exact-head closure and real-ceremony eligibility audit.

## Task 5 TDD acceptance surface (do not implement before Task 4 GREEN)

- Smoke/non-authoritative DEVELOPMENT execution must be rejected even if a prep looks READY.
- Canonical authoritative geometry digest/configuration and scientific+outer execution identities must agree across execution, prep and checkpoint.
- Checkpoint receipt validation must pass and authorization must reject tampering in receipt digest, scientific identity digest, file SHA-256, execution-contract digest, final functional-state identities or development/code lineage.
- Frozen confirmatory_n must be 32..128 and must match prep.
- Challenge-contract digest must be frozen before beacon entropy.
- Authorization status is `AUTHORIZED_NOT_EXECUTED`, EV-E2 / UNVERIFIED, with no seed/beacon/challenge observation fields.
- Immutable Gate-A seal binds freeze commit SHA/timestamp, code-tree digest, all authorization lineage and checkpoint authority; it remains `FROZEN_MACHINERY_READY_FOR_FUTURE_BEACON_ONLY` and pre-beacon.
- Integrity failure is never reinterpreted as scientific KILL/PASS.

## Task 6 TDD acceptance surface

- Future beacon only after immutable Gate-A seal.
- Domain-separated seed derivation; no operator raw-seed override.
- Raw challenge evidence persisted and validated before analysis.
- TEST-ONLY path cannot promote above EV-E2 / UNVERIFIED.
- Real Gate B uses frozen paired relative-RDER bootstrap analysis plus raw reconstructed over-prune and verified-solution guards.
- Relative-reduction lower confidence bound must meet 0.25 for promotion; over-prune >0.005 or solution-floor violation prevents promotion; integrity failures remain `INVALID_RUN`.

## Task 7 closure

- Lock exact PR HEAD and fetch focused EXP-289 CI plus normal core 3.11/3.13/model-smoke.
- Verify frozen Stage-A protocol digest unchanged.
- Audit whole diff from `main@8c0b5add4387cbf244d5df61b2ef1a6ebac79a66`.
- Invoke `superpowers:verification-before-completion` and `superpowers:requesting-code-review` before merge-ready claim.
