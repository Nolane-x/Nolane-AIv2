# Stage-A Neural Pilot 16M Implementation Plan

> **For agentic workers:** execute test-first and keep the frozen Stage-A v1 protocol unchanged.

**Goal:** Build an exact-16M trainable Stage-A neural pilot with deterministic curriculum, reserve-safe optimization, and provenance-bound checkpoints.

**Architecture:** Reuse the Stage-A neural region implementations but allocate only the mechanisms required by the six frozen first gates. Keep synthetic curriculum/training outputs explicitly EV-E2 until a separate confirmatory neural protocol execution is frozen and run.

**Tech Stack:** Python 3.11+, PyTorch 2.x, stdlib dataclasses/hashlib/json, pytest.

**Spec:** `docs/superpowers/specs/2026-09-06-stage-a-neural-pilot-design.md`

## Global Constraints
- Exact pilot footprint: 16,000,000 parameters.
- No mutation of frozen Stage-A v1 protocol bytes.
- Capacity reserve is excluded from optimizer parameter groups.
- Training smoke cannot produce EV-E3.
- Every checkpoint manifest binds protocol/config/curriculum/code/seed lineage.

### Task 1: Pilot budget/config
- [ ] Add failing tests for exact 16M total, explicit region set and no hidden authoritative-100M mutation.
- [ ] Implement pilot budget/config constructors.
- [ ] Verify exact meta parameter count.

### Task 2: Functional optimizer boundary
- [ ] Add failing test that reserve parameters receive no gradient/update.
- [ ] Implement functional parameter selectors and optimizer factory.
- [ ] Verify a real tiny CPU optimizer step.

### Task 3: Deterministic curriculum
- [ ] Add failing tests for same-seed determinism, different-seed divergence and tensor contracts.
- [ ] Implement Stage-A synthetic curriculum with digestable metadata.
- [ ] Verify batches are compatible with Stage-A multitask loss.

### Task 4: Trainer + checkpoint manifest
- [ ] Add failing tests for bounded training, loss telemetry and provenance completeness.
- [ ] Implement trainer and canonical checkpoint manifest.
- [ ] Reject incomplete evidence/provenance promotion.

### Task 5: CLI + CI
- [ ] Add CPU tiny-pilot runner and exact-16M audit command.
- [ ] Add CI smoke lane.
- [ ] Run full test suite, compileall and protocol digest verification.
