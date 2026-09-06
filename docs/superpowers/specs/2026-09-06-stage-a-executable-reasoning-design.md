# NLM Stage-A Executable Reasoning Design

## Status

Approved continuation of the V0.16.1 next action: turn the six frozen gates into executable falsification machinery without changing their protocol commitments or promoting neural claims.

## Goal

Create an executable structured-reasoning wind tunnel that can expose mechanism headroom cheaply, while simultaneously replacing selected generic 100M residual placeholders with real neural interfaces that can later be trained under the same gates.

## Scientific boundary

The algorithmic/oracle lane is capped at `EV-E2`. It may reveal implementation defects, benchmark pathologies or obvious mechanism headroom, but it cannot promote `H-CBRF-*`, `H-BELIEF-*`, `H-CONFLICT-*` or `H-FIDELITY-*` as neural capabilities. `protocols/stage_a_v1.json` and its SHA-256 remain unchanged.

## Architecture

1. **Canonical Problem State** — finite-domain variables and table constraints with exact solution semantics.
2. **Propagation engine** — generalized arc consistency with explicit operation accounting.
3. **Search engine** — chronological branch search, hybrid propagate-then-branch, oracle conflict-variable priority and episode-local nogoods.
4. **Belief lane** — normalized explicit belief state plus a deliberately simple recurrent evidence baseline for plumbing only; the strong matched learned recurrent rival remains future EV-E3 work.
5. **Fidelity lane** — exact bounded semantic-signature comparison for compile-valid traps.
6. **Six-gate harness** — paired world lineage using the frozen environment RNG stream; raw per-arm metrics are retained.
7. **Analyzer** — deterministic paired effects plus bootstrap intervals. Smoke analysis is descriptive and cannot change the protocol decision state from `UNVERIFIED`.
8. **Evidence runner** — refuses protocol digest drift and binds a source-tree digest into the EV-E2 packet.

## 100M neural integration

The authoritative 100M parameter table is unchanged. Four regions gain specialized functional modules:

- recurrent deliberation via a GRU residual path;
- constraint-belief variable/constraint message passing with belief logits;
- conflict-core scoring head;
- semantic fidelity pair scorer.

Every specialized region is sealed back to its exact regional budget with an explicit `capacity_reserve`. Capacity reserve receives no gradient through Stage-A functional paths.

## Failure handling

- If propagation-only dominates hybrid in the structure-dense smoke lane, the result is retained rather than tuned away.
- If an oracle mechanism shows no headroom, learned complexity should not be added to rescue it.
- An EV-E2 smoke packet always reports `UNVERIFIED` even when observed effects exceed MESI.
- The runner refuses protocol-byte drift before producing evidence.

## Testing

Tests cover exact propagation, search-cost contrasts, nogood reuse, belief normalization, fidelity traps, all six harness gates, deterministic analysis, Evidence Packet validity, exact 100M accounting, specialized neural region types/shapes, functional-vs-reserve audit and Stage-A multitask backpropagation.
