# Nolane-AIv2 — NLM V0.16.1 100M Research Substrate

This repository is the executable starting point for the **Nolane Living Model (NLM) V0.16.1** research program.

## What is frozen now

- Authoritative candidate footprint: **100,000,000 parameters**.
- Frozen support: **10,000,000**; currently trainable allocation: **90,000,000**.
- Canonical namespaces: `EXP-###` for experiments, `EV-*` for evidence maturity.
- `NLM Reasoning Stage-A Confirmatory Protocol v1` for six first gates:
  - EXP-277 — oracle constraint-structure headroom
  - EXP-279 — propagation vs branch vs hybrid
  - EXP-282 — explicit belief state vs recurrent hidden state
  - EXP-286 — oracle conflict-core value
  - EXP-289 — episode-local nogood learning
  - EXP-297 — compile-valid semantic-fidelity traps

The V0.16.1 source required these experiments to freeze arms, endpoints, MESI, sample-size planning, analysis, multiplicity, resource matching, failure rules, challenge rules and metric cards before confirmatory evidence exists. The numeric MESI choices in `protocols/stage_a_v1.json` are **new V1 execution commitments**, not claimed empirical results.

## What this repository does *not* claim

The 100M scaffold is not yet a trained or validated intelligence. Exact parameter accounting, deterministic seeds, passing tests and protocol hashes are structural/process evidence only. They do not promote any NLM neural hypothesis above `UNVERIFIED`.

## Architecture bootstrap

The PyTorch model has a functional residual path in each active named region and an explicit `capacity_reserve` filling the remainder of that region's budget. This is intentional: complexity that has not survived Stage-A is represented as unallocated capacity rather than silently turned into a decorative neural subsystem.

Once a gate survives falsification, reserve can be replaced by the earned mechanism while preserving the region budget and matched-rival accounting.

## Quick verification

```bash
python -m pip install -e '.[dev,model]'
pytest -q
python scripts/verify_protocol.py
```

To verify the full 100M shape without allocating the parameter storage:

```python
from nolane_ai.model.config import NLMConfig
from nolane_ai.model.nlm import NolaneLivingModel

model = NolaneLivingModel(NLMConfig.authoritative_100m(), device="meta")
print(sum(p.numel() for p in model.parameters()))  # 100000000
```

## Evidence discipline

- Architecture-caused divergence is a scientific failure and remains in the result ledger.
- Post-freeze challenge seeds are intentionally unavailable before freeze.
- Same hashes establish artifact identity, not scientific truth.
- Practical equivalence selects the simpler rival.
- Negative Stage-A evidence is allowed to delete NLM subsystems.
