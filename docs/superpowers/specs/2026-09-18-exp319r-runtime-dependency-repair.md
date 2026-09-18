# EXP-319R Runtime Dependency Repair

## Status

Procedural repair lineage for EXP-319 Learnability Foundation Diagnostic after authoritative run `35309778949` terminated before any Stage-A artifact was uploaded.

This document does not alter the scientific hypothesis, model geometry, data generators, training budget, optimizer, learning rates, thresholds, selection rules, scoring, reducer, or authorization policy.

## Failed authoritative attempt

Authoritative run:

- run ID: `35309778949`
- workflow: `EXP-319 learnability foundation diagnostic`
- sealed marker: `13efbe9312374f5d64d530c22e4bb58111f3edd1`
- frozen source: `faa51b858e3a211d7f3614d53e6cec9ca0bec74f`
- run identity: `github-run-35309778949-exp319-v1`

Preflight completed successfully, including preregistration verification, marker/execution-identity verification, immutable runtime binding, and exact marker-parent verification.

All six Stage-A chunk-0 coordinates then failed before artifact upload. Stage-A chunk 1+, Stage-A selection/gate, all Stage-B stages, and all Stage-C stages were skipped. The run produced zero workflow artifacts.

Therefore run `35309778949` has no scientific capability disposition. It is a procedural/runtime failure.

## Root cause

The standard runner installs `.[dev,model]`, which provides PyTorch but does not install NumPy.

At the end of a completed training chunk, EXP-319 computes immutable checkpoint provenance. `_tensor_bytes()` first attempts:

```python
materialized.numpy().tobytes(order="C")
```

Without NumPy, PyTorch raises:

```
RuntimeError: Numpy is not available
```

The fallback path then receives scalar optimizer-state tensors and attempts a dtype-changing `view(torch.uint8)` while the tensor is zero-dimensional. PyTorch rejects that conversion with:

```
RuntimeError: self.dim() cannot be 0 to view Float as Byte (different element sizes)
```

The common traceback occurred across the Stage-A arm/LR matrix, so the failure is independent of arm identity or learning rate.

## Repair scope

The minimal repair is workflow-only:

- before every scientific job that installs `.[dev,model]`, explicitly run `python -m pip install numpy`;
- leave Python scientific source, arm geometry, tokenizer, generator, optimizer, budgets, seeds, thresholds, selection, scoring, and reducer unchanged;
- leave Stage-A/B/C geometry unchanged;
- leave all implementation and scale authorization flags false.

Installing NumPy only enables the already-frozen tensor-to-byte provenance path. It does not change the optimizer update rule or the scientific decision thresholds.

## TDD evidence

A workflow-contract regression test requires one explicit NumPy installation for every `.[dev,model]` installation in the scientific workflow.

RED evidence:

- repair test commit: `d6a04b54ba6ae53fb149947e3aa44629a865bfbd`
- helper run: `35310853772`
- result: 1 failed, 7 passed
- failure: workflow had 16 model-runtime installs and 0 NumPy installs

GREEN repair commit:

- `eb0574ed4504779c5c917b51fe4c51a74df5061f`
- scientific workflow now has 16 model-runtime installs and 16 preceding NumPy installs

The GREEN helper run and full exact-head verification must complete before resealing.

## Reseal requirement

The previous EXP-319 marker remains historical evidence for failed run `35309778949`; it must not be reused for the repaired execution.

After the repaired source passes exact-head tests:

1. compute a new EXP-319 execution identity from the repaired source tree;
2. bind the new workflow SHA-256;
3. create a new marker-only commit changing exactly the two execution-identity files;
4. independently verify marker parent, source-tree digest, canonical marker bytes, and exact two-file diff;
5. register the repaired workflow byte-identically on the default branch only if required for dispatch;
6. dispatch exactly one new authoritative run on the repaired marker.

The repaired run receives a new GitHub run ID and therefore a new EXP-319 run identity. No artifacts from run `35309778949` may be treated as scientific inputs to the repaired run.

## Scientific boundary

EXP-301 remains `KILL_H_RD_01`.

EXP-319R does not authorize:

- EXP-302 implementation;
- EXP-320 implementation;
- 30M scaling;
- 100M scaling;
- any other scale increase.

All authorization fields remain false regardless of repair success. A positive repaired EXP-319 result can at most produce `FOUNDATION_READY_FOR_EXP320_DESIGN_ONLY`.
