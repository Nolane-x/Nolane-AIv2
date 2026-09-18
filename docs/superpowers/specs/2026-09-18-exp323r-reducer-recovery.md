# EXP-323R reducer recovery

## Purpose

Authoritative repaired EXP-323R run `35354410137` completed both scientific intervention arms successfully:

- HOLD_5E5 artifact id `10551422072`, ZIP digest `sha256:c71ff48acfe7de3e3b6a96b7a967e391a7746948385e707efaaf1b3598b12093`;
- DECAY_2P5E5 artifact id `10551321837`, ZIP digest `sha256:eda5e2fb46ecbe6b7e1bc6c8b6a3f014bfe4c72df31fad9e96d121163b516708`.

Both arm jobs concluded SUCCESS.

The run failed only in the reducer job because the reducer environment installed `.[dev]` without Torch, while the frozen reducer imports `exp323r_repair.py`, which imports Torch at module load.

This is a procedural reducer-environment defect. It occurred after both scientific arms had completed and uploaded immutable evidence. No arm retraining, replay, reselection, threshold change, or scientific mutation is permitted.

## Recovery contract

The recovery helper must:

1. bind exact sealed marker `217477a911aabbb764741e76cca38e54013e7da6`;
2. bind exact frozen source `471d211e15b4840d06536e9b2edbc404958612b2`;
3. verify the sealed execution identity with the frozen verifier;
4. verify both arm artifact IDs, names, expiry state and ZIP digests;
5. download only those two immutable arm artifacts from run `35354410137`;
6. install the exact frozen source with `.[dev,model]`;
7. invoke the unchanged frozen `scripts/exp323r_reduce.py`;
8. require all EXP-302 / EXP-320 / scale / 30M / 100M authorization flags to remain false;
9. upload exactly one recovered final evidence artifact.

No optimizer step, evaluation pass, replay, arm execution, or scientific selection is allowed in this helper.
