# EXP-323R reducer finalization recovery

## Scope

This recovery closes authoritative EXP-323R run `35354410137` without rerunning either scientific arm.

The source run reached the following immutable state:

- workflow event: `workflow_dispatch`
- sealed marker: `217477a911aabbb764741e76cca38e54013e7da6`
- frozen scientific source: `471d211e15b4840d06536e9b2edbc404958612b2`
- execution digest: `70ebb1bda4c91f457aae1007df0d939f2f2a144257dfebf5481ee6dc55b10cd9`
- `HOLD_5E5` job: SUCCESS
- `DECAY_2P5E5` job: SUCCESS
- reducer job: FAILURE before reducer logic because Torch was absent from the reducer environment.

The reducer failure was:

`ModuleNotFoundError: No module named 'torch'`

The failure occurred while importing the already-frozen `exp323r_repair.py` module. No reducer decision or final artifact was emitted by the source run.

## Immutable arm authority

HOLD:

- artifact id: `10551422072`
- artifact name: `exp323r-HOLD_5E5-35354410137`
- artifact ZIP digest: `c71ff48acfe7de3e3b6a96b7a967e391a7746948385e707efaaf1b3598b12093`
- evidence-file SHA-256: `4cb2296f8c7a40727f5087ed6da5b01b4379f84111668a107c5b23ad25caa0d1`
- arm evidence digest: `34326f59ceb05030403bad187c09b29f5b103e711551b451c60680858efe2946`

DECAY:

- artifact id: `10551321837`
- artifact name: `exp323r-DECAY_2P5E5-35354410137`
- artifact ZIP digest: `eda5e2fb46ecbe6b7e1bc6c8b6a3f014bfe4c72df31fad9e96d121163b516708`
- evidence-file SHA-256: `a05cc80410e3c7281122bfee7cde343262690a2c04cf8085188d75914ff9c451`
- arm evidence digest: `e91840bc531f74bf7899e0ddb587c55330df4b60610e9ba29e28879e76d88ac6`

Both artifacts report:

- `invalid_reason=null`
- `completed_step=3072`
- identical EXP-323R execution identity
- identical immutable step-2048 reconstruction authority
- every implementation/scale authorization flag false.

## Recovery rule

The recovery MUST:

1. verify the source run identity and job outcomes;
2. verify exact artifact IDs, names, ZIP digests and evidence-file hashes;
3. checkout the exact frozen scientific source `471d211e...`;
4. install the missing model dependency only so the frozen reducer module can import;
5. run the unchanged frozen `scripts/exp323r_reduce.py`;
6. verify the final evidence self-digest, arm digests, execution identity and authorization boundary;
7. emit a recovery receipt binding the final result to the original failed reducer run.

The recovery MUST NOT:

- call `exp323r_run_arm.py`;
- call any train/replay/materialization entrypoint;
- perform optimizer steps;
- evaluate a model;
- alter thresholds, arm metrics or decision order;
- create a second scientific execution.

The preregistered reducer geometry implies `NO_REGISTERED_RESCUE` from the immutable arm evidence, but that string is not authoritative until the unchanged frozen reducer emits and seals it.
