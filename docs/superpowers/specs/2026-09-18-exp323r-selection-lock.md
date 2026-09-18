# EXP-323R selection lock

The bounded reconstruction pool from run `35345351869` is closed.

Verified candidates existed at attempts 0 and 2. Both carried the exact same reconstructed checkpoint bytes (SHA-256 `4aa03459b5266a3455bcfbc8cb070d9ceaeb0e7b8390944e7d483b0953e567c5`). The preregistered deterministic rule therefore selects attempt 0, the lowest successful index.

Selected immutable input:

- artifact id: `10547681681`
- artifact name: `exp323r-reconstruction-candidate-35345351869-0`
- artifact ZIP digest: `c0862235302243e6d9ef689ea6ef7214431ce44f41575aea12954524ad1ac621`
- checkpoint SHA-256: `4aa03459b5266a3455bcfbc8cb070d9ceaeb0e7b8390944e7d483b0953e567c5`
- reconstruction receipt digest: `f8153c9f88d7d28b7e0240d994991c8d232b1d192b1688b901982610e266e03f`

The materialization helper is now manual-only. Repaired scientific execution must consume this selected artifact and must not replay steps 1024-2048.
