# EXP-277 EV-E3 persistence receipt

- Authoritative ceremony run: `34323835734`
- Exact armed ceremony SHA: `dd37ab9d0c59f9b86a9e75e190acd6436c3401ac`
- Scientific baseline: `b449099f4a8ac5673a9bb55be6c735511ebd786d`
- Source evidence artifact ID: `10093277653`
- Source evidence artifact digest: `sha256:043ad0015258a03cadbd7d6705474756425f1753af7bfba0ceb26d23ecbb247b`
- Prior pre-inference failure run: `34318307479`
- Prior arm SHA: `a24b1d3220b8d95934ce08979c08d5a00e899900`
- Prior failure boundary: `PRE_BEACON_PRE_INFERENCE`
- Prior failure artifact count: `0`
- Canonical Stage-A V1 digest: `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`
- Scientific outcome: `EV-E3 / KILL_SUBSYSTEM`

The persistence workflow validates immutable GitHub Actions metadata, all original `SHA256SUMS`, frozen source/protocol/checkpoint/beacon identities, inference-start evidence, primary/protected endpoints, and the evidence-only diff boundary. It does not invoke Gate B, derive a second challenge, retrain, or regenerate any scientific observation.
