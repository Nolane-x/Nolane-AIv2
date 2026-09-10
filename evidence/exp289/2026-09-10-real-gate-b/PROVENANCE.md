# EXP-289 real Gate-B provenance

Authoritative ceremony run: GitHub Actions run `34495415427` on ceremony SHA `cb03e14ed48df7a486570edbe7a6469c2c8708f0`, scientific baseline `55c73f05ebf9a94385e887a9ee6c3746bdaf0e8c`.

The run completed successfully and produced irreversible arm-lock, inference-start, raw, outcome, and evidence artifacts. The public beacon was drand round `6454100`, selected as the first round strictly after the actual Gate-A seal boundary. The ceremony consumed the frozen confirmatory replicate set `30032..30063` at `n=32` and executed the frozen decision rule once.

Scientific outcome: `EV-E3`, `PROMOTE_TO_NEXT_STAGE`. Primary paired relative repeat-dead-end-rate reduction was `1.0`, exceeding frozen MESI `0.25`; the valid-state over-prune guard observed `0.0` against ceiling `0.005`; verified-solution-rate difference observed `0.0` against floor `-0.01`. No epsilon denominator rescue was used.

Later workflow hardening required raw execution to depend explicitly on successful durable inference-barrier upload. This does not authorize a rerun: the historical ceremony artifact record itself shows the inference-start artifact was successfully published before the raw artifact (`15:27:02Z` vs `15:27:22Z`). The hardening closes the failure path for future ceremonies; it does not erase already-consumed confirmatory data.

Full raw evidence remains in Actions artifact `10159705168` (`exp289-real-gate-b-evidence-cb03e14ed48df7a486570edbe7a6469c2c8708f0`, artifact digest `sha256:4be4fed9ecc0ab75e5f6bd374231f4d25df7c8a1a3b352ffaf614638128c2d3a`). The repository copy intentionally persists the compact decision/provenance court plus SHA256 manifest; the 21 MB raw artifact is not rewritten or regenerated.

Scope boundary: this result applies only to the frozen EXP-289 episode-local exact-nogood challenge family. It does not establish lifelong lemma economy, learned-clause transfer, integrated NLM, or open-domain reasoning authority.
