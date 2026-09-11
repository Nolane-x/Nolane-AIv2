# EXP-286 real Gate-B provenance

Authoritative ceremony run: GitHub Actions run `34541785902` on ceremony SHA `5b23a45d00260128be41aa68c06eb2471751850e`, scientific baseline `962a9964d27506ce387bbdfdf8e7c11f339bf5e1`.

The run completed successfully and produced irreversible arm-lock, inference-start, raw, outcome, and evidence artifacts. The public beacon was drand round `6455054`, selected as the first round strictly after the actual Gate-A seal boundary (`2026-09-10T23:23:41.300668119Z`; beacon publication `2026-09-10T23:24:00Z`). Primary and mirror HTTP records agreed on round, randomness, and signature. The ceremony intentionally records `EXTERNAL_EVIDENCE_RECORDED`; it did **not** cryptographically verify the drand signature, so this persistence does not overclaim beacon authenticity.

Scientific outcome: `EV-E3`, `PROMOTE_TO_NEXT_STAGE`. The frozen primary endpoint was `accounted_reasoning_flops_to_verified_solution` with paired `n=32`. Observed relative reduction was `0.25`; the one-sided lower bound was `0.25`, exceeding frozen MESI `0.15`. The protected verified-solution-rate difference was `0.0`, above the frozen floor `-0.005`. Alpha remained `0.05`; the frozen decision rule executed exactly once on scientific, non-test data.

Full raw evidence remains in Actions artifact `10177594585` (`exp286-real-gate-b-evidence-5b23a45d00260128be41aa68c06eb2471751850e`, artifact digest `sha256:736ebb237afd93b68c2739673a8dc058a545888bc3fc87e397aeeda36a527d5f`). The repository copy intentionally persists the compact decision/provenance court plus the SHA256 manifest; the multi-megabyte raw artifact is not rewritten or regenerated.

Scope boundary: this result supports the frozen EXP-286 comparison `chronological_failure` vs `oracle_conflict_core` on the authoritative paired challenge family under the frozen Stage-A protocol. It does not by itself establish Stage-B integrated-system benefit, open-domain reasoning authority, transfer beyond this challenge family, or a general claim about lifelong neural-core performance.
