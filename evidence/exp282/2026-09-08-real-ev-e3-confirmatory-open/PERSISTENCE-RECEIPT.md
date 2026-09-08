# EXP-282 EV-E3 persistence receipt

This receipt attests the persistence-only copy produced by GitHub Actions persistence run `34225185401`.

- Source ceremony run: `34223658313`
- Exact armed ceremony SHA: `68deb39cb33f7697dbc1f88006a2b4868ae6fed3`
- Exact-head ceremony CI run: `34223665546`
- Source evidence artifact ID: `10055052239`
- Source evidence artifact digest: `sha256:51aefb693e2bde56e9e276cbbb8cc9d8f986b51f615961a9241883f62b1ce10a`
- Persistence bot commit: `f5d7611282a554c46d345d31193717aedd3eba9a`
- Scientific baseline: `77383b0a9c2ce92ded66f07234891790652a037b`
- Canonical Stage-A V1 digest: `c010d90b9d626cfde4f7727b76fcd053f1ebf7f2f7aa201d7d13d2d828cef440`

The persistence workflow completed successfully after validating the source artifact metadata, `SHA256SUMS`, frozen EXP-282 validators, and evidence-only staged path boundary. This receipt does not regenerate or re-execute scientific observations.

The authoritative frozen EV-E3 result remains `KILL_SUBSYSTEM` for the EXP-282 small-model partial-observability comparison. It remains bounded by `challenge_materialized=false`; EV-E4 post-freeze challenge replication and EV-E5 independent clean-room replication were not performed by this ceremony.

The evidence artifact still records the generic EV-E4/EV-E5 evidence-ladder blockers because no post-freeze challenge or clean-room replication was materialized. The approved closure plan applies the frozen decision fork after observing `KILL_SUBSYSTEM`: **persist/stop and do not run EV-E4 for the explicit-belief hypothesis**. Any future attempt to revisit that hypothesis would require a new development/protocol decision rather than a same-protocol continuation or outcome-shopping rerun.
