# EXP-332R — immutable artifact download repair

Authoritative EXP-332 dispatch run `35425548823` failed before scientific execution. Artifact metadata verification succeeded, but the custom Python `urllib` ZIP downloader followed the GitHub artifact redirect into blob storage and received HTTP 401. No training arm ran and no scientific evidence artifact was produced.

This repair changes infrastructure only. It replaces the custom ZIP downloader with official `actions/download-artifact@v5` calls pinned by unique artifact ID, repository and originating run ID. This is required in particular because the two EXP-327 witnesses intentionally have the same artifact name but distinct immutable artifact IDs.

The repair also corrects the verification typo `sha2556sum` to `sha256sum`. The existing metadata authority checks and post-download exact byte SHA-256 checks remain mandatory.

No preregistration field, model, data, optimizer, learning rate, clipping rule, exposure budget, effort cycle, pair lattice, projection rule, reducer, threshold or authorization flag changes. The failed infrastructure run remains preserved and must not be interpreted scientifically.
