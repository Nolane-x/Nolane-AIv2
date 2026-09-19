# EXP-332R2 — Historical Attempt Artifact Retrieval Repair

The sealed EXP-332R dispatch run `35427157225` failed before scientific execution while retrieving the original EXP-327 attempt-1 artifact `10574837015`.

GitHub's `actions/download-artifact@v5` resolves artifacts through a workflow `run-id` and only enumerated the latest attempt for run `35413434081`. It therefore saw the attempt-2 artifact but could not select the still-existing attempt-1 artifact by ID. The metadata authority gate had already verified artifact `10574837015`, its exact name, non-expired status and ZIP digest.

EXP-332R2 changes only artifact transport for that historical-attempt witness. It requests the exact artifact-ID archive endpoint with the GitHub token, captures the API's signed redirect URL, and downloads that signed blob URL without forwarding the GitHub token. The existing exact JSON SHA-256 check remains mandatory after extraction.

Reconstruction, EXP-330 parent and EXP-327 attempt-2 artifacts continue to use `actions/download-artifact@v5`.

No preregistration, model, optimizer, learning rate, training order, projection rule, threshold, reducer, population, authorization flag or scientific interpretation changes.
