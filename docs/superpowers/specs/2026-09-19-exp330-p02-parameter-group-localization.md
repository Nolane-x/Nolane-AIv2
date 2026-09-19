# EXP-330 — P02 Parameter-Group Cross-Damage Localization Court

EXP-329 sealed `BIDIRECTIONAL_LOCAL_CROSS_DAMAGE` on the exact P02 `ALT_0_2` trajectory. The parent court showed that a source-self-improving one-step update can directly increase the other world's same-effort answer-only loss in both directions. The parent result is local to worlds 0 and 2 under the frozen A_FIXED 10M reconstruction; it is not a universal claim about gradient conflict or capacity.

EXP-330 asks one narrower question: **which existing structural parameter groups can independently express the registered direct cross-damage when the authoritative model state is held fixed?**

The court changes no architecture, tokenizer, objective, optimizer, learning rate, scale, exposure count, replay ordering, or effort geometry. The authoritative main replay remains the exact EXP-329 `ALT_0_2` replay from the immutable step-2048 reconstruction. Every diagnostic branch is a disposable deep clone.

## Frozen structural partition

Every trainable A_FIXED named parameter must belong to exactly one nonempty group:

- `embedding_output`: the tied `backbone.token_embedding.weight` matrix used for input embeddings and output logits;
- `layer0`: all parameters under `backbone.layers.0.*`;
- `layer1`: all parameters under `backbone.layers.1.*`;
- `layer2`: all parameters under `backbone.layers.2.*`;
- `final_norm`: all parameters under `backbone.final_norm.*`;
- `capacity_exchange`: all parameters under `capacity_exchange.*`.

Unknown parameters, overlap, an empty registered group, or any trainable-parameter count other than exactly 10,000,000 invalidates the court.

## Probe geometry

EXP-330 probes only the exact union of EXP-329 parent damage rounds: `1, 4, 8, 16, 24, 31`. Before the authoritative update at each round:

1. compute answer-only gradients for world 0 and world 2 from the identical main state at the registered effort;
2. decompose gradient dot product, cosine and per-world norms by structural group;
3. for each group and source direction, deep-clone model+optimizer;
4. compute the ordinary full source gradient and apply the ordinary global norm clip at 1.0;
5. set gradients outside the selected group to `None`, which makes AdamW skip both parameter and optimizer-state updates for those parameters;
6. execute exactly one selected-group optimizer step;
7. evaluate source and target answer-only loss at the same effort;
8. discard the clone and restore torch CPU RNG.

A group-isolated directed damage event requires source loss improvement of at least `1e-6` and target loss increase of at least `1e-6`. Group gradient cosine is evidence only and cannot determine the disposition by itself.

## Exact parent reproduction gate

After every non-perturbing probe, the main replay continues unchanged with world 0 then world 2. After 32 exposures/world, EXP-330 requires the registered parent world outcomes **and** the exact EXP-329 final model, optimizer and RNG digests. A mismatch blocks all parameter-group interpretation.

## Bounded dispositions

The reducer distinguishes invalid/reproduction mismatch, a single bidirectionally damaging group, multi-group damage, directionally split group damage, one-direction-only group damage, or no isolated-group direct damage. These outcomes localize evidence; they do not authorize a mitigation.

In particular, EXP-330 does not authorize freezing, projecting, routing, gradient surgery, optimizer changes, learning-rate changes, architecture changes, EXP-302, EXP-320, or scaling. 30M and 100M authorization remain false.
