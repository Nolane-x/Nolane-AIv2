from __future__ import annotations

import torch

from .exp279_routing_worlds import (
    STRATA,
    Exp279RoutingBatch,
    _batch_digest,
)


CHALLENGE_STREAM = "challenge"
CHALLENGE_SCOPE = "post-freeze-exp279-confirmatory-challenge"


def build_exp279_challenge_batch(
    *,
    challenge_seed: int,
    replicate: int,
    stratum: str,
    batch_size: int,
    timesteps: int,
    variables: int,
    constraints: int,
    d_model: int,
    noise_std: float,
    device: str | torch.device = "cpu",
) -> Exp279RoutingBatch:
    """Build one post-freeze EXP-279 challenge batch from the sealed seed directly.

    ``challenge_seed`` is already the output of the preregistered public-beacon
    derivation.  This builder deliberately does not pass it through the
    DEVELOPMENT root-seed/stream derivation a second time.
    """

    if (
        not isinstance(challenge_seed, int)
        or isinstance(challenge_seed, bool)
        or challenge_seed < 0
    ):
        raise ValueError("challenge_seed must be a non-negative integer")
    if not isinstance(replicate, int) or isinstance(replicate, bool) or replicate < 0:
        raise ValueError("replicate must be a non-negative integer")
    if stratum not in STRATA:
        raise ValueError(f"stratum must be one of {STRATA}")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in (batch_size, timesteps, variables, constraints, d_model)
    ):
        raise ValueError("EXP-279 challenge geometry must be positive integers")
    if constraints > variables:
        raise ValueError("EXP-279 challenge constraints cannot exceed variables")
    if timesteps < constraints:
        raise ValueError("EXP-279 challenge timesteps must cover every constraint")
    if not isinstance(noise_std, (int, float)) or isinstance(noise_std, bool) or noise_std < 0.0:
        raise ValueError("EXP-279 challenge noise_std must be non-negative")

    generator = torch.Generator(device="cpu").manual_seed(challenge_seed)

    base = torch.arange(variables, dtype=torch.long) % constraints
    membership = torch.stack(
        [base[torch.randperm(variables, generator=generator)] for _ in range(batch_size)],
        dim=0,
    )
    incidence = torch.zeros(batch_size, constraints, variables, dtype=torch.float32)
    incidence.scatter_(1, membership.unsqueeze(1), 1.0)

    anchors = torch.randint(
        0,
        2,
        (batch_size, constraints),
        generator=generator,
        dtype=torch.long,
    )
    targets = anchors.gather(1, membership)

    component_codes = torch.randn(
        batch_size,
        constraints,
        d_model,
        generator=generator,
    )
    component_codes = component_codes / component_codes.norm(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-8)
    anchor_axis = torch.randn(d_model, generator=generator)
    anchor_axis = anchor_axis / anchor_axis.norm().clamp_min(1e-8)
    gather_index = membership.unsqueeze(-1).expand(-1, -1, d_model)
    variable_states = component_codes.gather(1, gather_index)
    variable_sign = targets.to(torch.float32).mul(2.0).sub(1.0).unsqueeze(-1)

    if stratum == "PROPAGATION_FIT":
        variable_states = variable_states + 0.75 * variable_sign * anchor_axis
        variable_noise = float(noise_std) * 0.5
        event_anchor_scale = 0.45
        semantics = (
            "dense aligned factor support; low residual ambiguity after propagation"
        )
    elif stratum == "BRANCH_FIT":
        variable_noise = float(noise_std) * 2.0 + 0.05
        event_anchor_scale = 1.25
        semantics = (
            "ambiguous local factor evidence; stronger sequential anchor evidence "
            "for recurrent branch deliberation"
        )
    else:
        component_mask = (membership % 2 == 0).to(torch.float32).unsqueeze(-1)
        variable_states = (
            variable_states
            + 0.55 * component_mask * variable_sign * anchor_axis
        )
        variable_noise = float(noise_std)
        event_anchor_scale = 0.85
        semantics = (
            "mixed factor support with residual ambiguity requiring routing "
            "discrimination"
        )

    if variable_noise:
        variable_states = variable_states + variable_noise * torch.randn(
            batch_size,
            variables,
            d_model,
            generator=generator,
        )

    order = torch.randperm(constraints, generator=generator)
    event_rows: list[torch.Tensor] = []
    for step in range(timesteps):
        component = int(order[step % constraints].item())
        sign = (
            anchors[:, component]
            .to(torch.float32)
            .mul(2.0)
            .sub(1.0)
            .unsqueeze(-1)
        )
        event = (
            component_codes[:, component, :]
            + event_anchor_scale * sign * anchor_axis.unsqueeze(0)
        )
        if noise_std:
            event = event + float(noise_std) * torch.randn(
                batch_size,
                d_model,
                generator=generator,
            )
        event_rows.append(event)
    surface_events = torch.stack(event_rows, dim=1)

    metadata = {
        "scope": CHALLENGE_SCOPE,
        "experiment_id": "EXP-279",
        "replicate": replicate,
        "rng_stream": CHALLENGE_STREAM,
        "seed": challenge_seed,
        "challenge_seed": challenge_seed,
        "stratum": stratum,
        "stratum_semantics": semantics,
        "predeclared_structure_fit_strata": list(STRATA),
        "batch_size": batch_size,
        "timesteps": timesteps,
        "variables": variables,
        "constraints": constraints,
        "d_model": d_model,
        "noise_std": float(noise_std),
    }

    surface_events = surface_events.to(device)
    variable_states = variable_states.to(device)
    incidence = incidence.to(device)
    targets = targets.to(device)
    tensors = (
        ("surface_events", surface_events),
        ("variable_states", variable_states),
        ("incidence", incidence),
        ("targets", targets),
    )
    return Exp279RoutingBatch(
        surface_events=surface_events,
        variable_states=variable_states,
        incidence=incidence,
        targets=targets,
        stratum=stratum,
        metadata=metadata,
        digest=_batch_digest(metadata, tensors),
        replicate=replicate,
        rng_stream=CHALLENGE_STREAM,
    )
