from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class V017Geometry:
    vocab_size: int = 4_608
    d_model: int = 448
    n_heads: int = 7
    head_dim: int = 64
    shared_layers: int = 3
    d_ff: int = 1_152
    max_loops: int = 16
    rms_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.d_model <= 0 or self.shared_layers <= 0:
            raise ValueError("V0.17 geometry dimensions must be positive")
        if self.n_heads <= 0 or self.head_dim <= 0 or self.n_heads * self.head_dim != self.d_model:
            raise ValueError("n_heads * head_dim must equal d_model")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embedding")
        if self.d_ff <= 0 or self.max_loops <= 0:
            raise ValueError("d_ff and max_loops must be positive")


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, *, eps: float = 1e-6, device=None) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(d_model, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def _apply_rope(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = q.shape[-1]
    half = head_dim // 2
    position = torch.arange(q.shape[-2], device=q.device, dtype=torch.float32)
    frequency = torch.arange(half, device=q.device, dtype=torch.float32)
    inv_frequency = torch.exp(-math.log(10_000.0) * frequency / half)
    angles = torch.outer(position, inv_frequency).to(dtype=q.dtype)
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]

    def rotate(x: torch.Tensor) -> torch.Tensor:
        left, right = x[..., :half], x[..., half:]
        return torch.cat((left * cos - right * sin, left * sin + right * cos), dim=-1)

    return rotate(q), rotate(k)


class CausalSelfAttention(nn.Module):
    def __init__(self, geometry: V017Geometry, *, device=None) -> None:
        super().__init__()
        self.d_model = geometry.d_model
        self.n_heads = geometry.n_heads
        self.head_dim = geometry.head_dim
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=False, device=device)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, timesteps, _ = x.shape
        qkv = self.qkv(x).view(batch, timesteps, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = _apply_rope(q, k)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view(batch, timesteps, self.d_model)
        return self.out_proj(attended)


class SwiGLU(nn.Module):
    def __init__(self, geometry: V017Geometry, *, device=None) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(geometry.d_model, geometry.d_ff, bias=False, device=device)
        self.up_proj = nn.Linear(geometry.d_model, geometry.d_ff, bias=False, device=device)
        self.down_proj = nn.Linear(geometry.d_ff, geometry.d_model, bias=False, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class V017TransformerLayer(nn.Module):
    def __init__(self, geometry: V017Geometry, *, device=None) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(geometry.d_model, eps=geometry.rms_eps, device=device)
        self.attention = CausalSelfAttention(geometry, device=device)
        self.norm_ff = RMSNorm(geometry.d_model, eps=geometry.rms_eps, device=device)
        self.feed_forward = SwiGLU(geometry, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm_attn(x))
        return x + self.feed_forward(self.norm_ff(x))


class V017CoreBackbone(nn.Module):
    def __init__(self, geometry: V017Geometry | None = None, *, device=None) -> None:
        super().__init__()
        self.geometry = geometry or V017Geometry()
        self.token_embedding = nn.Embedding(self.geometry.vocab_size, self.geometry.d_model, device=device)
        self.layers = nn.ModuleList(
            V017TransformerLayer(self.geometry, device=device)
            for _ in range(self.geometry.shared_layers)
        )
        self.final_norm = RMSNorm(self.geometry.d_model, eps=self.geometry.rms_eps, device=device)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(tokens)

    def forward_block(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def finalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_norm(x)


class ActiveCapacityExchange(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        bottleneck: int,
        tail_parameters: int,
        device=None,
    ) -> None:
        super().__init__()
        if d_model <= 0 or bottleneck <= 0 or tail_parameters < 0:
            raise ValueError("capacity-exchange dimensions must be positive except an optional zero tail")
        if tail_parameters > d_model:
            raise ValueError("tail_parameters cannot exceed d_model")
        self.d_model = int(d_model)
        self.bottleneck = int(bottleneck)
        self.tail_parameters = int(tail_parameters)
        self.up = nn.Linear(self.d_model, self.bottleneck, bias=False, device=device)
        self.down = nn.Linear(self.bottleneck, self.d_model, bias=False, device=device)
        if self.tail_parameters:
            self.tail = nn.Parameter(torch.empty(self.tail_parameters, device=device))
            if device != "meta":
                nn.init.normal_(self.tail, mean=0.0, std=0.02)
            direction = torch.linspace(-1.0, 1.0, self.d_model, device=device)
            direction = direction / direction.square().mean().sqrt().clamp_min(1e-12)
            self.register_buffer("tail_direction", direction, persistent=False)
        else:
            self.register_parameter("tail", None)
            self.register_buffer("tail_direction", None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.down(F.silu(self.up(x))) / math.sqrt(self.bottleneck)
        if self.tail is None:
            return x + residual
        tail_signal = (x[..., : self.tail_parameters] * self.tail).sum(dim=-1, keepdim=True)
        tail_signal = tail_signal / math.sqrt(self.tail_parameters)
        return x + residual + tail_signal * self.tail_direction


class LoopConditioner(nn.Module):
    def __init__(self, *, d_model: int, max_loops: int, device=None) -> None:
        super().__init__()
        if d_model <= 0 or max_loops <= 0:
            raise ValueError("loop-conditioner dimensions must be positive")
        self.d_model = int(d_model)
        self.max_loops = int(max_loops)
        self.embedding = nn.Embedding(self.max_loops, self.d_model, device=device)

    def forward(self, x: torch.Tensor, *, loop_indices: int | torch.Tensor) -> torch.Tensor:
        if isinstance(loop_indices, int):
            indices = torch.full((x.shape[0],), loop_indices, device=x.device, dtype=torch.long)
        else:
            indices = loop_indices.to(device=x.device, dtype=torch.long)
            if indices.ndim == 0:
                indices = indices.expand(x.shape[0])
        if indices.ndim != 1 or indices.shape[0] != x.shape[0]:
            raise ValueError("loop_indices must be scalar or have one index per batch item")
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= self.max_loops):
            raise ValueError(f"loop index must be in [0, {self.max_loops})")
        return x + self.embedding(indices).unsqueeze(1)


class V017RecurrentLM(nn.Module):
    def __init__(
        self,
        *,
        geometry: V017Geometry | None = None,
        capacity_bottleneck: int,
        capacity_tail_parameters: int = 192,
        loop_conditioning: bool,
        device=None,
    ) -> None:
        super().__init__()
        self.geometry = geometry or V017Geometry()
        self.backbone = V017CoreBackbone(self.geometry, device=device)
        self.capacity_exchange = ActiveCapacityExchange(
            d_model=self.geometry.d_model,
            bottleneck=capacity_bottleneck,
            tail_parameters=capacity_tail_parameters,
            device=device,
        )
        self.loop_conditioner = (
            LoopConditioner(
                d_model=self.geometry.d_model,
                max_loops=self.geometry.max_loops,
                device=device,
            )
            if loop_conditioning
            else None
        )

    @property
    def output_weight(self) -> torch.nn.Parameter:
        return self.backbone.token_embedding.weight

    def forward(self, tokens: torch.Tensor, *, loops: int) -> torch.Tensor:
        if loops <= 0 or loops > self.geometry.max_loops:
            raise ValueError(f"loops must be in [1, {self.geometry.max_loops}]")
        x = self.backbone.embed(tokens)
        for loop_index in range(loops):
            if self.loop_conditioner is not None:
                x = self.loop_conditioner(x, loop_indices=loop_index)
            x = self.backbone.forward_block(x)
            x = self.capacity_exchange(x)
        x = self.backbone.finalize(x)
        return F.linear(x, self.output_weight)


def build_simple_recurrent_10m(*, device=None) -> V017RecurrentLM:
    model = V017RecurrentLM(
        capacity_bottleneck=981,
        capacity_tail_parameters=192,
        loop_conditioning=False,
        device=device,
    )
    if count_trainable_parameters(model) != 10_000_000:
        raise RuntimeError(f"simple recurrent arm parameter drift: {count_trainable_parameters(model):,}")
    return model


def build_nrs_core_10m(*, device=None) -> V017RecurrentLM:
    model = V017RecurrentLM(
        capacity_bottleneck=973,
        capacity_tail_parameters=192,
        loop_conditioning=True,
        device=device,
    )
    if count_trainable_parameters(model) != 10_000_000:
        raise RuntimeError(f"NRS core arm parameter drift: {count_trainable_parameters(model):,}")
    return model
