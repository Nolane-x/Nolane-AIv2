from __future__ import annotations

import sys

import torch


def tensor_raw_bytes(tensor: torch.Tensor) -> bytes:
    """Return dense tensor payload bytes without requiring NumPy.

    A clone is intentional: ``contiguous()`` may return a view that still owns a
    larger backing storage. Cloning guarantees the untyped storage contains
    exactly this tensor and nothing else. The caller must bind shape/dtype (and,
    where relevant, byte order) into the surrounding digest metadata.
    """
    if tensor.layout != torch.strided:
        raise TypeError(f"raw tensor hashing only supports strided tensors, got {tensor.layout}")
    cpu = tensor.detach().cpu().contiguous().clone()
    raw = bytes(cpu.untyped_storage())
    expected_nbytes = cpu.numel() * cpu.element_size()
    if len(raw) != expected_nbytes:
        raise RuntimeError(
            f"tensor raw-byte size mismatch: expected {expected_nbytes}, got {len(raw)}"
        )
    return raw


def tensor_byteorder() -> str:
    """Expose the host byte order so raw-byte digests do not hide endian drift."""
    return sys.byteorder
