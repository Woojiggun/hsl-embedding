"""Substrate-ablation toolkit — the controlled A/Bs that isolate WHERE HSL's contribution lives.

The 27-D base factors cleanly:
    18 VALUE dims   = Δ8 + Fourier8 + phase2 — pure functions of the byte's value, i.e. ONE frozen
                      256×18 lookup table (a consequence of the anchor rule: every byte departs from
                      the same virtual origin 0, so per-byte channels depend on nothing outside the byte)
    9  CONTEXT dims = Δ²8 + boundary1 — the only sequence-dependent channels
A substrate ablation should therefore hold the 9 context dims fixed and swap ONLY the value geometry.
`ControlEmbedding` ships exactly that family, all emitting the SAME [..., L, 27] layout as
`hsl.Embedding()` so downstream model code is byte-compatible:

    kind='hsl'       frozen exact HSL LUT — the claim under test
    kind='learned'   trainable nn.Embedding(256, 18) (+4,608 params) + the SAME exact context dims —
                     "can SGD find an equivalent / better per-value representation?"
    kind='random'    FIXED random injective LUT (seeded; per-channel mean/std matched to HSL's) —
                     "is information-preservation (invertibility) alone enough?" A random injective
                     map preserves all information; if it underperforms, invertibility is NOT the
                     active ingredient — the geometry is.
    kind='permuted'  HSL's own 256 LUT rows, randomly permuted over byte values (seeded) — per-channel
                     MARGINAL statistics exactly identical, value-adjacency geometry destroyed.
                     The sharpest "capacity vs geometry" control.

For the classic fully-learned baseline (learn the context too), no library support is needed — that is
just `nn.Embedding(256, d)`. `ControlEmbedding` answers the finer question: value geometry under
identical information content, capacity, dimensionality, layout and context channels.

The cheapest, sharpest minimal pair needs no control class at all: raw bits (8) vs Δ (8). Both are
per-byte invertible — identical information, dimensionality and {0,1} scale; the ONLY difference is
geometry (Δ ≡ Gray code: ±1 value steps move exactly one coordinate; raw bits flip up to all 8,
e.g. 127→128). Run it with `embed(data, include_bits=True)` and
`select_channels(feats, ("bits",), include_bits=True)` vs `("dxor",)`.

Feature-family ablations (drop Δ² / Fourier / phase / …) are column selections: `feature_groups()`,
`channel_indices()`, `select_channels()`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import (FEAT_DIM, FEAT_NAMES, FEAT_NAMES_FULL, Embedding,
               _DXOR_LUT, _FOURIER_LUT, _PHASE_LUT)

__all__ = ["GROUPS", "VALUE_DIM", "feature_groups", "channel_indices", "select_channels",
           "value_lut", "ControlEmbedding"]

GROUPS = ("bits", "dxor", "d2xor", "boundary", "fourier", "phase", "value", "context")
VALUE_DIM = 18          # dxor8 + fourier8 + phase2 — the per-value LUT width


def feature_groups(include_bits: bool = False) -> dict:
    """Channel-group name → column indices (in embed()'s column order).
    'value' = the per-byte-value dims (bits? + dxor + fourier + phase); 'context' = Δ² + boundary,
    the only sequence-dependent dims."""
    names = FEAT_NAMES_FULL if include_bits else FEAT_NAMES
    out: dict = {}
    for i, n in enumerate(names):
        if n.startswith("d2xor"):
            g = "d2xor"
        elif n.startswith("dxor"):
            g = "dxor"
        elif n.startswith("bit"):
            g = "bits"
        elif n == "boundary":
            g = "boundary"
        elif n.startswith("fft"):
            g = "fourier"
        else:
            g = "phase"
        out.setdefault(g, []).append(i)
    out["value"] = (out.get("bits", []) + out["dxor"] + out["fourier"] + out["phase"])
    out["context"] = out["d2xor"] + out["boundary"]
    return out


def channel_indices(groups, include_bits: bool = False) -> list:
    """Sorted, de-duplicated column indices for the given group names."""
    fg = feature_groups(include_bits)
    idx: set = set()
    for g in ([groups] if isinstance(groups, str) else groups):
        if g not in fg:
            raise KeyError(f"unknown channel group {g!r} — choose from {sorted(fg)}")
        idx.update(fg[g])
    return sorted(idx)


def select_channels(feats, groups, include_bits: bool = False):
    """feats [..., D] → the columns of the given groups, e.g. select_channels(f, ("dxor", "phase"))."""
    return feats[..., channel_indices(groups, include_bits)]


def value_lut() -> np.ndarray:
    """The frozen 256×18 per-value LUT (column order: Δ8 | Fourier8 | phase2) — what hsl.Embedding's
    value dims ARE. Useful for exporting feature tensors (reproducibility packets) and custom controls."""
    return np.concatenate([_DXOR_LUT, _FOURIER_LUT, _PHASE_LUT], axis=1)


class ControlEmbedding(nn.Module):
    """Drop-in A/B counterpart of hsl.Embedding() for substrate ablations (tensor path only).

    All kinds share HSL's EXACT 9 context dims (Δ², boundary) and the exact 27-column layout;
    only the 18 value dims change (see module docstring for what each kind isolates):

        for kind in ("hsl", "learned", "random", "permuted"):
            emb = ControlEmbedding(kind, seed=0).to(device)
            feats = emb(ids)                       # [..., L, 27] — swap-in identical to hsl.Embedding()

    Only the 'learned' kind has trainable parameters (256×18 = 4,608; seeded init). The substrate
    itself stays zero-parameter — controls are explicitly-labeled experimental baselines, not HSL.
    """
    KINDS = ("hsl", "learned", "random", "permuted")

    def __init__(self, kind: str, seed: int = 0):
        super().__init__()
        if kind not in self.KINDS:
            raise ValueError(f"kind must be one of {self.KINDS}, got {kind!r}")
        self.kind, self.seed = kind, seed
        self.out_dim = FEAT_DIM
        self._hsl = Embedding()                       # exact reference: context dims + layout (0 params)
        base = value_lut()
        if kind == "learned":
            self.value = nn.Embedding(256, VALUE_DIM)
            g = torch.Generator().manual_seed(seed)   # seeded init → multi-seed runs are reproducible
            with torch.no_grad():
                self.value.weight.copy_(torch.randn(256, VALUE_DIM, generator=g))
        else:
            if kind == "hsl":
                lut = base
            elif kind == "permuted":                  # same rows — marginals identical, geometry destroyed
                lut = base[np.random.RandomState(seed).permutation(256)]
            else:                                     # 'random': injective, per-channel moments = HSL's
                z = np.random.RandomState(seed).randn(256, VALUE_DIM).astype(np.float32)
                z = (z - z.mean(0)) / z.std(0)        # exact sample moments after rescale
                lut = z * base.std(0) + base.mean(0)
                assert np.unique(lut.round(5), axis=0).shape[0] == 256   # information-preserving
            self.register_buffer("_value_lut", torch.from_numpy(np.ascontiguousarray(lut)),
                                 persistent=False)

    def forward(self, ids: torch.Tensor, return_phase: bool = False):
        if not isinstance(ids, torch.Tensor):
            raise TypeError("ControlEmbedding is a training-time A/B tool — pass integer id tensors "
                            "[..., L]; use hsl.embed()/hsl.Embedding() for bytes")
        feats, phase = self._hsl._embed_ids(ids)      # exact 27, incl. the shared context dims
        idx = ids.long()
        val = self.value(idx) if self.kind == "learned" else self._value_lut[idx]   # [..., L, 18]
        out = torch.cat([val[..., 0:8],               # the Δ slot       (cols  0:8)
                         feats[..., 8:17],            # Δ² + boundary    (cols  8:17) — IDENTICAL everywhere
                         val[..., 8:18]], dim=-1)     # Fourier + phase slots (cols 17:27)
        return (out, phase) if return_phase else out
