"""HSL — Holistic Signal Language: a non-learned, byte-level signal embedding (codec + encoder in one).

Everything is information — a fluctuation between 0 and 1. HSL turns raw bytes into a compact,
*change-rate-based* feature signal that any modality (text, image, audio, video, sensor) shares,
with no tokenizer and no learned parameters. The representation is grounded in a lossless codec,
so `decode(encode(x)) == x` — the substrate is byte-exact by construction.

THE ANCHOR RULE (the invariant everything rests on): every byte — every 8-bit stream — departs from
the SAME virtual origin 0. Δ is measured from that shared anchor, never from the previous byte's last
bit, so a byte's Δ depends on nothing but the byte's own value (no cross-byte contamination), and
integrating each byte's Δ from the anchor recovers it exactly. A byte's Δ is identical alone, after
any prefix, and at any position — position-independent by construction. Identity: Δ(v) is exactly the
binary-reflected GRAY CODE of v, `v ^ (v >> 1)` — so values that differ by ±1 differ in exactly ONE
Δ coordinate (raw bits would flip up to all 8, e.g. 127→128).

ONE modality-agnostic BASE embedding (FEAT_DIM = 27). Every lens is EXACT/lossless — nothing is
prematurely compressed here, because *which* lens helps *which* modality is decided downstream by the
model's per-modality adapters (selective input), not thrown away at this universal base:
    dxor0..7   (8)  Δ   = POSITION — per-byte change-rate from the SYMBOLIC anchor 0 (each byte measured
                    from the anchor, so bytes are INDEPENDENT; ≡ the byte's Gray code)
    d2xor0..7  (8)  Δ²  = FLOW / momentum — how the per-byte Δ changes BETWEEN bytes: Δ²[i] = Δ[i] ⊕ Δ[i-1]
                    (the "Δ — Δ² — Δ" connection; position is byte-independent, momentum is the cross-byte relation)
    boundary   (1)  편미분 경계 — windowed mean of transition energy Δ + 0.5·Δ² (1st+2nd derivative);
                    a heuristic salience signal, NOT part of the lossless codec
    fft_re0..4, fft_im1..3 (8)  EXACT complex rfft of the 8 bits — lossless/invertible, keeps the
                    spectral PHASE; minimal non-redundant form (imag[DC]=imag[Nyquist]=0 for real input)
    phase_cos/sin (2)  exact complex phasor z = e^{iθ}, θ = 2π·byte/256 (momentum_phase: z = r·e^{iθ})

The raw 8 bits are NOT included by default: Δ-from-origin-0 already encodes the bytes losslessly,
so the bits are redundant. Pass include_bits=True for the 35-D variant (raw bits prepended).

All features are BOUNDED, finite and drift-free by construction: Δ/Δ² ∈ {0,1}, boundary ∈ [0, 1.5],
Fourier ∈ [-4, 8] (fft_re0 = bit count ∈ [0, 8]), phase ∈ [-1, 1], momentum radius ∈ [0.5, ~0.998].
There are no data-dependent divisions, logs or recurrences anywhere — nothing can produce NaN/Inf,
diverge, or decay toward 0 with sequence length.

    import hsl_embedding as hsl
    feats, phase = hsl.embed(b"hello")     # [L, 27], [L]
    emb = hsl.Embedding();  feats = emb(b"hello")
    ids = torch.randint(0, 256, (4, 128)); feats = emb(ids)   # v0.4: batched tensor path, any device
    assert hsl.decode(hsl.encode(b"hello")) == b"hello"

Author: Jinhyun Woo (ggunio5782@gmail.com). MIT-licensed; no learned weights included.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

__version__ = "0.4.0"

__all__ = ["__version__", "FEAT_DIM", "FEAT_DIM_FULL", "FEAT_NAMES", "FEAT_NAMES_FULL", "feat_names",
           "ORIGIN_BIT", "CLOSURE_BIT", "BOUNDARY_D2_WEIGHT", "BOUNDARY_WINDOW_RADIUS",
           "HSLFrame", "encode", "decode", "embed", "Embedding"]

ORIGIN_BIT = 0          # the "0": codec origin — the ONE virtual zero every byte departs from
                        # (Δ integrates from here on decode; lossless reconstruction)
CLOSURE_BIT = 1         # the "1": fixed codec closure bit — appended to the bit-signal, verified on decode
                        # (end-of-content marker for lossless reconstruction). Codec-internal: NOT learned,
                        # and NOT part of embed()'s features. (The HoLo *model* separately learns an
                        # end-marker; that is a model property, not this non-learned library's.)
BOUNDARY_D2_WEIGHT = 0.5      # weight of Δ² (2nd-order) vs Δ (1st-order) in the boundary transition energy
BOUNDARY_WINDOW_RADIUS = 4    # ±bits averaged around each byte start for the boundary salience score

FEAT_DIM = 27           # the modality-agnostic BASE: Δ8 + Δ²8 + boundary1 + exact-Fourier8 + phase2
FEAT_DIM_FULL = 35      # include_bits=True: also prepend the 8 raw bits (redundant with Δ; optional lens)
_BITS = [f"bit{i}" for i in range(8)]
_FFT = [f"fft_re{i}" for i in range(5)] + [f"fft_im{i}" for i in (1, 2, 3)]   # 8-D exact complex rfft
_DELTAS = [f"dxor{i}" for i in range(8)] + [f"d2xor{i}" for i in range(8)] + ["boundary"]
_PHASE = ["phase_cos", "phase_sin"]
FEAT_NAMES = _DELTAS + _FFT + _PHASE          # 27 (base)
FEAT_NAMES_FULL = _BITS + FEAT_NAMES          # 35 (with raw bits)


def feat_names(include_bits: bool = False):
    return (_BITS + FEAT_NAMES) if include_bits else list(FEAT_NAMES)


# ───────────────────────────── codec (numpy, lossless) ─────────────────────────────
@dataclass(frozen=True)
class HSLFrame:
    payload_len_bytes: int
    bits: np.ndarray
    signal: np.ndarray         # data bits + CLOSURE_BIT(1) appended (Δ integrates from origin 0 to invert losslessly)
    delta: np.ndarray          # Δ  (XOR-delta from origin 0)
    delta2: np.ndarray         # Δ²
    byte_boundary_score: np.ndarray   # 편미분 경계 (heuristic salience; not read by decode)


def _bytes_to_bits(data: bytes) -> np.ndarray:
    if not data:
        return np.zeros((0,), dtype=np.uint8)
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big").astype(np.uint8)


def _per_byte_delta(bits2d: np.ndarray, origin: int = ORIGIN_BIT) -> np.ndarray:
    """Δ (1st change-rate) = POSITION. THE ANCHOR RULE: each byte's 8 bits are XOR-delta'd from the SAME
    virtual origin (the symbolic 0) — never from the previous byte's last bit — so bytes are INDEPENDENT
    (no cross-byte contamination) and a byte's Δ is a pure function of its own value.
    bits2d [N,8] → [N,8]. Integrate-per-byte from the anchor recovers each byte exactly."""
    prev = np.empty_like(bits2d)
    prev[:, 0] = origin                                  # every byte STARTS from the anchor (the virtual 0)
    prev[:, 1:] = bits2d[:, :-1]
    return np.bitwise_xor(bits2d, prev).astype(np.uint8)


def _cross_byte_delta2(delta2d: np.ndarray, origin: int = ORIGIN_BIT) -> np.ndarray:
    """Δ² (2nd change-rate) = FLOW / momentum. How the per-byte Δ (position) changes BETWEEN consecutive
    bytes: Δ²[i] = Δ[i] ⊕ Δ[i-1], with Δ²[0] = Δ[0] ⊕ anchor. The 'connection structure' Δ — Δ² — Δ:
    position is byte-independent (Δ), momentum is the cross-byte relation (Δ²). delta2d [N,8] → [N,8]."""
    prev = np.empty_like(delta2d)
    prev[0, :] = origin
    prev[1:, :] = delta2d[:-1, :]
    return np.bitwise_xor(delta2d, prev).astype(np.uint8)


def _boundary_windowed(energy2d: np.ndarray) -> np.ndarray:
    """Windowed mean of per-bit transition energy around each byte start — closed form, exact.
    The window [8i−R, 8i+R+1) is the LAST R bits of byte i−1 plus the FIRST R+1 bits of byte i
    (R < 8, so it spans at most one byte boundary; the right edge never clips since 8i+R+1 ≤ 8n;
    the left edge clips only at byte 0). Row sums of ≤(2R+1) half-integers are exact, and the
    division is done in float64 before the float32 store — so the score is EXACT at every input
    length. (v0.3 used a float32 running sum that silently rounded above ~1.4 MB of input.)"""
    n = energy2d.shape[0]
    R = BOUNDARY_WINDOW_RADIUS
    head = energy2d[:, : R + 1].sum(axis=1, dtype=np.float64)    # first R+1 bits of byte i
    tail = energy2d[:, 8 - R:].sum(axis=1, dtype=np.float64)     # last R bits of byte i (the i−1 term)
    out = np.empty(n, dtype=np.float64)
    out[0] = head[0] / (R + 1)                                   # byte 0: window clips at bit 0
    if n > 1:
        out[1:] = (tail[:-1] + head[1:]) / (2 * R + 1)
    return out.astype(np.float32)


def _byte_boundary(delta: np.ndarray, delta2: np.ndarray, nbytes: int) -> np.ndarray:
    """편미분 경계 — per-byte salience = windowed mean of transition energy Δ + BOUNDARY_D2_WEIGHT·Δ²
    (1st + 2nd derivative). A heuristic: NOT read by decode, carries no losslessness obligation."""
    if nbytes == 0:
        return np.zeros((0,), dtype=np.float32)
    energy2d = (delta.astype(np.float32)
                + np.float32(BOUNDARY_D2_WEIGHT) * delta2.astype(np.float32)).reshape(nbytes, 8)
    return _boundary_windowed(energy2d)


def encode(data: bytes) -> HSLFrame:
    """bytes → HSLFrame. Δ = per-byte POSITION from the ONE virtual origin 0 (byte-INDEPENDENT); Δ² =
    cross-byte FLOW/momentum (Δ[i]⊕Δ[i-1]). Context-level 0→1: the origin anchor and the CLOSURE_BIT
    endpoint. Lossless."""
    n = len(data)
    bits = _bytes_to_bits(data)                          # flat [8n]
    signal = np.concatenate([bits, np.asarray([CLOSURE_BIT], dtype=np.uint8)])   # context 0→1 endpoint
    if n == 0:
        z = np.zeros((0,), dtype=np.uint8)
        return HSLFrame(0, bits, signal, z, z, np.zeros((0,), dtype=np.float32))
    b2 = bits.reshape(n, 8)
    delta_b = _per_byte_delta(b2)                        # [n,8] Δ  position (byte-independent)
    delta2_b = _cross_byte_delta2(delta_b)               # [n,8] Δ² flow/momentum (cross-byte)
    delta, delta2 = delta_b.reshape(-1), delta2_b.reshape(-1)
    return HSLFrame(n, bits, signal, delta, delta2, _byte_boundary(delta, delta2, n))


def decode(frame: HSLFrame) -> bytes:
    """HSLFrame → bytes. Integrate Δ PER BYTE from the ONE virtual origin 0 (each byte independent), then
    verify the context CLOSURE ("1" endpoint). Lossless by construction."""
    if frame.signal.size < 1 or int(frame.signal[-1]) != CLOSURE_BIT:
        raise ValueError("closure / length check failed")
    n = frame.payload_len_bytes
    if n == 0:
        return b""
    db = np.asarray(frame.delta, dtype=np.uint8).reshape(n, 8)
    bits = np.bitwise_xor.accumulate(db, axis=1)         # per-byte prefix-XOR from anchor 0 → recovers the bits
    return np.packbits(bits, axis=1).tobytes()


# ───────────────────────────── embedding (LUT fast path + torch tensor path) ──────────────────
def _build_luts():
    """Per-byte channels are pure functions of the byte's value (0..255) — precompute all 256 rows.
    The LUT shortcut is valid PRECISELY because of the anchor rule: every byte departs from the same
    virtual origin 0, so Δ (and everything derived per byte) depends on nothing outside the byte."""
    b = np.arange(256, dtype=np.uint8)
    bits_u8 = np.unpackbits(b[:, None], axis=1)                            # [256, 8] uint8
    dxor_u8 = _per_byte_delta(bits_u8)                                     # Δ of every value, BY the codec's own rule
    gray = np.bitwise_xor(b, b >> 1)                                       # binary-reflected Gray code
    assert (np.packbits(dxor_u8, axis=1).ravel() == gray).all()            # identity: Δ(v) ≡ gray(v)
    bits = bits_u8.astype(np.float32)
    spec = np.fft.rfft(bits, axis=1)                                       # [256, 5] COMPLEX (amplitude + phase)
    # EXACT complex rfft in minimal non-redundant form: for real 8-pt input the imaginary parts of the
    # DC[0] and Nyquist[4] bins are identically 0, so real[0:5] + imag[1:4] = 8 real DOF fully and
    # losslessly represent the 8 bits (invertible via irfft). Guard the invariant the 8-D rests on.
    assert np.abs(spec.imag[:, 0]).max() < 1e-6 and np.abs(spec.imag[:, 4]).max() < 1e-6
    fourier = np.concatenate([spec.real, spec.imag[:, 1:4]], 1).astype(np.float32)    # [256, 8]
    angle = (b.astype(np.float32) / 256.0 * (2.0 * math.pi))               # [256]
    phasor = np.stack([np.cos(angle), np.sin(angle)], 1).astype(np.float32)  # [256, 2]
    return bits, dxor_u8.astype(np.float32), gray, fourier, phasor, angle.astype(np.float32)


_BITS_LUT, _DXOR_LUT, _GRAY_LUT, _FOURIER_LUT, _PHASE_LUT, _ANGLE_LUT = _build_luts()


def embed(data: bytes, include_bits: bool = False, momentum_phase: bool = False):
    """bytes → (feats [L, 27|35], phase [L]). Deterministic, non-learned — the ONE modality-agnostic
    base embedding; every lens is exact/lossless.

    v0.4 computes straight from the 256-entry LUTs (no per-call bit unpacking). The shortcut is valid
    precisely because of the anchor rule: every byte starts at the same virtual origin 0, so its Δ is a
    pure function of the byte's own value. Output is bit-identical to the codec path (and to v0.3).

    include_bits=True (+8-D → 35): also prepend the 8 raw bits — redundant with Δ (an optional extra lens).
    momentum_phase=True (no extra dims): the phasor z = e^{iθ} (unit magnitude, angle = byte value)
        becomes z = r·e^{iθ}, r = 0.5 + 0.5·|Δbyte|/256 — ONE complex carries BOTH position (angle) and
        momentum (magnitude). Sequence-dependent (origin 0 for the first byte). The angle (= value,
        = affect channel) is preserved exactly; the magnitude adds velocity.
    Empty input is treated as a single 0x00 byte (returns L=1), for safe downstream batching.
    """
    if len(data) == 0:
        data = b"\x00"                                                # empty → one 0x00 byte (documented)
    arr = np.frombuffer(data, dtype=np.uint8)
    dxor = _DXOR_LUT[arr]                                             # [L,8] Δ from the ONE virtual origin (≡ Gray bits)
    g = _GRAY_LUT[arr]                                                # Δ packed as a byte (= gray(value))
    gprev = np.empty_like(g)
    gprev[0] = 0xFF if ORIGIN_BIT else 0x00                           # byte 0's Δ² partner is the anchor itself
    gprev[1:] = g[:-1]
    d2xor = _BITS_LUT[np.bitwise_xor(g, gprev)]                       # [L,8] Δ² = bits(Δᵢ ⊕ Δᵢ₋₁)
    boundary = _boundary_windowed(dxor + np.float32(BOUNDARY_D2_WEIGHT) * d2xor)[:, None]
    fourier = _FOURIER_LUT[arr]                                       # [L,8] EXACT complex rfft (lossless, keeps phase)
    phasor = _PHASE_LUT[arr]                                          # [L,2] z = e^{iθ}, angle = byte value
    if momentum_phase:                                                # z = r·e^{iθ}: angle=position, magnitude=momentum
        dv = np.abs(np.diff(arr.astype(np.int16), prepend=0)).astype(np.float32)   # |byteₜ-byteₜ₋₁| (ARITHMETIC value diff, not the XOR-Δ)
        r = (0.5 + 0.5 * dv / 256.0)[:, None]                         # ∈[0.5, 0.998]; keeps angle readable, encodes velocity
        phasor = (phasor * r).astype(np.float32)                      # ONE complex carries position + momentum
    cols = [dxor, d2xor, boundary, fourier, phasor]
    if include_bits:
        cols.insert(0, _BITS_LUT[arr])                                # [L,8] raw bits (redundant lens)
    feats = np.concatenate(cols, axis=1)                              # one float32 array
    return torch.from_numpy(feats), torch.from_numpy(_ANGLE_LUT[arr])


class Embedding(nn.Module):
    """HSL byte → signal embedding as an nn.Module (no parameters), usable like nn.Embedding.

        self.hsl = hsl.Embedding()
        feats = self.hsl(b"...")                 # [L, 27]  (include_bits=True -> [L, 35])
        feats, phase = self.hsl(b"...", return_phase=True)
        feats = self.hsl(ids)                    # v0.4: integer tensor [..., L] of byte values 0..255
                                                 #       → [..., L, out_dim]; batched, runs on ids' device

    The tensor path is bit-identical to the bytes path. Buffers are non-persistent — state_dict()
    stays empty, so checkpoints saved around earlier versions load unchanged, and .to(device) moves
    the LUTs along with the model.
    """
    def __init__(self, include_bits: bool = False, momentum_phase: bool = False):
        super().__init__()
        self.include_bits = include_bits
        self.momentum_phase = momentum_phase                         # phasor magnitude carries |Δbyte| (no extra dims)
        self.out_dim = FEAT_DIM + (8 if include_bits else 0)         # 27, or 35 with raw bits
        self.register_buffer("_bits_lut", torch.from_numpy(_BITS_LUT.copy()), persistent=False)
        self.register_buffer("_dxor_lut", torch.from_numpy(_DXOR_LUT.copy()), persistent=False)
        self.register_buffer("_gray_lut", torch.from_numpy(_GRAY_LUT.astype(np.int64)), persistent=False)
        self.register_buffer("_fourier_lut", torch.from_numpy(_FOURIER_LUT.copy()), persistent=False)
        self.register_buffer("_phase_lut", torch.from_numpy(_PHASE_LUT.copy()), persistent=False)
        self.register_buffer("_angle_lut", torch.from_numpy(_ANGLE_LUT.copy()), persistent=False)

    def forward(self, data, return_phase: bool = False):
        if isinstance(data, torch.Tensor):
            feats, phase = self._embed_ids(data)
        else:
            feats, phase = embed(data, self.include_bits, self.momentum_phase)
        return (feats, phase) if return_phase else feats

    def _embed_ids(self, ids: torch.Tensor):
        """ids [..., L] integer byte values 0..255 → (feats [..., L, out_dim], phase [..., L]).
        Same math as embed(), in torch ops — batched, device-agnostic, bit-identical."""
        if ids.is_floating_point() or ids.dtype == torch.bool:
            raise TypeError(f"ids must be an integer tensor of byte values 0..255, got {ids.dtype}")
        if ids.dim() == 0:
            raise ValueError("ids needs a length dimension — shape [..., L]")
        idx = ids.long()
        if ids.dtype != torch.uint8 and idx.numel():                 # uint8 cannot leave 0..255
            if int(idx.min()) < 0 or int(idx.max()) > 255:
                raise ValueError("byte values must be in 0..255")
        L = idx.shape[-1]
        if L == 0:                                                   # empty stays empty in the tensor path
            return (idx.new_zeros((*idx.shape[:-1], 0, self.out_dim), dtype=torch.float32),
                    idx.new_zeros((*idx.shape[:-1], 0), dtype=torch.float32))
        dxor = self._dxor_lut[idx]                                   # [..., L, 8] Δ from the ONE virtual origin
        g = self._gray_lut[idx]                                      # Δ packed as an int (= gray(value))
        gprev = torch.full_like(g, 0xFF) if ORIGIN_BIT else torch.zeros_like(g)
        gprev[..., 1:] = g[..., :-1]                                 # byte 0's Δ² partner is the anchor itself
        d2xor = self._bits_lut[torch.bitwise_xor(g, gprev)]          # [..., L, 8] Δ² = bits(Δᵢ ⊕ Δᵢ₋₁)
        energy = dxor + BOUNDARY_D2_WEIGHT * d2xor                   # per-bit transition energy (exact halves)
        R = BOUNDARY_WINDOW_RADIUS
        head = energy[..., : R + 1].sum(dim=-1, dtype=torch.float64)
        tail = energy[..., 8 - R:].sum(dim=-1, dtype=torch.float64)
        bnum = torch.empty_like(head)
        bnum[..., 0] = head[..., 0] / (R + 1)
        if L > 1:
            bnum[..., 1:] = (tail[..., :-1] + head[..., 1:]) / (2 * R + 1)
        boundary = bnum.to(torch.float32).unsqueeze(-1)              # same float64-divide-then-store as numpy
        fourier = self._fourier_lut[idx]
        phasor = self._phase_lut[idx]
        if self.momentum_phase:                                      # z = r·e^{iθ}: angle=position, magnitude=momentum
            v = idx.to(torch.float32)
            vprev = torch.zeros_like(v)
            vprev[..., 1:] = v[..., :-1]
            r = (0.5 + 0.5 * (v - vprev).abs() / 256.0).unsqueeze(-1)
            phasor = phasor * r
        cols = [dxor, d2xor, boundary, fourier, phasor]
        if self.include_bits:
            cols.insert(0, self._bits_lut[idx])                      # raw bits (redundant lens)
        return torch.cat(cols, dim=-1), self._angle_lut[idx]

    def pack(self, byte_list: list[bytes], max_len: int):
        """list[bytes] → feats[B,L,out_dim], phase[B,L], mask[B,L] (pad/truncate to max_len).
        Pad rows are all-zero — distinguishable from a real 0x00 byte, whose phase_cos = 1."""
        B = len(byte_list)
        feats = torch.zeros(B, max_len, self.out_dim)
        phase = torch.zeros(B, max_len)
        mask = torch.zeros(B, max_len)
        for i, data in enumerate(byte_list):
            f, p = embed(data, self.include_bits, self.momentum_phase)
            n = min(f.shape[0], max_len)
            feats[i, :n], phase[i, :n], mask[i, :n] = f[:n], p[:n], 1.0
        return feats, phase, mask
