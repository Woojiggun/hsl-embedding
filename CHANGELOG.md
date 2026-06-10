# Changelog

## 0.5.1 — 2026-06-10

Documentation only — no code changes; outputs identical to 0.5.0.

- **Practical notes** added to the README and docstrings (things a model integrator must know):
  - Channel scales are heterogeneous **by design** (`fft_re0` = DC bin spans 0–8, most channels
    ±1–2) — apply per-feature scaling (e.g. `LayerNorm`) at the model input boundary; HSL ships
    exact values, not normalized ones.
  - Empty-input behavior made explicit: the bytes path returns `[1, 27]` (`embed(b"")` ≡ one `0x00`
    byte, indistinguishable from `embed(b"\x00")`); the tensor path returns `[..., 0, 27]`.
  - Clarified that the Fourier channels describe the byte's 8-bit pattern (not the waveform's
    temporal spectrum) and that the phase channel is cyclic (255 wraps next to 0).
- PyPI metadata now points at the dedicated repository (github.com/Woojiggun/hsl-embedding);
  `CITATION.cff` ships in the sdist.

## 0.5.0 — 2026-06-10

Substrate-ablation toolkit. **Core encoder untouched — outputs bit-identical to 0.4.0**; the base
substrate remains zero-parameter.

### Added
- `hsl_embedding.ablation` — the controlled A/Bs that isolate where HSL's contribution lives:
  - `feature_groups()` / `channel_indices()` / `select_channels()` — feature-family ablations by
    column selection, plus the canonical **18 value dims / 9 context dims** split (the 18 value dims
    are pure functions of the byte's value — ONE frozen 256×18 LUT, a consequence of the anchor rule;
    Δ² + boundary are the only sequence-dependent channels).
  - `value_lut()` — export the frozen 256×18 per-value LUT (reproducibility packets, custom controls).
  - `ControlEmbedding(kind, seed)` — drop-in counterparts of `hsl.Embedding()` emitting the same
    `[..., L, 27]` layout with HSL's exact context dims in every variant:
    `hsl` (frozen exact LUT — bit-identical to `hsl.Embedding()`), `learned` (trainable
    `nn.Embedding(256, 18)`, seeded init), `random` (fixed injective LUT, per-channel moments matched
    to HSL's — tests whether invertibility alone explains anything), `permuted` (HSL's own rows
    permuted — identical marginal statistics, geometry destroyed: capacity vs geometry).
  - The documented minimal pair: raw bits (8) vs Δ (8) — identical information/dims/scale, geometry
    (Gray code) is the only difference; runnable via `include_bits=True` + `select_channels`.
- `examples/substrate_ablation.py` — the full protocol in one screen.
- `tests/test_ablation.py` — controls share exact context dims, swap only value dims; `permuted`
  preserves per-column multisets exactly; `random` is injective, seeded, moment-matched; `learned`
  is trainable and seed-reproducible; `hsl` is bit-identical to the real embedding.

## 0.4.0 — 2026-06-10

Fast paths & exactness hardening. **Feature values are unchanged — bit-identical to 0.3.0**, verified
over text / image / audio / random / edge inputs across all flag combinations (`include_bits`,
`momentum_phase`), for both `embed()` and the full `encode()` frame.

### Added
- **Tensor / GPU path**: `Embedding()(ids)` now accepts integer tensors `[..., L]` of byte values
  0..255 — batched, runs on the ids' device, and bit-identical to the bytes path (CUDA output verified
  equal to CPU). Buffers are non-persistent: `state_dict()` stays empty, old checkpoints load unchanged.
- Exhaustive **anchor-rule** tests: every byte (8-bit stream) departs from the SAME virtual origin 0 —
  a byte's Δ is identical alone, after any prefix, and at any position (all 256 values × prefixes).
- Boundedness tests: every channel is finite and bounded (Δ/Δ² ∈ {0,1}, boundary ∈ [0, 1.5],
  Fourier ∈ [−4, 8], phase ∈ [−1, 1]); no divergence and no decay-to-zero with sequence length
  (constant streams produce exactly constant features).
- Documented the identity **Δ(v) ≡ binary-reflected Gray code** `v ^ (v >> 1)`: values that differ by
  ±1 differ in exactly ONE Δ coordinate (raw bits would flip up to all 8, e.g. 127→128) — which is why
  Δ² is a faithful momentum measure on smooth byte streams.

### Fixed
- **`boundary` is now exact at every input length.** 0.3.0 accumulated the windowed transition energy
  in a float32 running sum; above ~1.4 MB of input the sum exceeded float32's exact-integer range and
  silently rounded the boundary channel (measured on a 3 MB random input: ~44% of rows off, max error
  1.0). v0.4 computes closed-form per-window sums (last R bits of byte i−1 + first R+1 bits of byte i)
  with a float64 divide — exact at any length. All other channels were already exact.
- Removed a stale comment ("slice drops the closure-transition") describing a no-op slice in `embed()`.

### Changed (performance; values identical)
- `embed()` computes straight from 256-entry LUTs — no per-call bit unpacking. The shortcut is valid
  precisely because of the anchor rule: per-byte channels are pure functions of the byte's own value.
  Measured on the dev machine (1 MB random bytes): bytes path ~1.4× faster; tensor path ~2.7× (CPU)
  and ~30× (CUDA) vs the 0.3 bytes path.

### Removed
- Dead legacy chain-mode helpers `_xor_delta` / `_integrate` (pre-0.3 cross-byte chaining — the
  opposite of the anchor rule) and the unused `_bits_to_bytes`. Only the per-byte anchor rule ships;
  a regression test asserts these symbols stay gone.

## 0.3.0 — 2026-06-09

Baseline of the 27-D exact base (Δ / Δ² / boundary / exact complex rfft / phase), per-byte anchor
semantics, lossless codec, `include_bits` and `momentum_phase` options. See README for details.
