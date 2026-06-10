# HSL — Holistic Signal Language

[![PyPI](https://img.shields.io/pypi/v/hsl-embedding.svg)](https://pypi.org/project/hsl-embedding/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20581805.svg)](https://doi.org/10.5281/zenodo.20581805)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> 🇰🇷 이 프로젝트는 개인 시간에 독립적으로 연구·공개한 오픈 연구 산출물입니다.
> 🇬🇧 This is an independent, open research project — researched and released on personal time.

**A non-learned, byte-level signal encoder for PyTorch.** Instead of splitting text into tokens, it reads
raw bytes *holistically as signal*: change-rate (Δ, XOR-delta), 2nd-order change (Δ²), 편미분-boundary,
**exact complex Fourier**, and phase — **27 dimensions per byte** (35 with raw bits), losslessly invertible.
One modality-agnostic input layer for text, image, audio, video — any byte stream.

> Everything is information — a fluctuation between 0 and 1. HSL doesn't ask *what a token means*; it
> measures *how the signal changes*, with exact formulas, so the same representation works under every modality.
> It's **one base embedding** applied to every modality; *which* channels help *which* modality is decided
> downstream by your model's adapters — nothing is prematurely thrown away at the base.

```python
import hsl_embedding as hsl

feats, phase = hsl.embed(b"hello")          # -> Tensor [L, 27], Tensor [L]
emb = hsl.Embedding()                        # an nn.Module, no parameters (like nn.Embedding)
feats = emb("강아지".encode())               # -> [L, 27]
assert hsl.decode(hsl.encode(b"hello")) == b"hello"   # lossless, by construction
```

> **Import name:** the package is installed as `hsl-embedding` but imported as **`import hsl_embedding`** (`import hsl` will fail).
> **`embed(data)` returns a tuple** `(feats, phase)` — `feats` is a `torch.Tensor [L, 27]` (per-byte features; `[L, 35]` with `include_bits=True`) and `phase` is a `torch.Tensor [L]` (the raw phase angle θ). Unpack both: `feats, phase = hsl.embed(...)`.

## Install

```bash
pip install hsl_embedding      # import as `import hsl_embedding as hsl` (pip treats - and _ the same)
# deps: numpy, torch
```

## Why not just `nn.Embedding`?

They solve **different problems** — this is *not* a performance claim, it's a "when to use which".

| | `torch.nn.Embedding` | `hsl.Embedding` |
|---|---|---|
| what it is | a **learned lookup table** (trainable params) | an **exact formula** (zero params, deterministic) |
| input | a token id (`int`) | raw `bytes` |
| needs | a tokenizer + fixed vocab + training data | nothing — works on any bytes, day one |
| dimensions | opaque, learned | **named & interpretable** (Δ / Δ² / boundary / Fourier / phase) |
| modality | one tokenizer per modality (text ≠ image ≠ audio) | **one substrate for all** (byte-native) |
| invertible | no | **yes** (`decode(encode(x)) == x`) |
| new scripts / formats | breaks / out-of-vocab | just bytes — never breaks |

**They compose.** HSL is an *input substrate*, not a replacement for learned representations: `nn.Embedding`
learns *what tokens mean*; HSL gives *exact structural signal* for free. Stack learned layers **on top** of
HSL features.

**Reach for HSL when** you want: tokenizer-free input · one model across modalities · structure/change-aware
features · exact reconstruction · small-data or from-scratch training · interpretable input channels.

## What each channel captures (and where it's good)

HSL is built from **exact formulas**, each chosen to carry information a plain learned embedding tends to
throw away. The default is the **27-D exact base** — the pure change-rate substrate, every channel lossless:

| channel (dims) | exact formula | captures | especially good for |
|---|---|---|---|
| **Δ** `dxor` 0–7 (8) | **per-byte** XOR-delta from the symbolic anchor 0 (each byte from the anchor → **byte-INDEPENDENT POSITION**); ≡ the byte's binary-reflected **Gray code** `v ^ (v >> 1)`, so values that differ by ±1 differ in exactly **one** Δ coordinate | **change / position** — *where the signal sits, measured from the 0-anchor* | edges, topic/region shifts, the modality-shared "rate of change". *Measured: shift-detection AUC **0.725** vs content **0.698**.* |
| **Δ²** `d2xor` 0–7 (8) | `Δ[byteᵢ] ⊕ Δ[byteᵢ₋₁]` (cross-byte) | **flow / momentum** (2nd order) — how the per-byte position changes *between* bytes | sharp **corners / onsets**; segment cuts, audio attacks, image corners |
| **boundary** (1) | windowed mean of `Δ + 0.5·Δ²` (편미분 경계) | **transition-energy salience** (1st+2nd derivative) | **tokenizer-free segmentation** — natural byte/word/chunk cuts without decoding (heuristic; not part of the codec) |
| **Fourier** `fft_re0–4, fft_im1–3` (8) | **exact** complex 8-bit rFFT (real+imag) | **frequency / texture / periodicity — *and* spectral phase** | smooth vs busy, periodic vs random — audio timbre, image texture. **Lossless/invertible** (`irfft` → byte) |
| **phase** cos/sin (2) | exact phasor `z = e^{iθ}, θ = 2π·byte/256` | **cyclic relation / angle** — exact `cos(θᵢ−θⱼ)` | **affect / mood** and relative/positional structure. *Measured: phase-variation tracks the audio affect-line **0.912**, better than loudness alone.* (`momentum_phase=True`: `z = r·e^{iθ}` carries velocity in the magnitude too.) |

The point: a single learned vector blurs all of this together. HSL keeps **change (Δ), curvature (Δ²),
spectrum (exact Fourier), and phase** as separate, exact, interpretable channels — and your model selects
which ones each modality needs (no premature compression at the base).

*Optional 35-D:* `include_bits=True` prepends the 8 raw byte bits. They're **redundant** (the per-byte Δ
already encodes each byte losslessly) — an optional extra lens, not part of the base.

## Practical notes (read before wiring into a model)

- **Channel scales are heterogeneous by design** — HSL ships *exact* values, not normalized ones.
  `fft_re0` (the DC bin = the byte's bit count) spans **0–8** while most channels sit in ±1–2.
  Apply per-feature scaling at the model input boundary (a `LayerNorm` right after HSL, or a learned
  `Linear(27, d)` — either works); feeding raw features straight into attention lets `fft_re0`
  dominate early training.
- **Empty input:** the bytes path returns `[1, 27]` — `embed(b"")` is treated as a single `0x00`
  byte (deliberate, keeps downstream batching safe), so it is **indistinguishable from
  `embed(b"\x00")`**. The tensor path instead returns `[..., 0, 27]` for zero-length ids
  (batching/padding is the caller's job there). If your pipeline must tell empties apart, mask them
  upstream.
- **Fourier channels read the byte's 8-bit pattern**, not the waveform's temporal spectrum (values
  127 and 128 are spectrally distant); the **phase channel is cyclic**, so byte 255 sits next to
  byte 0. Every lens is exact — which lens fits which modality is your model's call, downstream.

## Lossless by construction

The features are grounded in a lossless codec, so the substrate is byte-exact:

```python
frame = hsl.encode(b"any bytes \x00\xff")
hsl.decode(frame) == b"any bytes \x00\xff"     # True
```
Δ is a **per-byte** XOR-delta from the symbolic anchor 0 — each byte is measured from the anchor (so bytes
are independent), and integrating each byte's Δ from 0 recovers it exactly. That's why the raw `bits` channel
is redundant and can be dropped.

## 27-D (default) vs 35-D (with raw bits)

```python
hsl.embed(data)                      # 27-D  (default exact base; change-rate + exact Fourier + phase)
hsl.embed(data, include_bits=True)   # 35-D  (also prepend the 8 raw bits — redundant optional lens)
hsl.embed(data, momentum_phase=True) # 27-D  (phasor magnitude also carries |Δbyte| velocity)
hsl.Embedding(include_bits=True).out_dim   # 35
```

## Batch

```python
emb = hsl.Embedding()
feats, phase, mask = emb.pack([b"a", b"abcdef"], max_len=8)   # [B, L, D], [B, L], [B, L]
```

## Tensor / GPU path (v0.4)

`Embedding` also accepts **integer tensors of byte values** (0..255) — batched, on any device, and
**bit-identical** to the bytes path. The LUTs ride along as non-persistent buffers (`state_dict()`
stays empty; `.to(device)` moves them with your model):

```python
emb = hsl.Embedding().to(device)
ids = torch.randint(0, 256, (B, L), device=device)   # byte values you already batched yourself
feats = emb(ids)                                     # [B, L, 27] on `device` — torch ops end to end
```

## Substrate ablation toolkit (v0.5)

The 27-D base factors as **18 value dims** (Δ8 + Fourier8 + phase2 — pure functions of the byte's
value, i.e. ONE frozen 256×18 LUT; a consequence of the anchor rule) + **9 context dims** (Δ²8 +
boundary1, the only sequence-dependent channels). `hsl_embedding.ablation` ships the controlled A/Bs
that isolate the value **geometry** — every control keeps HSL's exact context dims and the exact
27-column layout, so the same downstream model runs unchanged:

```python
from hsl_embedding import ablation as ab

ab.ControlEmbedding("hsl")                 # frozen exact LUT — the claim under test
ab.ControlEmbedding("learned", seed=0)     # trainable nn.Embedding(256,18): can SGD find an equivalent?
ab.ControlEmbedding("random", seed=0)      # FIXED injective LUT, HSL per-channel moments —
                                           #   "is invertibility alone enough?" (it preserves all info)
ab.ControlEmbedding("permuted", seed=0)    # HSL's own rows, permuted — identical marginals,
                                           #   geometry destroyed: capacity vs geometry

ab.feature_groups()["value"]               # 18 per-value dims / ["context"] → 9 sequence dims
no_fft = ab.select_channels(feats, ("dxor", "d2xor", "boundary", "phase"))   # feature-family ablations
```

The cheapest, sharpest minimal pair needs no control at all: **raw bits (8) vs Δ (8)** — both
per-byte invertible, identical information / dimensionality / {0,1} scale; the only difference is
geometry (Δ ≡ Gray code: a ±1 value step moves exactly **one** coordinate; raw bits flip up to all 8).
`embed(data, include_bits=True)` + `select_channels(..., ("bits",))` vs `("dxor",)`.

```bash
python examples/substrate_ablation.py     # the full protocol in one screen
```

## Examples

```bash
python examples/quickstart.py        # bytes in, features out; named channels
python examples/roundtrip_all.py     # text / image / audio / video -> embed -> EXACT reconstruction
python examples/vs_nn_embedding.py   # nn.Embedding vs hsl.Embedding — when to use which
python examples/benchmark_vs_nn.py   # honest capability + speed comparison
```

`roundtrip_all.py` — one modality-agnostic encoder, lossless by construction:

```
modality              bytes     feat shape   reconstruction
----------------------------------------------------------------
text  (utf-8)            98       (98, 27)   EXACT ✓
image (RGB u8)         3072     (3072, 27)   EXACT ✓
audio (PCM i16)        8000     (8000, 27)   EXACT ✓
video (6 frames)       4608     (4608, 27)   EXACT ✓
```

## Scope (honest)

HSL is a **non-learned input substrate** — a possibility-proof from an independent, single-GPU project, not a
benchmark-beating system. It gives exact structural signal; the *meaning* still comes from a model you stack on
top. See the paper and live demo:

- 📄 Paper: [A Feasibility Study of Change-Rate-Based Multimodal Unification](https://doi.org/10.5281/zenodo.20581805) (Zenodo)
- 🌐 Live demo: https://holo-demo-p5txmh4dda-as.a.run.app
- 💻 HoLo project: https://github.com/Woojiggun/holo-hsl

## Changelog

**0.5.1** — docs only: "Practical notes" (heterogeneous channel scales — normalize at the model
input boundary; empty-input behavior on both paths; Fourier = bit-pattern spectrum, phase is
cyclic). PyPI metadata now points at this dedicated repo.

**0.5.0** — substrate-ablation toolkit (`hsl_embedding.ablation`): channel-group selection
(`feature_groups` / `select_channels`), the frozen 256×18 value-LUT export (`value_lut`), and
capacity-matched control substrates (`ControlEmbedding`: `hsl` / `learned` / `random` / `permuted`)
sharing HSL's exact context dims and layout — controlled A/Bs over the value geometry, reproducible
from `pip install` alone. Core encoder untouched: outputs bit-identical to 0.4.0; the base substrate
remains zero-parameter (the `learned` control is an explicitly-labeled experimental baseline).

**0.4.0** — fast paths & exactness hardening. **Feature values are unchanged — bit-identical to 0.3.0** (verified over text/image/audio/random/edge inputs × all flag combos):

- **Tensor / GPU path**: `Embedding()(ids)` with integer tensors `[..., L]` — batched, device-agnostic,
  bit-identical to the bytes path (measured ~30× faster than the 0.3 bytes path on CUDA, ~2.7× on CPU).
- `embed()` now computes straight from 256-entry LUTs (no per-call bit unpacking) — valid precisely
  because of the **anchor rule**: every byte departs from the same virtual origin 0, so every per-byte
  channel is a pure function of the byte's own value.
- **`boundary` made exact at every input length**: 0.3.0 accumulated the windowed transition energy in a
  float32 running sum, which silently rounded the boundary channel above ~1.4 MB of input (on a 3 MB
  random input, ~44% of rows were off by up to 1.0). Now closed-form per-window sums with a float64
  divide — exact at any length. All other channels were already exact; **no approximation ships**.
- Documented the identity **Δ(v) ≡ binary-reflected Gray code** `v ^ (v >> 1)` — adjacent byte values
  differ in exactly one Δ coordinate (raw bits flip up to all 8, e.g. 127→128); exhaustive anchor-rule
  tests added (every byte's Δ is identical alone, after any prefix, at any position).
- Removed the dead legacy chain-mode helpers (`_xor_delta`, `_integrate`, `_bits_to_bytes`) — only the
  per-byte anchor rule ships.

## License & citation

**MIT License — © 2026 Jinhyun Woo (ggunio5782@gmail.com).**
Free to use, modify, and **distribute, including for commercial use** — the only condition is that the
copyright notice and attribution to **Jinhyun Woo** are kept. See [LICENSE](LICENSE).

```bibtex
@software{woo_hsl_2026,
  author = {Jinhyun Woo},
  title  = {HSL: a byte-native, modality-agnostic signal embedding},
  year   = {2026},
  doi    = {10.5281/zenodo.20581805},
  url    = {https://github.com/Woojiggun/hsl-embedding}
}
```
