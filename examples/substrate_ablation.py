"""Substrate ablation harness — controlled A/Bs that isolate WHERE HSL's contribution lives.

Protocol: hold params / context / FLOPs / data / seeds matched; swap ONLY the substrate.
  kind      isolates
  hsl       the claim under test (frozen exact value geometry)
  learned   can SGD find an equivalent/better per-value representation? (+4,608 params)
  random    is information-preservation (injectivity) alone enough?
  permuted  identical marginal statistics, geometry destroyed — capacity vs geometry
All four share HSL's exact 9 context dims (Δ², boundary) and the same [B, L, 27] layout, so the
same downstream model runs unchanged — swap one line, train, compare.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import torch
import hsl_embedding as hsl
from hsl_embedding import ablation as ab

ids = torch.randint(0, 256, (4, 128), generator=torch.Generator().manual_seed(0))
ref = hsl.Embedding()(ids)
ctx = ab.feature_groups()["context"]

print("kind       trainable params   context dims == HSL exact   value dims")
print("-" * 70)
for kind in ab.ControlEmbedding.KINDS:
    m = ab.ControlEmbedding(kind, seed=0)
    out = m(ids)                                          # [4, 128, 27] — drop-in for hsl.Embedding()
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"{kind:9} {n:16,}   {str(torch.equal(out[..., ctx], ref[..., ctx])):25}   "
          f"{'frozen exact LUT' if kind == 'hsl' else kind}")

# feature-family ablations are just column selections
feats, _ = hsl.embed(b"ablation by column selection")
no_fft = ab.select_channels(feats, ("dxor", "d2xor", "boundary", "phase"))
print(f"\nfeature-family ablation: drop Fourier -> {tuple(no_fft.shape)} (from {tuple(feats.shape)})")

# the cheapest, sharpest minimal pair: raw bits(8) vs Δ(8) — same information, same dims,
# same {0,1} scale; the ONLY difference is geometry (Δ ≡ Gray code)
f, _ = hsl.embed(bytes(range(256)), include_bits=True)
bits = ab.select_channels(f, ("bits",), include_bits=True)
dxor = ab.select_channels(f, ("dxor",), include_bits=True)
mb = (bits[1:] - bits[:-1]).abs().sum(1)
md = (dxor[1:] - dxor[:-1]).abs().sum(1)
print("\nminimal pair, adjacent byte values 0..255 (feature movement per +1 step):")
print(f"  raw bits : mean {mb.mean():.2f}, max {int(mb.max())}   (127->128 flips all 8)")
print(f"  Delta    : mean {md.mean():.2f}, max {int(md.max())}   (Gray code: always exactly 1)")
print("\nSame information, same capacity — only the GEOMETRY differs. That is the substrate question.")
