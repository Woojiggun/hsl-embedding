"""Quickstart — bytes in, signal features out. No tokenizer, no training."""
import hsl_embedding as hsl

# 1) functional: any bytes -> [L, 27] features + [L] phase
feats, phase = hsl.embed("변화율이 공통 언어다".encode())
print("feats", tuple(feats.shape), "| phase", tuple(phase.shape))

# 2) as an nn.Module (no parameters) — drop into a model like nn.Embedding
emb = hsl.Embedding()                      # 27-D exact base (include_bits=True -> 35-D)
print("out_dim", emb.out_dim)
x = emb(b"\x89PNG\r\n\x1a\n")              # works on image bytes just the same
print("image bytes ->", tuple(x.shape))

# 3) named channels — read what each dimension means
names = hsl.feat_names()                    # 27 base channel names (matches feats column order)
row0 = feats[0]
for name, val in list(zip(names, row0.tolist()))[:16]:    # the Δ and Δ² change-rate channels
    print(f"  {name:8} {val:+.2f}")

# 4) lossless — the substrate is byte-exact
b = b"round trip \x00\xff"
assert hsl.decode(hsl.encode(b)) == b
print("lossless:", True)
