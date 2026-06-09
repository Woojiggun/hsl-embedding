"""HSL vs torch.nn.Embedding — an honest benchmark (capabilities + a few real measurements).

This is NOT a "HSL is better" pitch. They are different tools:
  nn.Embedding is a fast learned lookup table; HSL is an exact, invertible, modality-agnostic signal.
We report what each *can* and *cannot* do, and we're upfront that nn.Embedding is faster at raw lookup.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")   # so the ✓/✗ marks print on any console (e.g. Windows cp949)
except Exception: pass
import time
import numpy as np
import torch
import torch.nn as nn
import hsl_embedding as hsl

blob = (np.random.RandomState(0).rand(20000) * 256).astype(np.uint8).tobytes()   # 20 KB of bytes
ids = torch.tensor(list(blob))
D = hsl.FEAT_DIM        # 27 (the exact base substrate)

nn_emb = nn.Embedding(256, D)        # smallest fair vocab = 256 byte values
hsl_emb = hsl.Embedding()

# ---- 1) capability matrix -------------------------------------------------------------
def yn(b): return "yes ✓" if b else "no  ✗"
print("capability                         nn.Embedding        hsl.Embedding")
print("-" * 70)
rows = [
    ("learnable parameters",              f"{256*D:,} (trained)", "0 (formula)"),
    ("needs a tokenizer / vocab",         "yes",                  "no (raw bytes)"),
    ("meaningful before any training",    "no  ✗",                "yes ✓"),
    ("one encoder across modalities",     "no  ✗ (per-modality)", "yes ✓"),
    ("handles any of 256 byte values",    "only if in vocab",     "yes ✓ (all)"),
    ("invertible (reconstruct input)",    "no  ✗",                "yes ✓ (lossless)"),
    ("interpretable dims",                "no  ✗ (opaque)",       "yes ✓ (Δ/Δ²/FFT/phase)"),
]
for a, b, c in rows:
    print(f"{a:34} {b:19} {c}")

# ---- 2) reconstruction: can you get the input back? -----------------------------------
restored = hsl.decode(hsl.encode(blob))
print(f"\nreconstruction error   HSL: {0 if restored == blob else 1}.0 (exact)   "
      f"nn.Embedding: N/A (a learned vector cannot be inverted to the input)")

# ---- 3) unseen value: nn.Embedding with a smaller vocab breaks; HSL never does --------
small = nn.Embedding(128, D)                       # vocab only covers bytes 0..127
try:
    small(torch.tensor([200]))                     # byte 200 -> out of range
    nn_ok = True
except Exception:
    nn_ok = False
hsl.embed(bytes([200]))                            # always fine
print(f"unseen byte (200) with vocab=128   nn.Embedding: {'ok' if nn_ok else 'IndexError ✗'}   HSL: ok ✓")

# ---- 4) throughput: a one-time input transform (NOT a like-for-like race) -------------
# nn.Embedding does a memory lookup; HSL *computes* an exact signal. These are different jobs,
# so this is not a fair head-to-head — HSL is a feature transform you run once and cache, the way
# you would any preprocessing. We report its throughput for context, not as a competition.
hsl_emb(blob)                                                      # warm up
t = time.perf_counter()
for _ in range(20):
    hsl_emb(blob)
mbps = 20 / ((time.perf_counter() - t) / 20) / 1024
print(f"\nHSL feature-extraction throughput: ~{mbps:.1f} MB/s (one-time transform; cache and reuse)")
print("nn.Embedding is a table lookup, not a signal computation — speed isn't a meaningful comparison.")

print("""
Takeaway
--------
  nn.Embedding  -> a fast learned lookup; needs a vocab + training; one per modality.
  hsl.Embedding -> zero params, no training, one substrate for every modality, exact & invertible,
                   interpretable channels. It computes a signal (so it's a one-time input transform,
                   not a lookup). Use HSL for the input layer; learn meaning on top.
""")
