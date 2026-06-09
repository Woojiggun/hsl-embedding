"""hsl.Embedding vs torch.nn.Embedding — *when to use which* (not a performance comparison).

nn.Embedding  : token id -> learned vector. Needs a tokenizer + vocab + training. One per modality.
hsl.Embedding : raw bytes -> exact signal features. No tokenizer, no params, works across modalities.
They compose: stack nn layers ON TOP of HSL features.
"""
import torch
import torch.nn as nn
import hsl_embedding as hsl

text = "강아지".encode("utf-8")
image_bytes = bytes([0, 0, 5, 250, 255, 250, 5, 0])     # a tiny 1-D "edge"
audio_bytes = bytes([128, 130, 126, 160, 96, 200, 56])  # a tiny "transient"

# --- nn.Embedding: you must first define a vocab and tokenize -------------------------
vocab_size = 256
nn_emb = nn.Embedding(vocab_size, 32)                    # 256*32 LEARNED params, random until trained
ids = torch.tensor(list(text))                           # you had to choose a tokenization (here: bytes)
print("nn.Embedding:", tuple(nn_emb(ids).shape), "(learned, random until trained; needs a vocab)")

# --- hsl.Embedding: bytes straight in, meaningful from day one -------------------------
hsl_emb = hsl.Embedding()                                # 0 params, deterministic
for name, b in [("text", text), ("image", image_bytes), ("audio", audio_bytes)]:
    feats = hsl_emb(b)
    print(f"hsl.Embedding({name:5}): {tuple(feats.shape)} - same call, any modality")

# --- what HSL gives that a single learned vector blurs together -----------------------
names = hsl.feat_names()        # default 27-D exact base channel names
feats = hsl_emb(audio_bytes)
print("\naudio 'transient' - interpretable channels at byte 3:")
for ch in ("dxor0", "d2xor0", "boundary", "fft_re2", "fft_im2", "phase_sin"):
    print(f"  {ch:14} = {feats[3, names.index(ch)]:+.3f}")

# --- composing: HSL features -> your learned head -------------------------------------
head = nn.Sequential(nn.Linear(hsl_emb.out_dim, 64), nn.GELU(), nn.Linear(64, 16))
out = head(hsl_emb(text))                                # learn meaning on top of exact signal
print("\nHSL -> learned head:", tuple(out.shape))

print("""
Rule of thumb
-------------
  nn.Embedding  -> fixed vocab, lots of data, you want learned semantics.
  hsl.Embedding -> tokenizer-free, cross-modal, structure/change-aware input, exact & invertible.
  Best together -> HSL for the input substrate, nn layers for the meaning.
""")
