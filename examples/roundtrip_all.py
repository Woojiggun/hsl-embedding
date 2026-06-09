"""One encoder, any modality, exact reconstruction.

HSL reads text / image / audio / video as the SAME thing — bytes — and its substrate is lossless,
so the original comes back *exactly*. Here we embed each modality and rebuild it straight from the
embedding's Δ (change-rate) channel. No tokenizer, no per-modality code, no information lost.
(Self-contained: samples are synthesized with numpy; no extra dependencies.)
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")   # so the ✓/✗ marks print on any console (e.g. Windows cp949)
except Exception: pass
import numpy as np
import torch
import hsl_embedding as hsl


def restore_from_embedding(feats: torch.Tensor) -> bytes:
    """Rebuild the original bytes straight from the embedding's Δ channel (dxor = first 8 dims).
    Δ is the PER-BYTE change-rate from the anchor 0 — integrate each byte independently (cumulative-XOR per row)."""
    dxor = feats[:, 0:8].round().to(torch.uint8).numpy()                   # [L, 8] per-byte Δ
    bits = np.bitwise_xor.accumulate(dxor, axis=1)                         # integrate each byte from anchor 0
    return np.packbits(bits, axis=1).tobytes()


# --- synthesize one real sample per modality (as its natural raw bytes) ---------------
text = "변화율은 모든 모달리티의 공통 언어다. Everything is a fluctuation between 0 and 1.".encode("utf-8")
image = (np.add.outer(np.arange(32), np.arange(32)) % 256).astype(np.uint8)         # 32x32 gradient
image = np.stack([image, image[::-1], image.T], -1).astype(np.uint8)               # 32x32x3 RGB
audio = (np.sin(np.linspace(0, 50 * np.pi, 4000)) * 30000).astype(np.int16)        # 4000-sample tone
video = (np.random.RandomState(0).rand(6, 16, 16, 3) * 255).astype(np.uint8)       # 6 frames 16x16 RGB

samples = {
    "text  (utf-8)":  (text, text),
    "image (RGB u8)":  (image.tobytes(), image),
    "audio (PCM i16)": (audio.tobytes(), audio),
    "video (6 frames)":(video.tobytes(), video),
}

print(f"{'modality':18} {'bytes':>8} {'feat shape':>14}   reconstruction")
print("-" * 64)
for name, (raw, original) in samples.items():
    feats, _ = hsl.embed(raw)                                   # ONE call, any modality -> [L, 27]
    restored = restore_from_embedding(feats)                    # rebuild straight from the embedding
    exact = restored == raw
    # round-trip back into the modality's native array, bit-for-bit
    if isinstance(original, np.ndarray):
        rebuilt = np.frombuffer(restored, dtype=original.dtype).reshape(original.shape)
        exact = exact and np.array_equal(rebuilt, original)
    print(f"{name:18} {len(raw):>8} {str(tuple(feats.shape)):>14}   {'EXACT ✓' if exact else 'MISMATCH ✗'}")

print("\nOne modality-agnostic encoder. Lossless by construction — embed, then restore the original exactly.")
