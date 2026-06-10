"""HSL invariants — run with `pytest`."""
import numpy as np
import torch
import hsl_embedding as hsl


SAMPLES = [b"", b"\x00", b"A", "강아지 dog 🐕 0101".encode(), bytes(range(256)), b"hello world " * 30]


def test_lossless_roundtrip():
    for d in SAMPLES:
        if not d:
            continue
        assert hsl.decode(hsl.encode(d)) == d


def test_shapes_and_dims():
    feats, phase = hsl.embed(b"hello")
    assert feats.shape == (5, hsl.FEAT_DIM)
    assert phase.shape == (5,)
    assert feats.dtype == torch.float32


def test_default_is_27d_exact_base():
    feats, _ = hsl.embed(b"hello")              # default = the exact base, no raw bits
    assert feats.shape[1] == hsl.FEAT_DIM == 27


def test_lean_is_full_minus_bits():
    for d in SAMPLES:
        full, _ = hsl.embed(d or b"\x00", include_bits=True)
        lean, _ = hsl.embed(d or b"\x00", include_bits=False)
        assert full.shape[1] == hsl.FEAT_DIM_FULL == 35
        assert lean.shape[1] == hsl.FEAT_DIM == 27
        assert torch.allclose(lean, full[:, 8:], atol=1e-6)   # the dropped 8 are the raw bits


def test_delta_is_per_byte_position_and_independent():
    # Δ is the POSITION measured PER BYTE from the symbolic anchor 0 — byte-INDEPENDENT (no cross-byte coupling)
    data = b"redundancy?"
    fr = hsl.encode(data)
    db = fr.delta.reshape(len(data), 8)
    bits = np.bitwise_xor.accumulate(db, axis=1)              # integrate each byte's Δ from anchor 0
    assert np.packbits(bits, axis=1).tobytes() == data        # per-byte reconstruction is exact
    # byte-independence: a byte's Δ does NOT depend on the previous byte (Δ[byte i, bit0] = bit0 ⊕ anchor)
    fr2 = hsl.encode(b"\xff\x00")
    assert int(fr2.delta.reshape(2, 8)[1, 0]) == 0            # byte1 bit0 = 0 ⊕ 0, NOT 0 ⊕ bit7(byte0)


def test_delta2_is_cross_byte_flow():
    # Δ² (flow/momentum) = how the per-byte Δ changes BETWEEN bytes: Δ²[i] = Δ[i] ⊕ Δ[i-1] (anchor for byte 0)
    fr = hsl.encode(b"flow!")
    d, d2 = fr.delta.reshape(5, 8), fr.delta2.reshape(5, 8)
    for i in range(5):
        prev = d[i - 1] if i > 0 else np.zeros(8, np.uint8)
        assert (d2[i] == (d[i] ^ prev)).all()


def test_fourier_is_exact_and_invertible():
    # the ONE base Fourier is the exact complex rfft: all 256 bytes distinct, losslessly invertible
    lut = hsl._FOURIER_LUT                                   # [256, 8] real[0:5] + imag[1:4]
    assert lut.shape == (256, 8)
    assert np.unique(np.round(lut, 5), axis=0).shape[0] == 256   # no collisions (lossy ratio had 27)
    re = lut[:, :5]
    im = np.zeros((256, 5), dtype=np.float32); im[:, 1:4] = lut[:, 5:8]
    bits = np.fft.irfft(re + 1j * im, n=8, axis=1)
    recon = (bits > 0.5).astype(np.uint8)
    orig = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1)
    assert (recon == orig).all()                            # exact byte recovery from the Fourier feature


def test_no_lossy_or_dead_symbols():
    # the lossy abs-ratio Fourier and the dead HF helper are gone (no approximation ships)
    assert not hasattr(hsl, "FEAT_DIM_EXACT")
    assert not hasattr(hsl, "_FOURIER_EXACT_LUT")
    assert not hasattr(hsl, "_hf_energy")
    # v0.4: the legacy CHAIN-mode helpers are gone too — only the per-byte anchor rule ships
    assert not hasattr(hsl, "_xor_delta")
    assert not hasattr(hsl, "_integrate")
    assert not hasattr(hsl, "_bits_to_bytes")


def test_every_byte_departs_from_the_same_virtual_origin():
    # THE ANCHOR RULE: each byte's Δ is measured from the SAME virtual origin 0 — never from the
    # previous byte's last bit. So a byte's Δ is identical alone, after ANY prefix, at ANY position.
    lone = {v: hsl.encode(bytes([v])).delta.tolist() for v in range(256)}     # Δ of every value alone
    for prefix in (b"\x00", b"\xff", b"\xaa\x55", b"prefix \xf0\x0f"):
        for v in range(256):                                                  # exhaustive over all byte values
            d = hsl.encode(prefix + bytes([v])).delta.reshape(-1, 8)
            assert d[-1].tolist() == lone[v]          # Δ does not depend on what came before
    # position-independence at distance: value 200 at positions 0 and 9999 has the very same Δ row
    far = hsl.encode(bytes([200]) + b"\x37" * 9998 + bytes([200])).delta.reshape(-1, 8)
    assert far[0].tolist() == far[-1].tolist() == lone[200]
    # and the embedding's Δ channel (first 8 dims) obeys the same rule
    f1, _ = hsl.embed(bytes([200]))
    f2, _ = hsl.embed(b"\x37" * 9999 + bytes([200]))
    assert torch.equal(f1[0, :8], f2[-1, :8])


def test_dxor_is_the_gray_code():
    # identity: Δ(v) ≡ binary-reflected Gray code v ^ (v >> 1) — a consequence of the anchor rule
    v = np.arange(256, dtype=np.uint8)
    gray_bits = np.unpackbits((v ^ (v >> 1))[:, None], axis=1)
    dx = hsl.encode(bytes(range(256))).delta.reshape(256, 8)
    assert (dx == gray_bits).all()
    adj = (gray_bits[1:] ^ gray_bits[:-1]).sum(1)
    assert adj.min() == 1 and adj.max() == 1          # ±1 value change moves exactly ONE Δ coordinate


def test_bounded_finite_and_no_drift():
    # no approximation, no divergence, no decay-to-zero: every channel is bounded and finite, and
    # features at position 100000 are exactly what they are at position 2 (no recurrence anywhere)
    rng = np.random.RandomState(7)
    long_random = rng.randint(0, 256, 65536, dtype=np.uint8).tobytes()
    for include_bits in (False, True):
        for momentum_phase in (False, True):
            for data in (bytes(range(256)) * 8, long_random, b"\x00" * 4096, b"\xff" * 4096):
                f, p = hsl.embed(data, include_bits=include_bits, momentum_phase=momentum_phase)
                assert torch.isfinite(f).all() and torch.isfinite(p).all()
                assert float(f.abs().max()) <= 8.0     # global bound (fft_re0 = bit count ≤ 8)
    names = hsl.feat_names()
    f, _ = hsl.embed(long_random)
    assert float(f[:, names.index("boundary")].min()) >= 0.0     # boundary never goes negative
    assert float(f[:, names.index("fft_re0")].min()) >= 0.0      # DC bin = bit count ≥ 0
    g, _ = hsl.embed(b"\x42" * 100000)
    assert torch.equal(g[2], g[50000]) and torch.equal(g[2], g[-1])   # constant stream → exactly constant
    assert float(g[-1].abs().max()) > 0.0                             # ... and it never collapses to zero


def test_boundary_is_exact_at_any_length():
    # v0.3's float32 running sum rounded the boundary above ~1.4 MB; v0.4 is closed-form.
    # Check bit-exact agreement with a float64 brute-force windowed mean.
    rng = np.random.RandomState(3)
    data = rng.randint(0, 256, 200_000, dtype=np.uint8).tobytes()
    fr = hsl.encode(data)
    e = fr.delta.astype(np.float64) + hsl.BOUNDARY_D2_WEIGHT * fr.delta2.astype(np.float64)
    R = hsl.BOUNDARY_WINDOW_RADIUS
    starts = np.arange(len(data)) * 8
    csum = np.concatenate([[0.0], np.cumsum(e)])                 # float64 — exact reference
    lo = np.maximum(0, starts - R)
    hi = np.minimum(e.size, starts + R + 1)
    ref = ((csum[hi] - csum[lo]) / (hi - lo)).astype(np.float32)
    assert np.array_equal(fr.byte_boundary_score, ref)


def test_tensor_path_is_bit_identical_to_bytes_path():
    rng = np.random.RandomState(11)
    raw = rng.randint(0, 256, 4096, dtype=np.uint8).tobytes()
    for include_bits in (False, True):
        for momentum_phase in (False, True):
            emb = hsl.Embedding(include_bits=include_bits, momentum_phase=momentum_phase)
            f_b, p_b = emb(raw, return_phase=True)
            f_t, p_t = emb(torch.tensor(list(raw), dtype=torch.uint8), return_phase=True)
            assert torch.equal(f_b, f_t) and torch.equal(p_b, p_t)
    emb = hsl.Embedding()
    batch = torch.randint(0, 256, (3, 257))                      # long ids, batched [B, L]
    fb = emb(batch)
    assert fb.shape == (3, 257, 27)
    for i in range(3):
        assert torch.equal(fb[i], emb(bytes(batch[i].tolist())))
    assert emb(torch.zeros(2, 0, dtype=torch.long)).shape == (2, 0, 27)   # empty stays empty (tensor path)
    assert len(emb.state_dict()) == 0                            # buffers non-persistent → checkpoint-compatible


def test_momentum_phase_carries_velocity_without_extra_dims():
    data = bytes(range(256))
    f0, _ = hsl.embed(data)
    f1, _ = hsl.embed(data, momentum_phase=True)
    assert f0.shape == f1.shape                              # no extra dims
    ph0, ph1 = f0[:, -2:].numpy(), f1[:, -2:].numpy()
    ang0 = np.arctan2(ph0[:, 1], ph0[:, 0]); ang1 = np.arctan2(ph1[:, 1], ph1[:, 0])
    assert np.abs(ang0 - ang1).max() < 1e-5                  # angle (=value, =affect) preserved exactly
    mag = np.sqrt((ph1 ** 2).sum(1))
    dv = np.abs(np.diff(np.frombuffer(data, np.uint8).astype(np.int16), prepend=0)).astype(np.float32)
    assert np.corrcoef(mag, dv)[0, 1] > 0.99                 # magnitude carries |Δbyte| (momentum)


def test_module_and_pack():
    emb = hsl.Embedding()                       # default = 27-D exact base
    assert emb.out_dim == 27
    assert emb(b"test").shape == (4, 27)
    assert hsl.Embedding(include_bits=True).out_dim == 35
    feats, phase, mask = emb.pack([b"a", b"abcdef"], max_len=8)
    assert feats.shape == (2, 8, 27)
    assert mask.sum(1).tolist() == [1.0, 6.0]


def test_empty_input_is_safe():
    feats, _ = hsl.embed(b"")
    assert feats.shape[0] >= 1            # empty -> treated as a single zero byte, never crashes


def test_feat_names():
    assert len(hsl.feat_names(True)) == 35
    assert len(hsl.feat_names(False)) == 27
    assert hsl.feat_names(False)[0] == "dxor0"   # change-rate is the first channel of the base substrate
    assert "fft_re0" in hsl.feat_names(False) and "fft_im3" in hsl.feat_names(False)
