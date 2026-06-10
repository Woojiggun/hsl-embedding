"""Ablation toolkit invariants — the controls must isolate EXACTLY the value geometry."""
import numpy as np
import pytest
import torch
import hsl_embedding as hsl
from hsl_embedding import ablation as ab


def test_feature_groups_cover_everything_exactly_once():
    fg = ab.feature_groups()
    cols = sum((fg[g] for g in ("dxor", "d2xor", "boundary", "fourier", "phase")), [])
    assert sorted(cols) == list(range(27))
    assert fg["value"] == fg["dxor"] + fg["fourier"] + fg["phase"] and len(fg["value"]) == ab.VALUE_DIM == 18
    assert fg["context"] == fg["d2xor"] + fg["boundary"] and len(fg["context"]) == 9
    full = ab.feature_groups(include_bits=True)
    assert sorted(full["value"] + full["context"]) == list(range(35)) and len(full["value"]) == 26


def test_channel_indices_validation():
    assert ab.channel_indices("phase") == [25, 26]
    assert ab.channel_indices(("dxor", "phase")) == list(range(8)) + [25, 26]
    with pytest.raises(KeyError):
        ab.channel_indices(("nope",))


def test_select_channels_bits_vs_dxor_minimal_pair():
    # the sharpest cheap A/B: same information, same dims, same {0,1} scale — geometry only
    f, _ = hsl.embed(bytes(range(256)), include_bits=True)
    bits = ab.select_channels(f, ("bits",), include_bits=True)
    dxor = ab.select_channels(f, ("dxor",), include_bits=True)
    assert bits.shape == dxor.shape == (256, 8)
    v = np.arange(256, dtype=np.uint8)
    assert np.array_equal(bits.numpy(), np.unpackbits(v[:, None], axis=1).astype(np.float32))
    assert np.array_equal(dxor.numpy(), np.unpackbits((v ^ (v >> 1))[:, None], axis=1).astype(np.float32))
    # adjacent byte values: Δ moves exactly 1 coordinate, raw bits up to all 8
    assert int((dxor[1:] - dxor[:-1]).abs().sum(1).max()) == 1
    assert int((bits[1:] - bits[:-1]).abs().sum(1).max()) == 8


def test_value_lut_is_what_embedding_uses():
    lut = ab.value_lut()
    assert lut.shape == (256, ab.VALUE_DIM)
    ids = torch.arange(256)
    ref = hsl.Embedding()(ids)
    vi = ab.feature_groups()["value"]
    assert np.array_equal(ref[:, vi].numpy(), lut)


def test_control_hsl_is_the_exact_reference():
    ids = torch.randint(0, 256, (2, 333))
    out, ph = ab.ControlEmbedding("hsl")(ids, return_phase=True)
    ref, ph_ref = hsl.Embedding()(ids, return_phase=True)
    assert torch.equal(out, ref) and torch.equal(ph, ph_ref)


def test_controls_share_exact_context_and_swap_only_value():
    ids = torch.randint(0, 256, (2, 257))
    ref = hsl.Embedding()(ids)
    fg = ab.feature_groups()
    for kind in ("learned", "random", "permuted"):
        out = ab.ControlEmbedding(kind, seed=1)(ids)
        assert out.shape == ref.shape == (2, 257, 27)
        assert torch.equal(out[..., fg["context"]], ref[..., fg["context"]])   # context identical
        assert not torch.equal(out[..., fg["value"]], ref[..., fg["value"]])   # geometry swapped
        assert torch.isfinite(out).all()


def test_permuted_keeps_marginals_exactly():
    lut_p = ab.ControlEmbedding("permuted", seed=3)._value_lut.numpy()
    lut_h = ab.value_lut()
    assert np.array_equal(np.sort(lut_p, axis=0), np.sort(lut_h, axis=0))   # per-column multisets equal
    assert not np.array_equal(lut_p, lut_h)                                 # ... but geometry destroyed


def test_random_is_injective_seeded_and_moment_matched():
    a = ab.ControlEmbedding("random", seed=0)._value_lut.numpy()
    b = ab.ControlEmbedding("random", seed=0)._value_lut.numpy()
    c = ab.ControlEmbedding("random", seed=1)._value_lut.numpy()
    assert np.array_equal(a, b) and not np.array_equal(a, c)                # seed-reproducible
    assert np.unique(a.round(5), axis=0).shape[0] == 256                    # information-preserving
    h = ab.value_lut()
    assert np.allclose(a.mean(0), h.mean(0), atol=1e-3)                     # per-channel moments match
    assert np.allclose(a.std(0), h.std(0), atol=1e-3)


def test_learned_is_trainable_and_seeded_others_are_zero_param():
    m = ab.ControlEmbedding("learned", seed=0)
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == 256 * 18
    m2 = ab.ControlEmbedding("learned", seed=0)
    m3 = ab.ControlEmbedding("learned", seed=1)
    assert torch.equal(m.value.weight, m2.value.weight)
    assert not torch.equal(m.value.weight, m3.value.weight)
    loss = m(torch.randint(0, 256, (4, 64))).pow(2).mean()
    loss.backward()
    assert m.value.weight.grad is not None and torch.isfinite(m.value.weight.grad).all()
    for kind in ("hsl", "random", "permuted"):
        assert sum(p.numel() for p in ab.ControlEmbedding(kind).parameters()) == 0


def test_control_rejects_bytes_and_bad_kind():
    with pytest.raises(TypeError):
        ab.ControlEmbedding("hsl")(b"abc")
    with pytest.raises(ValueError):
        ab.ControlEmbedding("nope")
