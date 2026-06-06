"""
Unit tests for the BERT-pivot encoder, MLM masker, and Phase-3 task heads.

Scope is shape / smoke verification on CPU — we do not exercise training
schedules here.  Schedule end-to-end is covered by Phase-2 / Phase-3 smoke
runs in ``train.py``.
"""
import math
import torch
import pytest

from intervene_enc.dataset import EMRTokenizer
from intervene_enc.embedder import EMREmbedding
from intervene_enc.transformer import (
    InterveneEncoder, AdaLNBlock, BidirectionalSelfAttention,
    PerOutcomeAttnPool, TaskHeads, time_to_neighbour_targets,
)
from intervene_enc.utils import apply_mlm_mask, build_luts


# ─── fixtures ──────────────────────────────────────────────────────────── #
@pytest.fixture(scope="module")
def mini_tokenizer():
    """
    Hand-rolled tokenizer with one outcome ("DEATH_EVENT") so InterveneEncoder can
    build its outcome bookkeeping without invoking the full DataProcessor.
    """
    from intervene_enc.config.dataset_config import DEATH_TOKEN, RELEASE_TOKEN, ADMISSION_TOKEN

    specials = ["[PAD]", "[MASK]", "[NULL]",
                "[MASK_INTERVAL_START]", "[MASK_INTERVAL_END]"]
    extras = [ADMISSION_TOKEN, DEATH_TOKEN, RELEASE_TOKEN,
              "A_STATE_HIGH_START", "A_STATE_HIGH_END", "B_event"]
    toks = specials + extras

    token2id     = {t: i for i, t in enumerate(toks)}
    raw_toks     = specials + ["A", "B", DEATH_TOKEN, RELEASE_TOKEN, ADMISSION_TOKEN]
    rawconcept2id = {t: i for i, t in enumerate(raw_toks)}
    concept_toks = specials + ["A_STATE", "B_event", DEATH_TOKEN, RELEASE_TOKEN, ADMISSION_TOKEN]
    concept2id   = {t: i for i, t in enumerate(concept_toks)}
    value2id     = dict(concept2id)

    token_weights   = torch.ones(len(toks))
    outcome_weights = torch.ones(len(toks))                # n_neg/n_pos placeholder
    token_counts    = torch.full((len(toks),), 100, dtype=torch.long)

    # parent_raw lookup — every token maps to the [NULL] raw concept for
    # simplicity (the encoder averages parents and tolerates this).
    null_raw_id = rawconcept2id["[NULL]"]
    tokenid2parent_raw_ids = torch.full((len(toks), 1), null_raw_id, dtype=torch.long)
    parent_pad_len = 1

    tk = EMRTokenizer(
        token2id=token2id,
        rawconcept2id=rawconcept2id,
        concept2id=concept2id,
        value2id=value2id,
        special_tokens=specials,
        token_weights=token_weights,
        outcome_weights=outcome_weights,
        token_counts=token_counts,
        tokenid2parent_raw_ids=tokenid2parent_raw_ids,
        parent_pad_len=parent_pad_len,
        outcome_patient_ratios={DEATH_TOKEN: 0.1, RELEASE_TOKEN: 0.9},
    )
    return tk


@pytest.fixture(scope="module")
def mini_embedder(mini_tokenizer):
    return EMREmbedding(
        tokenizer=mini_tokenizer, ctx_dim=2, time2vec_dim=4, embed_dim=8, dropout=0.0,
    )


@pytest.fixture(scope="module")
def encoder_cfg():
    return {
        "time2vec_dim": 4,
        "embed_dim": 8,
        "n_head": 2,
        "n_layer": 2,
        "dropout": 0.0,
        "bias": True,
    }


def _dummy_batch(tokenizer, B=2, T=6, ctx_dim=2, P=1):
    """Build a synthetic batch matching ``collate_emr`` keys.

    Ids are sampled inside the *smallest* hierarchy size so the embedder's
    raw / concept / value / position embedding tables all accept them.
    """
    g = torch.Generator().manual_seed(0)
    V_raw  = len(tokenizer.rawconcept2id)
    V_con  = len(tokenizer.concept2id)
    V_val  = len(tokenizer.value2id)
    V_pos  = len(tokenizer.token2id)
    return {
        "parent_raw_ids": torch.randint(1, V_raw, (B, T, P), generator=g),
        "concept_ids":    torch.randint(1, V_con, (B, T),    generator=g),
        "value_ids":      torch.randint(1, V_val, (B, T),    generator=g),
        "position_ids":   torch.randint(1, V_pos, (B, T),    generator=g),
        "abs_ts":         torch.linspace(0.0, 0.5, T).repeat(B, 1),
        "context_vec":    torch.randn(B, ctx_dim, generator=g),
    }


# ─── component-level tests ─────────────────────────────────────────────── #
def test_bidirectional_self_attention_shape(encoder_cfg):
    layer = BidirectionalSelfAttention(encoder_cfg)
    x = torch.randn(2, 5, encoder_cfg["embed_dim"])
    abs_ts = torch.linspace(0, 0.5, 5).repeat(2, 1)
    pad = torch.ones(2, 5, dtype=torch.bool)
    y = layer(x, key_pad_mask=pad, abs_ts=abs_ts)
    assert y.shape == x.shape


def test_adaln_block_returns_same_shape(encoder_cfg):
    block = AdaLNBlock(encoder_cfg)
    B, T, D = 2, 5, encoder_cfg["embed_dim"]
    x = torch.randn(B, T, D)
    cond = torch.randn(B, D)
    out = block(x, cond, key_pad_mask=torch.ones(B, T, dtype=torch.bool),
                abs_ts=torch.linspace(0, 0.5, T).repeat(B, 1))
    assert out.shape == (B, T, D)


def test_per_outcome_attn_pool_shapes():
    pool = PerOutcomeAttnPool(d_model=8, n_outcomes=3, n_heads=2, dropout=0.0)
    h = torch.randn(2, 4, 8)
    kpm = torch.zeros(2, 4, dtype=torch.bool)  # no padding
    z = pool(h, kpm)
    assert z.shape == (2, 3, 8)


def test_task_heads_split_correctly():
    K = 4
    risk_idx = torch.tensor([0, 2, 3], dtype=torch.long)  # drops index 1 (RELEASE-like)
    time_idx = torch.arange(K, dtype=torch.long)
    heads = TaskHeads(d_model=8, n_outcomes=K, risk_idx_buf=risk_idx,
                      time_idx_buf=time_idx, hidden=16, dropout=0.0, n_heads=2)
    h = torch.randn(2, 5, 8)
    kpm = torch.zeros(2, 5, dtype=torch.bool)
    risk, time = heads(h, kpm)
    assert risk.shape == (2, 3)
    assert time.shape == (2, 4)
    # Time head emits non-negative values (softplus).
    assert (time >= 0).all()


def testtime_to_neighbour_targets_basic():
    # B=1, T=5: position 2 is masked, neighbours at t={0.0, 0.1, 0.3, 0.4} (h/336).
    abs_ts = torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4]])
    pad_mask = torch.ones(1, 5, dtype=torch.bool)
    mlm = torch.tensor([[False, False, True, False, False]])
    target, valid = time_to_neighbour_targets(abs_ts, pad_mask, mlm, max_hours=24.0)
    # Local gap at position 2 = min(0.2-0.1, 0.3-0.2) * 336 = 0.1 * 336 = 33.6h.
    # 33.6 / 24 = 1.4.
    assert valid[0, 2].item() is True
    assert target[0, 2].item() == pytest.approx(33.6 / 24.0, rel=1e-3)
    # Unmasked positions are filtered out via `valid`.
    assert valid[0, 0].item() is False


# ─── full-encoder smoke ────────────────────────────────────────────────── #
def test_encoder_forward_and_predict(mini_embedder, mini_tokenizer, encoder_cfg):
    model = InterveneEncoder(cfg=encoder_cfg, embedder=mini_embedder, use_checkpoint=False)
    model.eval()
    V = mini_embedder.decoder.out_features
    B, T = 2, 6
    batch = _dummy_batch(mini_tokenizer, B=B, T=T, ctx_dim=2)

    lm_logits, t_pos_pred, t_local_pred, pad_mask = model(**{
        k: batch[k] for k in ("parent_raw_ids", "concept_ids", "value_ids",
                              "position_ids", "abs_ts", "context_vec")
    })
    assert lm_logits.shape == (B, T, V)
    assert t_pos_pred.shape == (B, T)
    assert t_local_pred.shape == (B, T)
    assert pad_mask.shape == (B, T)

    # Phase-3 task heads
    model.attach_task_heads(hidden=16, n_heads=2)
    risk_logits, time_pred, hidden, pad_mask = model.predict(
        parent_raw_ids=batch["parent_raw_ids"],
        concept_ids=batch["concept_ids"], value_ids=batch["value_ids"],
        position_ids=batch["position_ids"], abs_ts=batch["abs_ts"],
        context_vec=batch["context_vec"],
    )
    K = model.num_outcomes
    K_risk = K - 1 if model._release_idx >= 0 else K
    assert risk_logits.shape == (B, K_risk)
    assert time_pred.shape == (B, K)
    assert hidden.shape == (B, T, encoder_cfg["embed_dim"])


def test_encoder_save_load_roundtrip(tmp_path, mini_embedder, encoder_cfg):
    model = InterveneEncoder(cfg=encoder_cfg, embedder=mini_embedder, use_checkpoint=False)
    model.attach_task_heads(hidden=16, n_heads=2)

    path = tmp_path / "encoder.pt"
    model.save(path, epoch=1, best_val=0.5)

    loaded, epoch, best_val, *_ = InterveneEncoder.load(
        path, embedder=mini_embedder, attach_task_heads=True,
    )
    assert epoch == 1
    assert best_val == pytest.approx(0.5)
    assert loaded.num_outcomes == model.num_outcomes


# ─── MLM masker tests ──────────────────────────────────────────────────── #
def test_apply_mlm_mask_atomic_intervals(mini_tokenizer):
    """Sampling an interval START should atomically mask its END partner."""
    tk = mini_tokenizer
    start_id = tk.token2id["A_STATE_HIGH_START"]
    end_id   = tk.token2id["A_STATE_HIGH_END"]

    # Build a batch where positions (0, 1) = START, END of the same interval,
    # and position (0, 2) = a stand-alone event.
    B, T, P = 1, 4, 1
    pos_ids = torch.tensor([[start_id, end_id, tk.token2id["B_event"], tk.pad_token_id]])
    batch = {
        "position_ids":   pos_ids.clone(),
        "parent_raw_ids": torch.zeros(B, T, P, dtype=torch.long),
        "concept_ids":    pos_ids.clone(),
        "value_ids":      pos_ids.clone(),
        "abs_ts":         torch.zeros(B, T),
    }
    luts = build_luts(tk)
    torch.manual_seed(0)
    # p=1.0 — every eligible position is sampled, but the partner-completion
    # rule means we never leave half an interval masked.
    out, target_ids, mlm_mask = apply_mlm_mask(
        batch, tk, forbid_ids=luts["forbid_mask_ids"], luts=luts, p=1.0,
    )

    # Target ids match the originals at masked positions.
    masked_positions = mlm_mask.nonzero(as_tuple=False)
    for b, t in masked_positions.tolist():
        assert target_ids[b, t].item() == pos_ids[b, t].item()

    # If either endpoint is masked, both must be masked.
    if mlm_mask[0, 0] or mlm_mask[0, 1]:
        assert mlm_mask[0, 0] and mlm_mask[0, 1]

    # Replacement tokens follow the hierarchical rule:
    if mlm_mask[0, 0]:
        assert out["position_ids"][0, 0].item() == tk.mask_interval_start_id
    if mlm_mask[0, 1]:
        assert out["position_ids"][0, 1].item() == tk.mask_interval_end_id
    if mlm_mask[0, 2]:
        # Plain B_event maps to the generic MASK.
        assert out["position_ids"][0, 2].item() == tk.mask_token_id


def test_apply_mlm_mask_hierarchical_mode(mini_tokenizer):
    """Hierarchical mode falls back to generic [MASK] when no family LUT
    is present on the tokenizer (mini_tokenizer was built without one),
    matching the documented backward-compat behaviour."""
    tk = mini_tokenizer
    # No family LUT on this hand-rolled fixture.
    assert getattr(tk, "tokenid2family_mask_id", None) is None

    pos_ids = torch.tensor([[tk.token2id["B_event"], tk.pad_token_id]])
    B, T, P = 1, 2, 1
    batch = {
        "position_ids":   pos_ids.clone(),
        "parent_raw_ids": torch.zeros(B, T, P, dtype=torch.long),
        "concept_ids":    pos_ids.clone(),
        "value_ids":      pos_ids.clone(),
        "abs_ts":         torch.zeros(B, T),
    }
    luts = build_luts(tk)
    out, _, mlm_mask = apply_mlm_mask(
        batch, tk, forbid_ids=luts["forbid_mask_ids"], luts=luts,
        p=1.0,
    )
    if mlm_mask[0, 0]:
        # Fallback to generic [MASK].
        assert out["position_ids"][0, 0].item() == tk.mask_token_id


def test_apply_mlm_mask_skips_forbid_ids(mini_tokenizer):
    tk = mini_tokenizer
    from intervene_enc.config.dataset_config import DEATH_TOKEN

    # Build a single-sequence batch containing only a forbid token (DEATH).
    death_id = tk.token2id[DEATH_TOKEN]
    pos_ids = torch.tensor([[death_id, death_id, tk.pad_token_id]])
    B, T, P = 1, 3, 1
    batch = {
        "position_ids":   pos_ids.clone(),
        "parent_raw_ids": torch.zeros(B, T, P, dtype=torch.long),
        "concept_ids":    pos_ids.clone(),
        "value_ids":      pos_ids.clone(),
        "abs_ts":         torch.zeros(B, T),
    }
    luts = build_luts(tk)
    out, _, mlm_mask = apply_mlm_mask(
        batch, tk, forbid_ids=luts["forbid_mask_ids"], luts=luts, p=1.0,
    )
    # DEATH must never be masked.
    assert mlm_mask.sum().item() == 0
    assert torch.equal(out["position_ids"], pos_ids)
