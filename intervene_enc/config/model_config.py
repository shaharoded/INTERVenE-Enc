import os

# Get project root (2 levels up from config/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Checkpoint paths
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'checkpoints')
PHASE1_CHECKPOINT = os.path.join(CHECKPOINT_PATH, 'phase1', 'ckpt_best.pt')
PHASE2_CHECKPOINT = os.path.join(CHECKPOINT_PATH, 'phase2', 'ckpt_best.pt')
PHASE3_CHECKPOINT = os.path.join(CHECKPOINT_PATH, 'phase3', 'ckpt_best.pt')

# Global RNG seed — applied via utils.set_seed() in every model constructor and
# training-phase entry point so runs are reproducible.
SEED = 42

MODEL_CONFIG = {
      "time2vec_dim": 32,
      "embed_dim": 128,   # M-128 starting point (head_dim=64, n_head=2); grow during the size sweep.
      "n_head": 2,
      "n_layer": 4,
      "dropout": 0.1,
      "bias": True,
    }

TRAINING_SETTINGS = {
    "phase1_n_epochs": 100,
    "phase2_n_epochs": 100,
    "phase3_n_epochs": 100,
    "sample": 10000,  # Stage A/B/C decision sweep convention (Phase 1/2/3). D/E use None.

    # Phase-2 optimizer LR warmup (OneCycleLR pct_start).
    # This controls optimizer step size ramp-up, not auxiliary-loss lambda warmup.
    "lr_warmup_epochs": 5,
    "early-stop-patience": 10,
    "early-stop-min-delta-rel": 1e-3,  # relative improvement threshold (0.1%)

    "phase1_learning_rate":       3e-4,
    "phase2_learning_rate":       3e-4,
    "phase3_learning_rate":       1e-4,
    "phase3_backbone_lr_factor":  0.1,   # was 0.01 (near-frozen); unfreeze so P3 backbone adapts  # backbone LR = phase3_lr * factor (1e-6); 0.0 = fully frozen
    "phase3_weight_decay":        1e-3,  # weight decay for outcome_head in P3 (matches backbone)
    "weight_decay":               1e-3,

    "batch_size": 16, # Number of patients processed concurrently (effective batch=64 via grad accumulation)
    "grad_accumulation_steps": 4, # Accumulate gradients over N steps before optimizer.step(), memory-friendly way to get effective batch size > GPU batch size.
    "phase1_bce_window_hours": 3.0,
    # Soft-kernel horizon for the Phase-2 LM-head BCE. The kernel decay constant
    # tau is learnable per token class (model.log_tau_lm); this value is both the
    # init for terminal tokens and the hard outer horizon beyond which the kernel
    # contribution is zero.
    "phase2_terminal_bce_window_hours": 168.0,

    # Phase-1 auxiliary scheduler.
    # Main loss = per-window outcome BCE. Single stage: the `dt` (Δt MSE) aux
    # activates after `main_only_epochs` epochs of main-loss-only training. The
    # lambda max is calibrated ONCE from train losses at the first active epoch
    # (λ = aux_fraction_cap × tr_main / tr_aux) and then kept fixed. The
    # weighted contribution is capped to `aux_fraction_caps[name]` of tr_main.
    "phase1_scheduler": {
        "main_only_epochs": 3,     # epochs of BCE-only training before dt activates
        "aux_fraction_caps": {
            "dt":  0.40,           # Δt MSE capped at 40% of BCE at calibration
        },
        "order": [["dt"]],         # single stage with one aux
        "ramp_epochs": {
            "dt":  0,              # no ramp; jump straight to λ_max once unlocked
        },
    },

    # Phase-2 auxiliary scheduler.
    # Main loss = MLM cross-entropy. Single stage: `t_pos` (time-since-admission
    # MSE at every non-pad position) and `t_local` (time-to-neighbour MSE at
    # masked positions only) activate together after `main_only_epochs` of
    # MLM-only training. Lambda max is calibrated ONCE from training losses at
    # the first active epoch (λ = aux_fraction_cap × tr_main / tr_aux) and then
    # kept fixed. `main_only_epochs` doubles as LRScheduleController's
    # OneCycleLR pct_start anchor.
    "phase2_scheduler": {
        "main_only_epochs": 4,     # epochs of MLM-only training before t_pos/t_local activate
        "aux_fraction_caps": {
            "t_pos":   0.40,       # time-since-admission MSE capped at 40% of MLM CE
            "t_local": 0.30,       # time-to-neighbour MSE capped at 30% of MLM CE
        },
        "order": [["t_pos", "t_local"]],   # single stage, both auxes unlock together
        "ramp_epochs": {
            "t_pos":   0,          # no ramp; jump to λ_max at unlock
            "t_local": 0,
        },
        # Per-aux λ_max ceiling (overrides the global hard clamp of 10). The
        # global clamp is a safety against a tiny-magnitude aux getting a runaway
        # λ from the fraction rule (λ = fraction_cap × MLM/aux). Both time auxes
        # are now uncapped so the fraction-cap rule alone governs their weight:
        #   t_local → λ≈61 (0.30 share);  t_pos → λ≈41 (0.40 share).
        # (Phase-1 `dt` keeps the default-10 safety; it's tiny and never binds.)
        "max_lambda": {"t_local": 400.0, "t_pos": 400.0},
    },

    # Outcome head — time-decayed soft labels.
    # For each position t the target for outcome k is:
    # sum_s { exp(-dt(t,s) / tau_k) * 1[token_s == outcome_k] }.clamp(0, 1)
    # tau_k is a per-outcome learnable parameter (model.outcome_log_tau), initialised
    # at log(12 / 336). outcome_horizon_hours hard-zeros any contribution beyond that
    # horizon (kept in sync with the eval window family).
    "outcome_horizon_hours": 48.0,

    # P4 — patient-level attention pool head (Phase 3 only).
    # Per-outcome learnable query embeddings cross-attend over the backbone's
    # final hidden states to produce one pooled feature per (patient, outcome).
    # A scalar projection turns each pooled feature into a patient-level
    # logit; BCE against patient_label[b, k] = "outcome k appears anywhere in
    # the non-pad GT trajectory". λ_pool calibrated once at the end of
    # Phase-3 epoch 1, capped at this fraction of raw outcome BCE — same
    # regime as ranking. Pool head trains at Phase-3 head LR; its gradient
    # flows through the hidden-state stash into the backbone at
    # backbone_lr_factor=0.01, protecting the outcome head from patient-level
    # coarseness.
    "phase3_pool_fraction_cap": 0.05,   # I2 P4-tight: lowered 0.20 -> 0.05

    # --- BERT-style Phase-2 settings ---
    # MLM ratio applied per batch (BERT-default 15%; ramped from 0 over main_only_epochs).
    "phase2_mlm_ratio": 0.15,
    # Atomic-interval mask replacement: three generic tokens
    # ([MASK], [MASK_INTERVAL_START/END]); hierarchical/HEART-family masking
    # was tested (i1-hier) and DISCARDED — removed from the codebase.
    # Phase-2 aux-loss fraction caps live inside `phase2_scheduler`
    # ("aux_fraction_caps": {"t_pos": …, "t_local": …}) — see above.

    # --- Phase-3 settings ---
    # phase3_time_lambda — weight of the per-outcome z-MSE time loss vs the
    # multi-label BCE risk loss. 0.5 keeps the time head on near-equal
    # footing with risk (matches STraTS / GRU-D's LoS-loss weight).
    "phase3_time_lambda": 0.1,   # was 0.5; time z-MSE dominated total/early-stop ~3:1, starved risk training
    "phase3_head_hidden": 256,
    # phase3_focal_gamma — focal-BCE on the risk head: loss is multiplied
    # by (1 − p_t)^γ, downweighting easy / confident examples so rare
    # positives (DEATH, SEVERE_HYPOGLYCEMIA) drive a larger share of the
    # gradient. 0 = plain BCE. 2.0 is the Lin et al. default.
    "phase3_focal_gamma": 0.0,   # was 2.0; focal squashed risk logits to std~0.33 -> low AUPRC
    # phase3_cbm_p — Curriculum-by-Masking input-token replacement during
    # Phase-3 training. Masks `p` fraction of non-special positions with
    # [MASK] (BERT-style), targets/labels computed from the un-noised
    # batch. Forces the model to use multi-source signal, helps rare
    # outcomes. 0 disables.
    "phase3_cbm_p": 0.0,    # was 0.25; remove input noise for the clean-baseline retry
    # phase3_pool_dropout — dropout inside the Phase-3 attention pool +
    # shared MLP. ``None`` inherits the backbone's MODEL_CONFIG["dropout"].
    # 0.20 matches STraTS's attention_dropout.
    "phase3_pool_dropout": 0.20,
    # phase3_pos_weight_mode — outcome weighting on the risk (focal-)BCE
    # loss. "inv_prev" (default): per-outcome pos_weight = n_neg / n_pos
    # from the tokenizer. "uniform": pos_weight = 1 for every outcome.
    # The Exp-A ablation flips this to verify class-imbalance weighting
    # is still load-bearing once focal-BCE is layered on top.
    "phase3_pos_weight_mode": "uniform",  # was inv_prev; plain-BCE clean baseline
}
