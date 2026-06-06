# Event Prediction in EMRs

This repository implements a three-phase deep learning pipeline for modeling longitudinal Electronic Medical Records (EMRs). The architecture combines temporal embeddings, patient context, and a **bidirectional Transformer encoder** trained with masked language modelling to read per-outcome complication risk and time-to-event predictions from learnable outcome queries.

> **BERT-pivot, 2026-06.** The original autoregressive (GPT-style) backbone in this repo was replaced with a BERT-style encoder. The AR results that motivated the pivot still appear in the thesis but live in a separate repository. See `PROPOSAL_BERT_PIVOT.md` for the full motivation, and `EXPERIMENTS.md` for the architectural-search knobs that are wired in but not yet swept.

<img src="images\Model Sceme.png" width="100%">

This repo is part of an unpublished thesis and will be finalized post-submission. **Please do not reuse without permission**.

The results shown here (in `evaluation.ipynb`) are on random data, as my research dataset is private. This model will be used on actual EMR data, stored in a closed environment. For that, it is organized as a package that can be installed:

```bash
transform-emr/
│
├── intervene_enc/                     # Core Python package
│   ├── config/                        # Configuration modules
│   │   ├── __init__.py
│   │   ├── tak-repo-portable.json     # TAKRepository object from Mediator (see related project)
│   │   ├── dataset_config.py
│   │   └── model_config.py
│   ├── __init__.py                    
│   ├── dataset.py                     # Dataset, DataPreprocess and Tokenizer
│   ├── embedder.py                    # Embedding model (EMREmbedding) + training
│   ├── transformer.py                 # Bidirectional encoder (InterveneEncoder) + TaskHeads + Phase-2/3 training
│   ├── inference.py                   # Single-pass inference (encode -> pool -> risk/time)
│   ├── schedulers.py                  # Utility module for training schedulers (LR & Aux tasks)
│   └── utils.py                       # Utility functions (plots + penalties + masks + MLM masker)
├── data/                              # External data folder (for synthetic or real EMR)
│   ├── generate_synthetic_data.ipynb  # A notebook that generates synthetic data similar in structure to mediator's output (for tests)
│   ├── source/                        # Notebook will point here and auto-generate the train-test splits
│   ├── train/
│   └── test/
├── unittests/                         # Unit and integration tests (dataset / model / utils)
├── evaluation.ipynb                   # Self-contained eval notebook — patient-level AUC/F1, peak MAE, length-of-stay, calibration & trajectory plots
├── README.md                         
├── .gitignore
├── requirements.txt
├── LICENCE
├── CITATION.cff
├── setup.py
└── pyproject.toml
```

As noted, this model feeds of the output of the [Mediator](https://github.com/shaharoded/Mediator) temporal abstraction engine. It can work with any temporal-interval dataset, but note that the embedding has knowledge-base component, so a `tak-repo-portable.json` like object is mandatory.

---

## 🛠️ Installation

Install the project as an editable package from the **root** directory:

```bash
pip install -e .

# Ensure your working directory is properly set to the root repo of this project
# Be sure to set the path in your local env properly.
```

---

## 🚀 Usage

### 1. Prepare Dataset and Update Config

```python
import pandas as pd
from intervene_enc.dataset import EMRDataset
from intervene_enc.config.dataset_config import *
from intervene_enc.config.model_config import *

# Load data (verify you paths are properly defined)
temporal_df = pd.read_csv(TRAIN_TEMPORAL_DATA_FILE, low_memory=False)
ctx_df = pd.read_csv(TRAIN_CTX_DATA_FILE)

print(f"[Pre-processing]: Building tokenizer...")
processor = DataProcessor(temporal_df, ctx_df, tak_repo_path=TAK_REPO_PATH, scaler=None)
temporal_df, ctx_df = processor.run()

tokenizer = EMRTokenizer.from_processed_df(temporal_df)
train_ds = EMRDataset(train_df, train_ctx, tokenizer=tokenizer)
MODEL_CONFIG['ctx_dim'] = int(train_ds.context_df.shape[1]) # Dinamically updating shape
```

### 2. Train Model

Training is orchestrated via the three phase entry-points exposed at the package level — Phase-1 trains the embedder, Phase-2 pretrains the bidirectional encoder backbone with MLM, and Phase-3 attaches `TaskHeads` and fine-tunes for per-patient risk + time-to-event. See the autoresearch repository's `api.py` for the reference training driver.

```python
from intervene_enc.embedder import train_embedder
from intervene_enc.transformer import pretrain_transformer, finetune_transformer
```

The training contract:

- **Three-way patient split**: train / val / test. The test split is held out and never seen during training or early-stop selection — it is consumed only by `evaluation.ipynb` for headline metrics.
- **Scaler is fit on train** (saved to `checkpoints/scaler.pkl`) and reused on val/test.
- **Tokenizer** is built once from train and cached at `checkpoints/tokenizer.pt`.
- **Phase 1 caching**: when `(embed_dim, time2vec_dim, ctx_dim)` match the cached Phase-1 checkpoint, the embedder is reused and Phase 1 is skipped — Phase 2/3 are always retrained on each call.
- **DataLoaders**: Phase 1 + Phase 3 use bucket-batched natural distribution; Phase 2 uses a weighted bucket sampler so rare outcomes get balanced exposure (`pos_weight` is omitted there because the sampler already rebalances).

Model checkpoints are saved under `checkpoints/phase1/`, `checkpoints/phase2/`, and `checkpoints/phase3/`. Each phase function (`train_embedder`, `pretrain_transformer`, `finetune_transformer`) can be invoked directly.

### 3. Inference and Complication Risk Prediction

Inference is a **single bidirectional encoder pass** per patient. No autoregressive trajectory is generated — the per-outcome attention pool produces one risk probability and one time-to-event prediction per (patient, outcome) directly from the encoder's hidden states.

```python
import joblib
from pathlib import Path
from intervene_enc.embedder import EMREmbedding
from intervene_enc.transformer import InterveneEncoder
from intervene_enc.dataset import DataProcessor, EMRTokenizer, EMRDataset
from intervene_enc.inference import predict
from intervene_enc.config.model_config import *

# Load tokenizer and scaler
tokenizer = EMRTokenizer.load(Path(CHECKPOINT_PATH) / "tokenizer.pt")
scaler = joblib.load(Path(CHECKPOINT_PATH) / "scaler.pkl")

# Preprocess test data, truncated to the same input window used during Phase-3 alignment
processor = DataProcessor(df, ctx_df, scaler=scaler, tak_repo_path=TAK_REPO_PATH, max_input_days=5)
df, ctx_df = processor.run()
dataset_input = EMRDataset(df, ctx_df, tokenizer=tokenizer)

# Load the best available checkpoint (Phase-3 if available, otherwise Phase-2 without task heads)
embedder_model, *_ = EMREmbedding.load(PHASE1_CHECKPOINT, tokenizer=tokenizer)
p3_ckpt = Path(PHASE3_CHECKPOINT)
p2_ckpt = Path(PHASE2_CHECKPOINT)
ckpt_path = p3_ckpt if p3_ckpt.exists() else p2_ckpt
model, *_ = InterveneEncoder.load(ckpt_path, embedder=embedder_model, attach_task_heads=True)
model.eval()

# One-row-per-patient predictions: P_<outcome>, T_<outcome>
predictions = predict(model, dataset_input)
```

`predictions` is a DataFrame indexed by `PatientId` with two column families:

* `P_<outcome>` — sigmoid of the risk-head logit (RELEASE is dropped; reported as `1 − P(DEATH)`).
* `T_<outcome>` — softplus of the time-head output, hours from the seed window. RELEASE's `T_*` slot is the model's length-of-stay prediction.


### 4. Using as a module

You can perform local tests (not unit-tests) by activating the `.py` files, using the module as a package, as long as the file you are activating has __main__ section.

For example, run this from the root:
```bash
python -m intervene_enc.inference

# The inference module has a __main__ activation to run on a trained model
```
---

## 🧪 Running Unit-Tests

Run all tests:

Without validation prints:
```bash
python -m pytest unittests/
```

With validation prints:
```bash
python -m pytest -q -s unittests/
```

---

## 📦 Packaging Notes

To package without data/checkpoints:

```powershell
# Clean up any existing temp folder
Remove-Item -Recurse -Force .\intervene_enc_temp -ErrorAction SilentlyContinue

# Recreate the temp folder
New-Item -ItemType Directory -Path .\intervene_enc_temp | Out-Null

# Copy only what's needed
Copy-Item -Path .\intervene_enc -Destination .\intervene_enc_temp -Recurse
Copy-Item -Path .\setup.py, .\evaluation.ipynb, .\README.md, .\requirements.txt -Destination .\intervene_enc_temp

# Remove __pycache__ folders (platform-specific bytecode, not for distribution)
Get-ChildItem -Path .\intervene_enc_temp -Filter __pycache__ -Recurse -Directory | Remove-Item -Recurse -Force

# Zip it
Compress-Archive -Path .\intervene_enc_temp\* -DestinationPath .\intervene_enc.zip -Force

# Clean up
Remove-Item -Recurse -Force .\intervene_enc_temp
```

---

## 📌 Notes

- This project uses synthetic EMR data (`data/train/` and `data/test/`).
- For best results, ensure consistent preprocessing when saving/loading models.

---

## 🔄 End-to-End Workflow

Raw EMR Tables
│
▼
Per-patient Event Tokenization (with normalized absolute timestamps)
│
▼
🧠 Phase 1 – Train EMREmbedding (token + time + patient context)
│
▼
📚 Phase 2 – Pretrain a bidirectional Transformer encoder on the learned embeddings with masked language modelling
             (atomic-interval mask + time-since-admission and time-to-neighbour auxiliaries).
│
▼
🎯 Phase 3 – Attach `TaskHeads` (per-outcome attention pool + shared MLP) and fine-tune for per-patient
             risk (multi-label BCE with `pos_weight`) and time-to-event regression (smooth-L1 on positives).
             Backbone is at `phase3_backbone_lr_factor` LR.
│
▼
→ Read per-patient complication risk + length-of-stay from a single encoder pass (`evaluation.ipynb`).

---

## 📦 Module Overview

### 1. **`dataset.py`** – Temporal EMR Preprocessing

| Component            | Role                                                                                             |
|---------------------|--------------------------------------------------------------------------------------------------|
| `DataProcessor`        | Performs all necessary data processing, from input data to tokens_df.  |
| `EMRTokenizer`        | Builds vocabulary and per-outcome prevalence ratios from a processed temporal_df; filters outcomes below `OUTCOME_RARE_THRESHOLD_PCT`; saves/loads with `BucketBatchSampler` / `WeightedBucketBatchSampler` support. |
| `EMRDataset`        | Converts raw EMR tables into per-patient token sequences with relative time.                     |

| `collate_emr()`     | Pads sequences and returns tensors|

📌 **Why it matters:**  
Medical data varies in density and structure across patients. This dynamic preprocessing handles irregularity while preserving medically-relevant sequencing via `START/END` logic and relative timing.

>> This modules assumes the existance of prepared tak-repo-portable.json file, outputed from the Mediator as a hierarchy mapper of the different concepts.
---

### 2. **`embedder.py`** – EMR Representation Learning

| Component           | Role                                                                                              |
|--------------------|---------------------------------------------------------------------------------------------------|
| `Time2Vec`          | Learns periodic + trend encoding from inter-event durations.                                      |
| `EMREmbedding`      | Combines token, time, and patient context embeddings to create token representation.  |
| `train_embedder()`  | Phase-1 training. Loss = temporal next-token BCE (multi-hot over a future window) + Δt MSE auxiliary (joined once a single-stage scheduler lifts it after a BCE-only warmup). MLM has been removed in favour of a cleaner BCE+Δt curriculum. |

⚙️ **Phase 1: Learning Events Representation**  
Phase 1 learns a robust, patient-aware representation of their event sequences. It isolates the core structure of patient timelines without being confounded by the autoregressive depth of Transformers.
The embedder uses:
- 4 levels of tokens - The event token is seperated to 4 hierarichal components to impose similarity between tokens of the same domain: `GLUCOSE` -> `GLUCOSE_TREND` -> `GLUCOSE_TREND_Inc` -> `GLUCOSE_TREND_Inc_START`
- 1 level of time - ABS T from ADMISSION, to understand global patterns and relationships between non sequential events.

This architecture constructs event representations by concatenating five hierarchical levels: Raw Concept, Concept, Value, Position, and Absolute Time. This creates a dense vector that captures the intrinsic hierarchy of medical concepts (e.g., Glucose_High is a child of Glucose) while explicitly binding them to their timestamp.

We choose concatenation (Early Fusion) for the temporal component-unlike the standard additive approach to preserve the integrity of the medical signal. By keeping the time dimensions separate from the concept dimensions in the input, the model can clearly distinguish the "what" from the "when". This ensures that the core identity of a pathology (e.g., Hyperglycemia) remains stable and recognizable ("Hyperglycemia is Hyperglycemia") regardless of its timing, while allowing the projection layer to learn how time modifies its clinical significance (e.g., Morning vs. Evening).

Context Handling To condition these embeddings on static patient attributes (e.g., Age, Sex), we project the patient context vector and **add** it to the event sequence. This acts as a global bias, shifting the entire event manifold into a patient-specific subspace. This ensures that even before the Transformer layers, the event representations are already calibrated to the patient's demographic risk profile. Since the inference output the context projection and event embedding separately, we use **context dropout** (passing p% of the trajectories with no context) so that the embedder will learn to work with / without it, while still pushing the context projection layer towards the shared latent space. 

The training uses next-token prediction loss (temporal-window BCE) + time-delta MSE (Δt) auxiliary. Δt is held back behind a BCE-only warmup, then unlocked once Phase-1 has a meaningful main signal; its λ is calibrated once from the loss ratio at unlock and capped at a fraction of BCE so it never dominates. The legacy MLM auxiliary was removed — CBM (curriculum-by-masking, applied during Phase 2 over interval-atomic pairs) covers the same robustness need without adding a separate head.

---

### 3. **`transformer.py`** – Bidirectional EMR Encoder

| Component           | Role                                                                                              |
|--------------------|---------------------------------------------------------------------------------------------------|
| `InterveneEncoder`               | Bidirectional Transformer encoder over the learned embeddings. MLM head (full-vocab logits) + two time-aware auxiliary heads (time-since-admission, time-to-neighbour). Model inputs a trained embedder. |
| `BidirectionalSelfAttention` | Multi-head bidirectional self-attention with temporal RoPE — every position attends to every non-pad position. |
| `MLP` | SwiGLU MLP (SiLU Gating), kept from the AR backbone for parameter parity.                                 |
| `AdaLNBlock` | Encoder block with AdaLN-Zero conditioning on patient context. Attention is bidirectional (no causal mask). |
| `PerOutcomeAttnPool` | K learnable outcome queries cross-attend over the encoder output to produce one pooled feature per (patient, outcome). |
| `TaskHeads` | Phase-3 head module: `PerOutcomeAttnPool` → shared MLP → (risk_head, time_head). RELEASE is dropped from `risk_head` (reported as `1 − P(DEATH)`) and kept on `time_head` as length-of-stay regression. |
| `pretrain_transformer()` | Phase-2 MLM pre-training. Main loss: full-vocab cross-entropy on positions selected by `apply_mlm_mask` (atomic-interval-aware, hierarchical replacement tokens). Aux losses (`t_pos`, `t_local`) scheduled by `LambdaScheduleController` with per-aux fraction caps. |
| `finetune_transformer()` | Phase-3 outcome + time fine-tune. Backbone held at `phase3_backbone_lr_factor` LR; task heads at full `phase3_learning_rate`. Risk loss = `BCEWithLogitsLoss` with per-outcome `pos_weight` from training prevalence. Time loss = `smooth_l1_loss` over positive patients only, `λ_time` configurable via `phase3_time_lambda`. |

⚙️ **Phase 2: Bidirectional MLM pre-training**
The encoder learns event semantics from corrupted context. At each step:

- `apply_mlm_mask` (in `utils.py`) samples ~15% of eligible positions. Interval START/END pairs are masked atomically and replaced with `[MASK_INTERVAL_START]` / `[MASK_INTERVAL_END]`; other tokens become `[MASK]`. The original token id is retained as the CE target.
- **Main loss**: full-vocab cross-entropy at the masked positions only.
- **`t_pos` aux**: at every non-pad position, regress normalised time-since-admission (MSE) — forces the hidden state to retain global temporal placement.
- **`t_local` aux**: at masked positions only, regress `min(t−t_prev, t_next−t) / 24h` — forces masked tokens to retain local-time context after their concept identity is hidden.
- The embedder is trainable in Phase 2 at 10× lower LR than the backbone.

⚙️ **Phase 3: Outcome + time fine-tune**
Phase 3 attaches `TaskHeads` on top of the Phase-2-best encoder and trains:

- **Risk**: multi-label `BCEWithLogitsLoss` over the K−1 risk-indexed outcomes (RELEASE dropped). Per-outcome `pos_weight` derived from training-set prevalence; natural-distribution batches.
- **Time**: smooth-L1 between the K time-head outputs and the first-occurrence hours from sequence start, restricted to positive patients per outcome. RELEASE's slot is trained on patients who were released within the horizon — at inference it serves as length-of-stay regression.
- Backbone runs at `phase3_learning_rate × phase3_backbone_lr_factor` (default `0.01`).
- `λ_time` defaults to `0.1` and is controlled by `phase3_time_lambda`.

---

### 4. **`inference.py`** – Single-pass prediction

| Component           | Role                                                                                              |
|--------------------|---------------------------------------------------------------------------------------------------|
| `predict()` | **Primary inference function.** Runs a single bidirectional encoder pass over each patient and returns a DataFrame indexed by `PatientId` with `P_<outcome>` (sigmoid of risk-head logit) and `T_<outcome>` (softplus of time-head output, hours). When DEATH is present in the risk head, `P_RELEASE_EVENT` is added as `1 − P(DEATH)`. |
| `get_token_embedding()` | Returns the embedding vector of a specific token from a trained embedder. |

NOTE: Inference is a single forward pass per patient — substantially faster than the AR pipeline that preceded this version. No KV cache, no trajectory generation, no terminal-token forcing.

---

### 5. **`evaluation.ipynb`** – Self-contained complication-risk evaluation.

End-to-end evaluation on the held-out test split: re-process raw test data with the fitted scaler, build a 2-day truncated seed dataset, run a single encoder pass per patient via `inference.predict`, then score.

Headline framing is **patient-level AUROC / AUPRC / F1** — each (patient, outcome) contributes a single `(P_<outcome>, did_it_ever_occur)` pair, so rare outcomes stay stable without per-window count noise. `RELEASE_EVENT` is excluded from the AUC headline (it is the negation of DEATH in this cohort) and reported separately via length-of-stay MAE.

| Component           | Role                                                                                              |
|--------------------|---------------------------------------------------------------------------------------------------|
| `extract_ground_truth()` / `extract_ground_truth_episodes()` | First-occurrence and all-episode GT extracted from the untruncated test set. |
| `per_patient_auc()` | **Headline**. Per-(patient, outcome) `(P_<outcome>, label)` pair; AUROC, AUPRC, max-F1 (sweep PR curve), F1@0.5. |
| `weighted_mean_auc()` | Support-weighted (by `n_pos`) mean across outcomes — rare outcomes contribute less. |
| `time_head_mae()` | Per-outcome MAE between the time head's prediction and the nearest GT occurrence (positives only). |
| `length_of_stay_mae()` | Length-of-stay regression: predicted hours from sequence start to RELEASE vs the first GT RELEASE_EVENT time. Reads directly from `T_RELEASE_EVENT`. |
| `calibrate_temperature()` | Per-outcome temperature scalar via LBFGS — does not change rank order; improves probability calibration. |
| `reliability_diagram()` | Before/after calibration curves per outcome. |

---

## ✅ Model Capabilities

- ✔️ **Handles irregular time-series data** using relative deltas and Time2Vec.
- ✔️ **Captures both short- and long-range dependencies** with deep transformer blocks.
- ✔️ **Supports variable-length patient histories** using custom collate and attention masks.
- ✔️ **Imputes and predicts** events in structured EMR timelines.

---

## 📚 Citation & Acknowledgments

This work builds on and adapts ideas from the following sources:

- **Time2Vec** (Kazemi et al., 2019):  
  The temporal embedding design is adapted from the Time2Vec formulation.  
  📄 *A. Kazemi, S. Ghamizi, A.-H. Karimi. "Time2Vec: Learning a Vector Representation of Time." NeurIPS 2019 Time Series Workshop.*  
  [arXiv:1907.05321](https://arxiv.org/abs/1907.05321)

- **BERT** (Devlin et al., 2019):
  Phase-2 masked language modelling follows the BERT recipe — bidirectional self-attention with full-vocab cross-entropy on masked positions.
  📄 *J. Devlin, M.-W. Chang, K. Lee, K. Toutanova. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL 2019.*
  [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

- **nanoGPT** (Karpathy, 2023):  
  The training loop and transformer backbone shape are adapted from [nanoGPT](https://github.com/karpathy/nanoGPT),  
  with modifications for multi-stream EMR inputs and bidirectional attention.

- **RoPE / RoFormer** (Su et al., 2021):  
  The attention module uses rotary position embeddings adapted to continuous/absolute timestamps (temporal RoPE) to inject time into Q/K rotations.  
  📄 *J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen. "RoFormer: Enhanced Transformer with Rotary Position Embedding." arXiv:2104.09864.*  
  [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

- **AdaLN-Zero** (Peebles, W., & Xie, S., 2023):  
  Inspired by the paper "Scalable diffusion models with transformers", I added a customized block to the transformer designed to allow static context influence all generation steps. The [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.pdf) uses this method to inform the diffusion model of the label of the image it should generate.

And more...
