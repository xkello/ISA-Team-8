Additional data (post-processed data) can be found [on this google drive](https://drive.google.com/drive/folders/1SAjOsitNoxZyHHl7nFBSpGM-w9Vl6Iaz?usp=drive_link).

---

## User Manual — Running the Demo Locally

Follow these steps **in order** before running `docker compose up`.

### 1. Prerequisites

Make sure the following are installed on your machine:

| Tool | Version | Notes |
|---|---|---|
| [Python](https://www.python.org/downloads/) | ≥ 3.12 | Required for notebooks and scripts |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | latest | Python package manager used by this project |
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | latest | For `docker compose up` |

---

### 2. Clone the repository

```powershell
git clone <repo-url>
cd ISA-Team-8
```

---

### 3. Install Python dependencies

This project uses `uv`. Run from the repo root:

```powershell
uv sync
```

This creates a virtual environment and installs everything from `pyproject.toml` (TensorFlow, Keras, scikit-learn, pandas, etc.).

> **Alternative (plain pip):** `pip install -r web-demo/backend/requirements.txt` is enough if you only want to run the web demo without the notebooks.

---

### 4. Place the Yelp dataset

Download the Yelp Open Dataset and place the JSON files at:

```
original_data/yelp_json/yelp_academic_dataset_business.json
original_data/yelp_json/yelp_academic_dataset_review.json
original_data/yelp_json/yelp_academic_dataset_user.json
original_data/yelp_json/yelp_academic_dataset_tip.json
original_data/yelp_json/yelp_academic_dataset_checkin.json
```

> The additional post-processed CSVs can be downloaded from the [Google Drive](https://drive.google.com/drive/folders/1SAjOsitNoxZyHHl7nFBSpGM-w9Vl6Iaz?usp=drive_link) and placed under `custom_data/`.

---

### 5. Configure credentials

Open `credentials.ini` and fill in your [Weights & Biases](https://wandb.ai) API key:

```ini
[WandB]
api_key = <your_wandb_api_key>
```

> WandB is only required if you run experiment-tracking cells in the notebooks. You can skip this if you are only running the web demo.

---

### 6. Train models and export artifacts

The web demo reads pre-built artifact files. You need to generate them from the notebooks.

**Step 6a — Patch notebooks with export cells:**

```powershell
python web-demo/scripts/patch_notebooks_for_export.py
```

**Step 6b — Run the LSTM notebook:**

Open `lstm.ipynb` in Jupyter, run all cells top-to-bottom, then run the last export cell.

```powershell
uv run jupyter notebook lstm.ipynb
```

**Step 6c — Run the Hybrid Recommender notebook:**

Open `hybrid_recommender.ipynb` in Jupyter, run all cells top-to-bottom, then run the last export cell.

```powershell
uv run jupyter notebook hybrid_recommender.ipynb
```

Exported files will appear in `web-demo/artifacts/notebook_exports/`.

---

### 7. Build the demo JSON artifacts

```powershell
python web-demo/scripts/build_demo_artifacts.py
```

This generates all the JSON files the API reads at runtime under `web-demo/artifacts/`.

---

### 8. Start the demo

```powershell
docker compose up --build
```

Then open **http://localhost:8000** in your browser.

---

## EDA notebooks
- EDA_basic.ipynb = EDA for the entire dataset and all businesses
- EDA_restaurant.ipynb = EDA for our selected subset (restaurants)

Specific instructions for running the EDA notebooks are written at the beginning of each of the notebooks.


## Hybrid Recommender Notebook

A full one-notebook pipeline is available at `hybrid_recommender.ipynb`.

It implements:
- city-scoped restaurant filtering (default: Philadelphia)
- collaborative filtering (SVD-based)
- content-based filtering from categories
- per-user model selection (CF vs CBF)
- evaluation metrics (`Precision@K`, `Recall@K`, `HitRate@K`, `NDCG@K`, `MAP@K`, `Coverage@K`)

Dataset location expected by the notebook:
- `original_data/yelp_json/yelp_academic_dataset_business.json`
- `original_data/yelp_json/yelp_academic_dataset_review.json`

Run the notebook with Jupyter in this project environment and execute cells top-to-bottom.

---

## Model Notes

### 1. LSTM (Sequential recommender)

**What it does:** predicts the next restaurant a user will enjoy based on their ordered visit history.

**Why LSTM:** visit sequences have temporal structure — what someone liked last month matters more than what they liked two years ago. An LSTM can learn that recency matters without you manually engineering it.

**Training data:** the post-processed `custom_data/merged_2.csv` file, filtered to reviews ≥ 4 stars only (average rating in the dataset is 3.86, so we focus on what people actually liked). Users with fewer than 2 reviews are excluded because you need at least an input + a target.

**Sequence config:**
- Sequence length: **5** visits per training sample
- Users with fewer than 5 visits get zero-padded sequences (Keras masking layer handles this)
- 80/20 train/test split (random seed 28)

**Architecture:**
- Restaurant embedding: dim **64**
- User embedding: dim **28**
- LSTM: **112 units** with LayerNorm
- Output: softmax over all restaurants in the training set

**Training config:** Adam, lr `1e-5`, **85 epochs**, batch size **1024**, loss: sparse categorical cross-entropy.

**Metrics tracked:** accuracy, top-5 accuracy, top-10 accuracy (train + test per epoch).

---

### 2. Naive Bayes (Baseline)

**What it does:** predicts the next restaurant from unordered visit history — it treats the past sequence as a bag of restaurants, not an ordered timeline.

**Why NB:** it is a fast, interpretable baseline. Running it alongside LSTM shows how much the temporal ordering actually matters. It also degrades gracefully on sparse users.

**Training data:** same train/test split as LSTM so metrics are directly comparable.

**Features:** sparse restaurant-count vectors (which restaurants appear in the user's history, how many times). Optionally includes mean attribute values per sequence window.

**Key hyperparameter:** Laplace smoothing alpha, selected by sweeping `[0.05, 0.1, 0.3, 0.5, 1.0]` and picking the run with the best `test_top_10_accuracy`.

**Training:** 12 epochs of mini-batch partial fitting (batch size 8192), sampling 20% of training rows per epoch to get curves without full re-scans.

---

### 3. Hybrid Recommender (CF + CBF)

**What it does:** blends collaborative filtering (SVD) and content-based filtering (category vectors) into a single ranked list of 10 restaurants per user.

**Why hybrid:** CF alone struggles with sparse users (not many reviews) and cold-start items (new restaurants). CBF alone ignores what similar users liked. The hybrid covers both gaps.

**Scope:** city-scoped to **Philadelphia, PA** for computational tractability. Full-dataset runs are possible by changing `TARGET_CITY` and `SAMPLE_USER_FRAC = 1.0`.

**Training data size (default run):**
- 20% user sample (`SAMPLE_USER_FRAC = 0.20`) for fast iteration
- Minimum 3 reviews per user, minimum 5 reviews per restaurant (interaction denoising)
- Only reviews ≥ 4 stars count as positive signal (`LIKE_STARS = 4.0`)
- Temporal split: leave-last-two per user → train / val / test

**CF component:** `TruncatedSVD` on the sparse user-item rating matrix, up to **64 latent dimensions** (capped at matrix size for numerical stability).

**CBF component:** multi-label one-hot encoding of restaurant categories + cosine similarity between user's weighted category profile and each restaurant's category vector.

**Hybrid blend:** `hybrid_score = 0.9 × CF_norm + 0.1 × CBF_norm` (CF-dominant by default; `HYBRID_ALPHA = 0.9`).

**Cold-start fallback:** users with 0 training interactions get a popularity-based ranking (average star rating across known restaurants ≥ 4.0).

**Evaluation metrics:** `Precision@10`, `Recall@10`, `HitRate@10`, `NDCG@10`, `MAP@10`, `Coverage@10`.

