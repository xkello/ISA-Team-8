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
| [Ollama](https://ollama.com/download) | latest | Local LLM inference server (only needed for notebook LLM cells) |

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

## LLM Integration (Ollama)

We run local LLM models on the server using Ollama.

### Models used

- gemma3:4b  
  General language model used for reasoning and decision making.

- qwen3-embedding:4b  
  Embedding model used to convert text into numeric vectors for similarity
  comparisons.

Both models run locally on the same server (CPU, no GPU required).

---

## How the models are used

The API server decides which model to use depending on the task.

### 1. Embedding model (qwen3-embedding:4b)

Used when we need numeric representations of text.

Typical uses:
- comparing restaurant descriptions
- comparing user review text to restaurant profiles
- semantic similarity
- clustering
- retrieval tasks

Flow:


API → Ollama embeddings endpoint → qwen3-embedding model → vectors → API


The server calls the embedding model when it needs vectors, not text output.

---

### 2. Language model (gemma3:4b)

Used when we need reasoning or text generation.

Typical uses:
- analyzing user preferences
- ranking candidate restaurants
- explaining recommendations
- summarizing reviews
- natural language responses

Flow:


API → Ollama generate endpoint → gemma3 model → text response → API


The server calls this model when it needs reasoning or written output.

---

## How both models work together

Example recommendation flow with both models:

1. User sends request
2. API builds restaurant candidate list
3. API extracts user preference text
4. Embedding model converts:
   - user preferences → vectors
   - restaurant descriptions → vectors
5. API computes similarity scores
6. Language model receives:
   - user preferences
   - top candidates
   - instructions
7. Language model:
   - refines ranking
   - generates explanation
8. API returns final JSON

So:


Embeddings → similarity math
Language model → reasoning + explanation


They do different jobs and do not talk to each other directly.
The API server coordinates both.

---

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

