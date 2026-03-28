Additional data (post-processed data) can be found [on this google drive](https://drive.google.com/drive/folders/1SAjOsitNoxZyHHl7nFBSpGM-w9Vl6Iaz?usp=drive_link).

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
