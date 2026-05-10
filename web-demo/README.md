# Web Demo (Docker Compose)

This folder contains a lightweight web app that presents recommendations from:
- LSTM
- Naive Bayes
- Hybrid

The app flow is:
1. Select `user_id` manually or with **Random user**
2. Optionally use **Min reviews** or **Cold start user (0 reviews)** for random selection
3. Click **Confirm**
4. View 10 recommendations from each model in separate rows

Each recommendation card shows:
- restaurant name
- categories
- rating
- latest review snippet

## Start with Docker Compose

From repo root:

```powershell
docker compose up --build
```

Open:
- `http://localhost:8000`

## Runtime artifacts

The demo API reads runtime JSON files from `web-demo/artifacts/`.

For a demo based on real trained notebook models, first export model files from notebooks (next section), then regenerate demo JSON.

## Export real models from notebooks

1) Patch notebooks with export cells:

```powershell
python web-demo/scripts/patch_notebooks_for_export.py
```

2) Open and run `lstm.ipynb` fully, then run the last export cell.

3) Open and run `hybrid_recommender.ipynb` fully, then run the last export cell.

Exported files are written to `web-demo/artifacts/notebook_exports/`:
- `lstm_model.keras`
- `naive_bayes_model.pkl`
- `lstm_nb_meta.json`
- `hybrid_model_bundle.pkl`
- `hybrid_meta.json`

## Rebuild demo JSON artifacts

```powershell
python web-demo/scripts/build_demo_artifacts.py
```

This regenerates:
- `web-demo/artifacts/users.json`
- `web-demo/artifacts/cold_start_users.json`
- `web-demo/artifacts/catalog.json`
- `web-demo/artifacts/metrics.json`
- `web-demo/artifacts/recs_lstm.json`
- `web-demo/artifacts/recs_naive_bayes.json`
- `web-demo/artifacts/recs_hybrid.json`
