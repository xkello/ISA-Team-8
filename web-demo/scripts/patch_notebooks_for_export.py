from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[2]
LSTM_NB = ROOT / "lstm.ipynb"
HYBRID_NB = ROOT / "hybrid_recommender.ipynb"


LSTM_EXPORT_CELL = """# EXPORT_REAL_MODELS_LSTM_NB_V2
# Export real trained LSTM + Naive Bayes models and minimal metadata.
from pathlib import Path
import json
import pickle

EXPORT_DIR = Path("web-demo/artifacts/notebook_exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

if "lstm_model" not in globals() and "model" in globals():
    # Keep a stable reference even if later cells reuse `model` for NB candidates.
    lstm_model = model

if "lstm_model" not in globals():
    raise RuntimeError("LSTM model not found. Run LSTM training/evaluation cells first.")

lstm_model.save(EXPORT_DIR / "lstm_model.keras")

if "best_model" not in globals():
    raise RuntimeError("Naive Bayes best_model not found. Run NB sweep cells first.")

with open(EXPORT_DIR / "naive_bayes_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

meta = {
    "sequence_length": int(sequence_length) if "sequence_length" in globals() else None,
    "num_restaurants": int(num_restaurants) if "num_restaurants" in globals() else None,
    "num_users": int(num_users) if "num_users" in globals() else None,
    "best_alpha": float(best_alpha) if "best_alpha" in globals() else None,
}

if "comparison_df" in globals():
    meta["comparison"] = comparison_df.to_dict(orient="records")
if "nb_metrics" in globals():
    meta["nb_metrics"] = nb_metrics.to_dict(orient="records")

(EXPORT_DIR / "lstm_nb_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"Saved real models to: {EXPORT_DIR}")
"""


HYBRID_EXPORT_CELL = """# EXPORT_REAL_MODELS_HYBRID_V2
# Export a real hybrid model bundle (CF+CBF+routing state), not precomputed rec lists.
from pathlib import Path
import json
import pickle

EXPORT_DIR = Path("web-demo/artifacts/notebook_exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

required_names = [
    "u2i", "b2i", "i2b", "all_users", "all_items",
    "user_factors", "item_factors", "X_item_cat", "item_cat_norm",
    "user_profiles", "train_seen", "all_seen", "train_counts",
    "popularity_rank", "popularity_scores", "HYBRID_ALPHA", "ROUTE_POP_MAX_REVIEWS",
]
missing = [name for name in required_names if name not in globals()]
if missing:
    raise RuntimeError(f"Hybrid export missing required variables: {missing}")

hybrid_bundle = {
    "u2i": u2i,
    "b2i": b2i,
    "i2b": i2b,
    "all_users": all_users,
    "all_items": all_items,
    "user_factors": user_factors,
    "item_factors": item_factors,
    "X_item_cat": X_item_cat,
    "item_cat_norm": item_cat_norm,
    "user_profiles": user_profiles,
    "train_seen": {k: list(v) for k, v in train_seen.items()},
    "all_seen": {k: list(v) for k, v in all_seen.items()},
    "train_counts": train_counts,
    "popularity_rank": popularity_rank,
    "popularity_scores": popularity_scores,
    "HYBRID_ALPHA": HYBRID_ALPHA,
    "ROUTE_POP_MAX_REVIEWS": ROUTE_POP_MAX_REVIEWS,
    "K": K if "K" in globals() else 10,
}

with open(EXPORT_DIR / "hybrid_model_bundle.pkl", "wb") as f:
    pickle.dump(hybrid_bundle, f)

meta = {}
if "overall_df" in globals():
    meta["overall_metrics"] = overall_df.to_dict(orient="records")
if "cohort_df" in globals():
    meta["cohort_metrics"] = cohort_df.to_dict(orient="records")

(EXPORT_DIR / "hybrid_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"Saved real hybrid bundle to: {EXPORT_DIR}")
"""


def _upsert_export_cell(nb_path: Path, unique_tag: str, code: str, old_markers: list[str]) -> None:
    nb = nbformat.read(nb_path, as_version=4)

    # Small value tweaks for faster local reruns and to disable cloud logging by default.
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        src = src.replace("NB_LOG_TO_WANDB = True", "NB_LOG_TO_WANDB = False")
        src = src.replace("logged = True", "logged = False")
        cell["source"] = src

    new_cells = []
    found = False
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            new_cells.append(cell)
            continue
        src = cell.get("source", "") or ""
        if unique_tag in src:
            cell["source"] = code
            found = True
            new_cells.append(cell)
            continue
        if any(marker in src for marker in old_markers):
            # Drop legacy export cells so only one authoritative export cell exists.
            continue
        new_cells.append(cell)

    if not found:
        new_cells.append(nbformat.v4.new_code_cell(source=code))

    nb.cells = new_cells

    nbformat.write(nb, nb_path)


def main() -> None:
    _upsert_export_cell(
        LSTM_NB,
        unique_tag="EXPORT_REAL_MODELS_LSTM_NB_V2",
        code=LSTM_EXPORT_CELL,
        old_markers=[
            "lstm_nb_metrics.json",
            "EXPORT_REAL_MODELS_LSTM_NB_V1",
        ],
    )
    _upsert_export_cell(
        HYBRID_NB,
        unique_tag="EXPORT_REAL_MODELS_HYBRID_V2",
        code=HYBRID_EXPORT_CELL,
        old_markers=[
            "hybrid_metrics_and_recs.json",
            "EXPORT_REAL_MODELS_HYBRID_V1",
        ],
    )
    print("Patched notebooks with real-model export cells and safer logging defaults.")


if __name__ == "__main__":
    main()


