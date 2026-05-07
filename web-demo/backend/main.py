from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FRONTEND_DIR = BASE_DIR / "frontend"


class RecommendRequest(BaseModel):
    user_id: str


app = FastAPI(title="ISA Recommender Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_json_optional(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_artifacts() -> Dict:
    users = _load_json(ARTIFACTS_DIR / "users.json").get("users", [])
    catalog = _load_json(ARTIFACTS_DIR / "catalog.json")
    metrics = _load_json(ARTIFACTS_DIR / "metrics.json")
    user_info = _load_json_optional(ARTIFACTS_DIR / "user_info.json", {})
    recs_lstm = _load_json(ARTIFACTS_DIR / "recs_lstm.json")
    recs_nb = _load_json(ARTIFACTS_DIR / "recs_naive_bayes.json")
    recs_hybrid = _load_json(ARTIFACTS_DIR / "recs_hybrid.json")
    return {
        "users": users,
        "catalog": catalog,
        "metrics": metrics,
        "user_info": user_info,
        "models": {
            "lstm": recs_lstm,
            "naive_bayes": recs_nb,
            "hybrid": recs_hybrid,
        },
    }


ART = load_artifacts()


@app.get("/api/health")
def health():
    return {"status": "ok", "users": len(ART["users"])}


@app.get("/api/users")
def users():
    return {"users": ART["users"]}


@app.get("/api/users/random")
def random_user(min_reviews: int | None = None):
    if not ART["users"]:
        raise HTTPException(status_code=500, detail="No users available in artifacts")
    if min_reviews is None:
        return {"user_id": random.choice(ART["users"])}
    if min_reviews < 0:
        raise HTTPException(status_code=400, detail="min_reviews must be >= 0")

    user_info = ART.get("user_info", {})
    eligible_users = [
        uid
        for uid in ART["users"]
        if int(
            user_info.get(uid, {}).get(
                "review_count_dataset_restaurants",
                user_info.get(uid, {}).get("review_count", 0),
            )
        )
        >= min_reviews
    ]
    if not eligible_users:
        raise HTTPException(
            status_code=404,
            detail=f"No users found with at least {min_reviews} reviews",
        )
    return {
        "user_id": random.choice(eligible_users),
        "min_reviews": min_reviews,
        "eligible_users": len(eligible_users),
    }


@app.post("/api/recommend")
def recommend(payload: RecommendRequest):
    user_id = payload.user_id
    if user_id not in ART["users"]:
        raise HTTPException(status_code=404, detail=f"Unknown user_id: {user_id}")

    out = {"user_id": user_id, "user_info": ART.get("user_info", {}).get(user_id, {}), "models": {}}
    for model_key in ("lstm", "naive_bayes", "hybrid"):
        user_recs = ART["models"][model_key].get(user_id, [])
        rows: List[dict] = []
        for rec in user_recs[:10]:
            bid = rec.get("business_id")
            meta = ART["catalog"].get(
                bid,
                {
                    "business_id": bid,
                    "name": "Unknown",
                    "categories": "",
                    "rating": 0.0,
                    "last_review": "No review text available.",
                    "last_review_date": "",
                },
            )
            rows.append(
                {
                    "business_id": bid,
                    "score": rec.get("score"),
                    "name": meta.get("name"),
                    "categories": meta.get("categories"),
                    "rating": meta.get("rating"),
                    "last_review": meta.get("last_review"),
                    "last_review_date": meta.get("last_review_date"),
                }
            )

        out["models"][model_key] = {
            "metric": ART["metrics"].get(model_key, {}),
            "recommendations": rows,
        }

    return out


# Serve the frontend from the same app for a single-container compose run.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


