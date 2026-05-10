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
    desired_category: str = ""


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
    cold_start_users = _load_json_optional(ARTIFACTS_DIR / "cold_start_users.json", {}).get("users", [])
    catalog = _load_json(ARTIFACTS_DIR / "catalog.json")
    metrics = _load_json(ARTIFACTS_DIR / "metrics.json")
    user_info = _load_json_optional(ARTIFACTS_DIR / "user_info.json", {})
    recs_lstm = _load_json(ARTIFACTS_DIR / "recs_lstm.json")
    recs_nb = _load_json(ARTIFACTS_DIR / "recs_naive_bayes.json")
    recs_hybrid = _load_json(ARTIFACTS_DIR / "recs_hybrid.json")
    return {
        "users": users,
        "cold_start_users": cold_start_users,
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
# print(ART["catalog"])
# print(type(ART["catalog"]))


@app.get("/api/health")
def health():
    return {"status": "ok", "users": len(ART["users"])}


@app.get("/api/users")
def users():
    return {"users": ART["users"]}


@app.get("/api/categories")
def categories():
    values = set()
    for meta in ART.get("catalog", {}).values():
        raw = meta.get("categories") or ""
        parts = [p.strip() for p in str(raw).split(",") if p and p.strip()]
        values.update(parts)
    return {"categories": sorted(values)}


@app.get("/api/users/random")
def random_user(
    min_reviews: int | None = None,
    cold_start: bool = False,
    min_positive_ratio: float | None = None,
    min_avg_rating: float | None = None,
    max_third_review_date: str | None = None,
):
    if not ART["users"]:
        raise HTTPException(status_code=500, detail="No users available in artifacts")
    if cold_start and any(v is not None for v in [min_reviews, min_positive_ratio, min_avg_rating, max_third_review_date]):
        raise HTTPException(status_code=400, detail="Cannot combine cold_start with other filters")
    if cold_start:
        if not ART.get("cold_start_users"):
            raise HTTPException(status_code=404, detail="No cold-start demo users available")
        return {
            "user_id": random.choice(ART["cold_start_users"]),
            "cold_start": True,
            "eligible_users": len(ART["cold_start_users"]),
        }

    user_info = ART.get("user_info", {})
    eligible_users = []
    for uid in ART["users"]:
        info = user_info.get(uid, {})
        review_count = int(info.get("review_count_dataset_restaurants", info.get("review_count", 0)))
        if min_reviews is not None and review_count < min_reviews:
            continue
        if min_positive_ratio is not None:
            ratio = float(info.get("positive_review_ratio") or 0)
            if ratio < min_positive_ratio:
                continue
        if min_avg_rating is not None:
            avg = float(info.get("avg_rating") or 0)
            if avg < min_avg_rating:
                continue
        if max_third_review_date is not None:
            visits = info.get("recent_visits", [])
            if len(visits) < 3:
                continue
            third_date = visits[2].get("date", "")
            if not third_date or third_date > max_third_review_date:
                continue
        eligible_users.append(uid)

    if not eligible_users:
        raise HTTPException(status_code=404, detail="No users match the given filters")
    return {
        "user_id": random.choice(eligible_users),
        "eligible_users": len(eligible_users),
    }


@app.post("/api/recommend")
def recommend(payload: RecommendRequest):
    user_id = payload.user_id
    desired_category = payload.desired_category
    known_users = set(ART["users"]) | set(ART.get("cold_start_users", []))
    if user_id not in known_users:
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

            categories = meta.get("categories") or ""

            if desired_category in categories:
                rows.append(
                    {
                        "business_id": bid,
                        "score": rec.get("score"),
                        "name": meta.get("name"),
                        "categories": meta.get("categories"),
                        "rating": meta.get("rating"),
                        "last_review": meta.get("last_review"),
                        "last_review_date": meta.get("last_review_date"),
                        "latitude": meta.get("latitude"),
                        "longitude": meta.get("longitude"),
                        "address": meta.get("address")
                    }
                )

        out["models"][model_key] = {
            "metric": ART["metrics"].get(model_key, {}),
            "recommendations": rows,
        }

    return out


# Serve the frontend from the same app for a single-container compose run.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


