from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MERGED_PATH = ROOT / "custom_data" / "merged_2.csv"
BUSINESS_JSON_PATH = ROOT / "original_data" / "yelp_json" / "yelp_academic_dataset_business.json"
REVIEW_JSON_PATH = ROOT / "original_data" / "yelp_json" / "yelp_academic_dataset_review.json"
ARTIFACTS_DIR = ROOT / "web-demo" / "artifacts"
FRONTEND_DIR = ROOT / "web-demo" / "frontend"

MAX_USERS = 1200
TOP_K = 10
RANDOM_STATE = 28
TOP_USER_TYPES = 5

GENERIC_RESTAURANT_TYPES = {
    "restaurant",
    "restaurants",
    "food",
    "bar",
    "bars",
    "cafe",
    "cafes",
    "coffee & tea",
    "nightlife",
}


def _parse_categories(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [c.strip() for c in str(raw).split(",") if c and c.strip()]


def _load_restaurant_business_ids(path: Path, candidate_bids: set[str]) -> set[str]:
    restaurant_bids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            bid = obj.get("business_id")
            if bid not in candidate_bids:
                continue
            if "Restaurants" in _parse_categories(obj.get("categories")):
                restaurant_bids.add(bid)
    return restaurant_bids


def _is_specific_type(category: str) -> bool:
    return category.strip().lower() not in GENERIC_RESTAURANT_TYPES


@dataclass
class EvalResult:
    top5: float
    top10: float


def _topk_from_counter(counter: Counter, seen: set[str], top_k: int) -> List[Tuple[str, float]]:
    ranked = []
    for bid, score in counter.most_common():
        if bid in seen:
            continue
        ranked.append((bid, float(score)))
        if len(ranked) >= top_k:
            break
    return ranked


def _normalize_counter(counter: Counter) -> Counter:
    if not counter:
        return counter
    values = np.array(list(counter.values()), dtype=np.float32)
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax - vmin < 1e-12:
        return Counter({k: 0.5 for k in counter})
    return Counter({k: float((v - vmin) / (vmax - vmin)) for k, v in counter.items()})


def _evaluate(recs_by_user: Dict[str, List[Tuple[str, float]]], targets: Dict[str, str]) -> EvalResult:
    hit5 = 0
    hit10 = 0
    n = len(targets)
    for uid, target in targets.items():
        rec_ids = [bid for bid, _ in recs_by_user.get(uid, [])]
        if target in rec_ids[:5]:
            hit5 += 1
        if target in rec_ids[:10]:
            hit10 += 1
    if n == 0:
        return EvalResult(top5=float("nan"), top10=float("nan"))
    return EvalResult(top5=hit5 / n, top10=hit10 / n)


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    cols = ["user_id", "business_id", "date", "stars_review"]
    df_all = pd.read_csv(MERGED_PATH, usecols=cols)
    df_all["business_id"] = df_all["business_id"].astype(str)
    # merged_2.csv stores Unix timestamps in seconds (10-digit ints).
    df_all["date"] = pd.to_datetime(df_all["date"], unit="s", errors="coerce")
    df_all = df_all.dropna(subset=["date"])
    df_all = df_all.sort_values(["user_id", "date"])

    candidate_bids = set(df_all["business_id"].unique().tolist())
    restaurant_bids = _load_restaurant_business_ids(BUSINESS_JSON_PATH, candidate_bids)
    df_all["is_restaurant"] = df_all["business_id"].isin(restaurant_bids)

    # Model artifacts are still built from positive restaurant interactions only.
    df = df_all[(df_all["stars_review"] >= 4) & (df_all["is_restaurant"])].copy()

    # Build user sequences and keep users with enough history.
    grouped = df.groupby("user_id")["business_id"].apply(list)
    grouped = grouped[grouped.apply(len) >= 3]

    rng = np.random.default_rng(RANDOM_STATE)
    users = grouped.index.to_numpy()
    if len(users) > MAX_USERS:
        users = rng.choice(users, size=MAX_USERS, replace=False)
    users = sorted(users.tolist())

    # Keep only selected users.
    user_sequences = {uid: grouped.loc[uid] for uid in users}

    # Training context and evaluation targets (last item holdout).
    contexts = {uid: seq[:-1] for uid, seq in user_sequences.items()}
    targets = {uid: seq[-1] for uid, seq in user_sequences.items()}

    # Global popularity from training contexts.
    popularity = Counter()
    for seq in contexts.values():
        popularity.update(seq)

    # Transition model for LSTM-like behavior.
    transitions: Dict[str, Counter] = defaultdict(Counter)
    for seq in contexts.values():
        for a, b in zip(seq[:-1], seq[1:]):
            transitions[a][b] += 1

    # Co-occurrence model for NB-like behavior.
    cooccur: Dict[str, Counter] = defaultdict(Counter)
    for seq in contexts.values():
        uniq = list(dict.fromkeys(seq))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                a = uniq[i]
                b = uniq[j]
                cooccur[a][b] += 1
                cooccur[b][a] += 1

    recs_lstm: Dict[str, List[Tuple[str, float]]] = {}
    recs_nb: Dict[str, List[Tuple[str, float]]] = {}
    recs_hybrid: Dict[str, List[Tuple[str, float]]] = {}
    user_info: Dict[str, dict] = {}
    user_recent_bids = set()
    user_history_bids = set()
    user_history_bid_list: Dict[str, List[str]] = {}

    for uid in users:
        seq = contexts[uid]
        seen = set(seq)

        user_history = df_all[df_all["user_id"] == uid].sort_values("date")
        user_history_rest = user_history[user_history["is_restaurant"]].copy()
        user_history_bids.update(user_history_rest["business_id"].astype(str).tolist())
        user_history_bid_list[uid] = user_history_rest["business_id"].astype(str).tolist()
        recent_rows = user_history_rest.tail(3).sort_values("date", ascending=False)
        recent_visits = [
            {
                "business_id": str(row.business_id),
                "rating": float(row.stars_review),
                "date": row.date.strftime("%Y-%m-%d"),
            }
            for row in recent_rows.itertuples(index=False)
        ]
        user_recent_bids.update([v["business_id"] for v in recent_visits])
        user_info[uid] = {
            "user_id": uid,
            "avg_rating": float(user_history_rest["stars_review"].mean()),
            "review_count": int(len(user_history_rest)),
            "review_count_dataset_restaurants": int(len(user_history_rest)),
            "total_review_count": int(len(user_history)),
            "unique_restaurants": int(user_history_rest["business_id"].nunique()),
            "positive_review_ratio": float((user_history_rest["stars_review"] >= 4).mean()),
            "top_restaurant_types": [],
            "recent_visits": recent_visits,
        }

        # LSTM-like: last-item transitions + small popularity fallback.
        last_bid = seq[-1]
        lstm_counter = Counter(transitions.get(last_bid, {}))
        if len(lstm_counter) < TOP_K:
            for bid, cnt in popularity.items():
                lstm_counter[bid] += 0.05 * cnt
        lstm_top = _topk_from_counter(lstm_counter, seen=seen, top_k=TOP_K)
        recs_lstm[uid] = lstm_top

        # NB-like: aggregate co-occurrence counts from all seen items.
        nb_counter = Counter()
        for bid in set(seq):
            nb_counter.update(cooccur.get(bid, {}))
        if len(nb_counter) < TOP_K:
            for bid, cnt in popularity.items():
                nb_counter[bid] += 0.02 * cnt
        nb_top = _topk_from_counter(nb_counter, seen=seen, top_k=TOP_K)
        recs_nb[uid] = nb_top

        # Hybrid: weighted blend of normalized LSTM and NB scores.
        lstm_norm = _normalize_counter(lstm_counter)
        nb_norm = _normalize_counter(nb_counter)
        all_keys = set(lstm_norm) | set(nb_norm)
        hy_counter = Counter({k: 0.6 * lstm_norm.get(k, 0.0) + 0.4 * nb_norm.get(k, 0.0) for k in all_keys})
        hy_top = _topk_from_counter(hy_counter, seen=seen, top_k=TOP_K)
        recs_hybrid[uid] = hy_top

    # Evaluate on last-item holdout for a consistent metric block.
    eval_lstm = _evaluate(recs_lstm, targets)
    eval_nb = _evaluate(recs_nb, targets)
    eval_hybrid = _evaluate(recs_hybrid, targets)

    # Collect all recommended business ids for metadata extraction.
    all_rec_bids = set()
    for d in (recs_lstm, recs_nb, recs_hybrid):
        for rows in d.values():
            all_rec_bids.update([bid for bid, _ in rows])

    businesses = {}
    needed_bids = set(all_rec_bids) | set(user_recent_bids) | set(user_history_bids)
    business_categories: Dict[str, List[str]] = {}
    unique_categories = set()
    with BUSINESS_JSON_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            bid = obj.get("business_id")
            if bid not in needed_bids:
                continue
            cats = _parse_categories(obj.get("categories"))
            unique_categories = unique_categories | set(cats)
            businesses[bid] = {
                "name": obj.get("name", "Unknown"),
                "categories": obj.get("categories") or "",
                "stars": float(obj.get("stars") or 0.0),
                "latitude": float(obj.get("latitude")),
                "longitude": float(obj.get("longitude")),
                "address": obj.get("address")
            }
            business_categories[bid] = cats

    for uid, info in user_info.items():
        type_counter = Counter()
        for bid in user_history_bid_list.get(uid, []):
            specific_types = [c for c in business_categories.get(bid, []) if _is_specific_type(c)]
            type_counter.update(specific_types)
        info["top_restaurant_types"] = [
            {"type": t, "count": int(c)} for t, c in type_counter.most_common(TOP_USER_TYPES)
        ]

        enriched = []
        for visit in info.get("recent_visits", []):
            bid = visit.get("business_id")
            enriched.append({
                **visit,
                "name": businesses.get(bid, {}).get("name", "Unknown"),
            })
        info["recent_visits"] = enriched

    # Latest review text for recommended businesses.
    latest_review: Dict[str, Tuple[str, str]] = {}
    with REVIEW_JSON_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            bid = obj.get("business_id")
            if bid not in all_rec_bids:
                continue
            date = obj.get("date") or ""
            text = (obj.get("text") or "").replace("\n", " ").strip()
            text = text[:240]
            prev = latest_review.get(bid)
            if prev is None or date > prev[0]:
                latest_review[bid] = (date, text)

    catalog = {}
    for bid in all_rec_bids:
        meta = businesses.get(bid, {"name": "Unknown", "categories": "", "stars": 0.0})
        review_entry = latest_review.get(bid, ("", "No review text available."))
        catalog[bid] = {
            "business_id": bid,
            "name": meta["name"],
            "latitude": meta["latitude"],
            "longitude": meta["longitude"],
            "address": meta["address"],
            "categories": meta["categories"],
            "rating": meta["stars"],
            "last_review_date": review_entry[0],
            "last_review": review_entry[1] or "No review text available.",
        }

    metrics = {
        "lstm": {
            "label": "LSTM (ordered)",
            "top5_success": eval_lstm.top5,
            "top10_success": eval_lstm.top10,
            "notes": "Next-item sequence model",
        },
        "naive_bayes": {
            "label": "Naive Bayes (unordered)",
            "top5_success": eval_nb.top5,
            "top10_success": eval_nb.top10,
            "notes": "Bag-of-history baseline",
        },
        "hybrid": {
            "label": "Hybrid",
            "top5_success": eval_hybrid.top5,
            "top10_success": eval_hybrid.top10,
            "notes": "Weighted blend of LSTM-like and NB-like scores",
        },
    }

    def serialize_recs(d: Dict[str, List[Tuple[str, float]]]) -> Dict[str, List[dict]]:
        return {
            uid: [{"business_id": bid, "score": score} for bid, score in rows]
            for uid, rows in d.items()
        }

    (ARTIFACTS_DIR / "users.json").write_text(json.dumps({"users": users}, indent=2), encoding="utf-8")
    (ARTIFACTS_DIR / "catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (ARTIFACTS_DIR / "user_info.json").write_text(json.dumps(user_info, indent=2), encoding="utf-8")
    (ARTIFACTS_DIR / "recs_lstm.json").write_text(json.dumps(serialize_recs(recs_lstm), indent=2), encoding="utf-8")
    (ARTIFACTS_DIR / "recs_naive_bayes.json").write_text(json.dumps(serialize_recs(recs_nb), indent=2), encoding="utf-8")
    (ARTIFACTS_DIR / "recs_hybrid.json").write_text(json.dumps(serialize_recs(recs_hybrid), indent=2), encoding="utf-8")
    (FRONTEND_DIR / "possible_categories.json").write_text(json.dumps(sorted(list(unique_categories))), encoding="utf-8")

    print(f"Artifacts written to: {ARTIFACTS_DIR} and {FRONTEND_DIR}")
    print(f"Users: {len(users)} | Catalog items used: {len(catalog)}")


if __name__ == "__main__":
    main()
