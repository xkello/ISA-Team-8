# FastAPI is used to expose the recommender as an HTTP API.
# HTTPException lets us return proper API errors with status codes.
from fastapi import FastAPI, HTTPException

# Pydantic models validate incoming JSON payloads automatically.
from pydantic import BaseModel, Field

# Type hints make the code easier to understand and help editors/linters.
from typing import Any, Dict, List, Optional

# Path gives us a cleaner way to work with file system paths.
from pathlib import Path

# Pandas is used for in-memory tabular processing/filtering/sorting.
import pandas as pd

# Yelp JSON files are in JSON Lines format, so we parse review rows line-by-line.
import json


# ------------------------------------------------------------------------------
# FastAPI application setup
# ------------------------------------------------------------------------------

# Create the API app instance.
# The title will appear in OpenAPI docs (/docs).
app = FastAPI(title="Yelp Recommender API")


# ------------------------------------------------------------------------------
# Dataset configuration
# ------------------------------------------------------------------------------

# Absolute paths to each Yelp dataset file on disk.
# These are expected to exist on the Ollama server.
DATASET_PATHS = {
    "business": Path("/home/kello/ollama/data/yelp_academic_dataset_business.json"),
    "review": Path("/home/kello/ollama/data/yelp_academic_dataset_review.json"),
    "user": Path("/home/kello/ollama/data/yelp_academic_dataset_user.json"),
    "tip": Path("/home/kello/ollama/data/yelp_academic_dataset_tip.json"),
    "checkin": Path("/home/kello/ollama/data/yelp_academic_dataset_checkin.json"),
}

# Optional column selection per dataset.
# This is useful to keep memory usage smaller by only loading fields we actually use.
#
# - business: only load columns needed for recommendation/filtering
# - review: handled separately by streaming, so this list is mainly documentation
# - user/tip/checkin: currently unused, so None means "load all columns if needed"
DATASET_COLUMNS = {
    "business": [
        "business_id",
        "name",
        "city",
        "state",
        "stars",
        "review_count",
        "is_open",
        "categories",
    ],
    "review": [
        "review_id",
        "user_id",
        "business_id",
        "stars",
        "text",
    ],
    "user": None,
    "tip": None,
    "checkin": None,
}

# In-process cache for datasets that were already loaded.
# This avoids rereading large JSON files on every request.
# Important: we always return copies of cached DataFrames so request logic
# does not mutate the shared cached object.
DATA_CACHE: Dict[str, pd.DataFrame] = {}


# ------------------------------------------------------------------------------
# Request models
# ------------------------------------------------------------------------------

class DatasetSpec(BaseModel):
    # Filters to apply to a dataset after loading.
    # Example:
    # {
    #   "city": "Las Vegas",
    #   "stars_gte": 4.0,
    #   "review_count_gte": 100
    # }
    filters: Dict[str, Any] = Field(default_factory=dict)


class Operation(BaseModel):
    # Type of operation to execute.
    # Supported values in this app:
    # - exclude_businesses_reviewed_by_user
    # - boost_by_reviewed_business_categories
    # - sort
    # - limit
    type: str

    # For operations that move information from one dataset to another.
    # Example:
    # source_dataset = "review"
    # target_dataset = "business"
    source_dataset: Optional[str] = None
    target_dataset: Optional[str] = None

    # For operations that apply directly to one dataset (sort, limit).
    dataset: Optional[str] = None

    # Sort fields.
    by: Optional[List[str]] = None

    # Sort directions matching "by".
    # Example: [False, False] means descending for both fields.
    ascending: Optional[List[bool]] = None

    # Limit size, e.g. top-k recommendations.
    k: Optional[int] = None


class ResponseSpec(BaseModel):
    # Which dataset should be returned as the final response.
    dataset: str

    # Optional list of fields to keep in the response.
    # If omitted, all columns in the selected dataset are returned.
    fields: Optional[List[str]] = None


class QueryRequest(BaseModel):
    # Datasets requested by the caller.
    # Example:
    # {
    #   "business": {"filters": {...}},
    #   "review": {"filters": {...}}
    # }
    datasets: Dict[str, DatasetSpec]

    # Ordered list of operations to perform after datasets are loaded.
    operations: List[Operation] = Field(default_factory=list)

    # Final response shape.
    response: ResponseSpec


# ------------------------------------------------------------------------------
# Dataset loading helpers
# ------------------------------------------------------------------------------

def load_review_filtered(filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Load the review dataset in a streaming way and apply only a small set of
    filters while reading.

    Why this is special:
    - The Yelp review file is usually huge.
    - Loading the full review dataset into memory can be expensive.
    - So instead of pd.read_json(...) on the full file, we scan line-by-line
      and only keep matching rows.

    Supported filters here:
    - user_id
    - stars_gte
    - stars_lte

    Returns:
        A DataFrame with columns:
        review_id, user_id, business_id, stars, text
    """
    path = DATASET_PATHS["review"]
    rows = []

    # Pull supported filter values from the request.
    user_id = filters.get("user_id")
    stars_gte = filters.get("stars_gte")
    stars_lte = filters.get("stars_lte")

    # Yelp dataset uses one JSON object per line.
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            # If a specific user_id is requested, skip rows from other users.
            if user_id is not None and obj.get("user_id") != user_id:
                continue

            # Star-based filtering for user reviews.
            stars = obj.get("stars")
            if stars_gte is not None and stars < stars_gte:
                continue
            if stars_lte is not None and stars > stars_lte:
                continue

            # Keep only the fields needed downstream.
            rows.append({
                "review_id": obj.get("review_id"),
                "user_id": obj.get("user_id"),
                "business_id": obj.get("business_id"),
                "stars": obj.get("stars"),
                "text": obj.get("text", ""),
            })

    # Return a DataFrame with consistent column order.
    return pd.DataFrame(
        rows,
        columns=["review_id", "user_id", "business_id", "stars", "text"]
    )


def load_dataset(name: str) -> pd.DataFrame:
    """
    Load one of the configured Yelp datasets into memory.

    Behavior:
    - Validates dataset name
    - Uses in-memory cache when available
    - Restricts columns when DATASET_COLUMNS[name] is defined
    - Fills common text columns with empty strings to avoid NaN/string issues
    - Returns a copy so per-request operations do not mutate the cache
    """
    # Reject unknown dataset names early.
    if name not in DATASET_PATHS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {name}")

    # Load once, then reuse.
    if name not in DATA_CACHE:
        path = DATASET_PATHS[name]
        usecols = DATASET_COLUMNS.get(name)

        # Read JSON Lines file into a DataFrame.
        df = pd.read_json(path, lines=True)

        # If we only want a subset of columns, verify they exist first.
        if usecols is not None:
            missing = [c for c in usecols if c not in df.columns]
            if missing:
                raise HTTPException(
                    status_code=500,
                    detail=f"Missing expected columns in {name}: {missing}"
                )

            # Keep only the requested subset.
            df = df[usecols].copy()

        # Normalize selected text-like columns.
        # This prevents errors when calling string methods on NaN values.
        for col in ["city", "categories", "text", "name", "state"]:
            if col in df.columns:
                df[col] = df[col].fillna("")

        # Save in cache.
        DATA_CACHE[name] = df

    # Always return a copy so caller logic is isolated per request.
    return DATA_CACHE[name].copy()


# ------------------------------------------------------------------------------
# Generic filter engine
# ------------------------------------------------------------------------------

def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply a flexible set of filters to a DataFrame.

    Supported suffixes:
    - _gte       >=
    - _lte       <=
    - _gt        >
    - _lt        <
    - _in        membership
    - _contains  case-insensitive substring search
    - _any       matches ANY item in a list
    - _all       matches ALL items in a list

    If no suffix is used, exact equality is applied.

    Examples:
    - {"stars_gte": 4.0}
    - {"review_count_gte": 100}
    - {"city": "Las Vegas"}
    - {"categories_contains": "Pizza"}
    - {"categories_any": ["Italian", "Pizza"]}
    - {"categories_all": ["Restaurants", "Mexican"]}
    """
    out = df.copy()

    for key, value in filters.items():
        # Greater-than-or-equal filter:
        # stars_gte -> column "stars"
        if key.endswith("_gte"):
            col = key[:-4]
            out = out[out[col] >= value]

        # Less-than-or-equal filter:
        elif key.endswith("_lte"):
            col = key[:-4]
            out = out[out[col] <= value]

        # Strict greater-than filter.
        elif key.endswith("_gt"):
            col = key[:-3]
            out = out[out[col] > value]

        # Strict less-than filter.
        elif key.endswith("_lt"):
            col = key[:-3]
            out = out[out[col] < value]

        # Membership filter:
        # state_in -> value should be a list like ["NV", "AZ"]
        elif key.endswith("_in"):
            col = key[:-3]
            out = out[out[col].isin(value)]

        # Case-insensitive substring filter:
        # categories_contains = "Sushi"
        elif key.endswith("_contains"):
            col = key[:-9]
            out = out[
                out[col].astype(str).str.contains(str(value), case=False, na=False)
            ]

        # ANY-of list match:
        # categories_any = ["Thai", "Vietnamese"]
        # row is kept if at least one appears in the target column
        elif key.endswith("_any"):
            col = key[:-4]

            # Start with False, OR-in each condition.
            mask = False
            for item in value:
                mask = mask | out[col].astype(str).str.contains(
                    str(item), case=False, na=False
                )
            out = out[mask]

        # ALL-of list match:
        # categories_all = ["Restaurants", "Mexican"]
        # row is kept only if every item appears in the target column
        elif key.endswith("_all"):
            col = key[:-4]

            # Start with True, AND-in each condition.
            mask = True
            for item in value:
                mask = mask & out[col].astype(str).str.contains(
                    str(item), case=False, na=False
                )
            out = out[mask]

        else:
            # Default behavior = exact equality.
            # We validate the column exists to avoid silent failures.
            if key not in out.columns:
                raise HTTPException(status_code=400, detail=f"Unknown filter column: {key}")

            # For selected text fields, do case-insensitive exact matching.
            if isinstance(value, str) and key in {"city", "state", "name", "categories", "text"}:
                out = out[out[key].astype(str).str.lower() == value.strip().lower()]
            else:
                out = out[out[key] == value]

    return out


# ------------------------------------------------------------------------------
# Category/profile helpers
# ------------------------------------------------------------------------------

def split_categories(value: str) -> list[str]:
    """
    Split Yelp's category string into a normalized list.

    Yelp categories typically look like:
        "Restaurants, Italian, Pizza"

    This function returns:
        ["Restaurants", "Italian", "Pizza"]
    """
    if not value:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def build_user_category_profile(
    review_df: pd.DataFrame,
    business_df: pd.DataFrame
) -> dict[str, int]:
    """
    Build a simple user preference profile from the filtered review dataset for this request.

    How it works:
    1. Take business_ids from the user's filtered review dataset
    2. Find those businesses in the business dataset
    3. Count how often each category appears

    Example result:
        {
            "Restaurants": 12,
            "Mexican": 5,
            "Pizza": 3,
            "Bars": 2
        }

    This is later used to score candidate businesses by category overlap.
    """
    # If user has no reviews in the request context, no profile can be built.
    if review_df.empty:
        return {}

    # Unique set of business IDs reviewed by the user.
    liked_ids = set(review_df["business_id"].dropna().unique().tolist())
    if not liked_ids:
        return {}

    # Keep only businesses that the user has reviewed.
    liked_businesses = business_df[business_df["business_id"].isin(liked_ids)].copy()
    if liked_businesses.empty:
        return {}

    counts: dict[str, int] = {}

    # Count frequency of each category across reviewed businesses.
    for cats in liked_businesses["categories"].fillna(""):
        for cat in split_categories(cats):
            counts[cat] = counts.get(cat, 0) + 1

    return counts


# ------------------------------------------------------------------------------
# API endpoints
# ------------------------------------------------------------------------------

@app.get("/health")
def health():
    """
    Lightweight health check endpoint.

    Useful for:
    - container readiness checks
    - reverse proxy health monitoring
    - verifying the API is up
    """
    return {"status": "ok"}


@app.post("/recommend")
def recommend(payload: QueryRequest):
    """
    Main recommendation endpoint.

    Flow:
    1. Load requested datasets
    2. Apply per-dataset filters
    3. Execute operations in order
    4. Select final response dataset/fields
    5. Return results + debug info

    The logic is intentionally generic so the caller can compose a pipeline
    without hardcoding a single recommendation strategy into the endpoint.
    """
    try:
        # Holds all working DataFrames for this request.
        # Key = dataset name, value = filtered/derived DataFrame.
        frames: Dict[str, pd.DataFrame] = {}

        # ------------------------------------------------------------------
        # Step 1: Load + filter requested datasets
        # ------------------------------------------------------------------
        for dataset_name, spec in payload.datasets.items():
            # Reviews are handled specially because the source file is large
            # and we want to stream-filter it instead of loading all of it.
            if dataset_name == "review":
                df = load_review_filtered(spec.filters)

            else:
                # Load dataset from cache/disk.
                df = load_dataset(dataset_name)

                # Business dataset is automatically restricted to restaurants only.
                # This means the recommender ignores non-restaurant businesses.
                if dataset_name == "business" and "categories" in df.columns:
                    df = df[
                        df["categories"].astype(str).str.contains(
                            "Restaurant",
                            case=False,
                            na=False
                        )
                    ]

                # Apply generic filters from request.
                df = apply_filters(df, spec.filters)

            # Store the working DataFrame for later operations.
            frames[dataset_name] = df

        # ------------------------------------------------------------------
        # Step 2: Execute requested operations in order
        # ------------------------------------------------------------------
        for op in payload.operations:
            # --------------------------------------------------------------
            # Operation: exclude businesses the user already reviewed
            # --------------------------------------------------------------
            if op.type == "exclude_businesses_reviewed_by_user":
                # This operation needs:
                # - source_dataset: usually "review"
                # - target_dataset: usually "business"
                if not op.source_dataset or not op.target_dataset:
                    raise HTTPException(
                        status_code=400,
                        detail="exclude_businesses_reviewed_by_user requires source_dataset and target_dataset"
                    )

                source = frames[op.source_dataset]
                target = frames[op.target_dataset]

                # Both datasets must have business_id for exclusion to work.
                if "business_id" not in source.columns or "business_id" not in target.columns:
                    raise HTTPException(
                        status_code=400,
                        detail="Both datasets must contain business_id"
                    )

                # Build a set of business IDs already reviewed by the user.
                reviewed_ids = set(source["business_id"].dropna().unique().tolist())

                # Remove those businesses from the candidate target set.
                target = target[~target["business_id"].isin(reviewed_ids)].copy()

                # Save updated target back into frames.
                frames[op.target_dataset] = target

            # --------------------------------------------------------------
            # Operation: boost candidates by category overlap with user's
            # previously reviewed businesses
            # --------------------------------------------------------------
            elif op.type == "boost_by_reviewed_business_categories":
                if not op.source_dataset or not op.target_dataset:
                    raise HTTPException(
                        status_code=400,
                        detail="boost_by_reviewed_business_categories requires source_dataset and target_dataset"
                    )

                # This operation depends on the business dataset being available,
                # because we need category info for reviewed businesses.
                if "business" not in frames:
                    raise HTTPException(
                        status_code=400,
                        detail="boost_by_reviewed_business_categories requires business dataset to be loaded"
                    )

                source = frames[op.source_dataset]
                target = frames[op.target_dataset]

                # Load the full business dataset from cache so we can build
                # the category profile using reliable business/category mappings.
                full_businesses = load_dataset("business")

                # Keep only restaurants here as well, for consistency.
                if "categories" in full_businesses.columns:
                    full_businesses = full_businesses[
                        full_businesses["categories"].astype(str).str.contains(
                            "Restaurant",
                            case=False,
                            na=False
                        )
                    ].copy()

                # Build user category preference counts from reviewed businesses.
                profile = build_user_category_profile(source, full_businesses)

                # Score one candidate business by summing the user's counts
                # for each category found in that business.
                #
                # Example:
                # profile = {"Pizza": 4, "Italian": 2}
                # business categories = ["Restaurants", "Pizza", "Italian"]
                # score = 4 + 2 = 6
                def category_score(cat_text: str) -> int:
                    score = 0
                    for cat in split_categories(cat_text):
                        score += profile.get(cat, 0)
                    return score

                # Add a derived score column to the target candidates.
                target = target.copy()
                target["category_score"] = target["categories"].fillna("").apply(category_score)

                frames[op.target_dataset] = target

            # --------------------------------------------------------------
            # Operation: sort dataset
            # --------------------------------------------------------------
            elif op.type == "sort":
                if not op.dataset or not op.by:
                    raise HTTPException(
                        status_code=400,
                        detail="sort requires dataset and by"
                    )

                # If caller does not specify ascending flags,
                # default all sort columns to ascending=True.
                asc = op.ascending if op.ascending is not None else [True] * len(op.by)

                frames[op.dataset] = frames[op.dataset].sort_values(
                    by=op.by,
                    ascending=asc
                )

            # --------------------------------------------------------------
            # Operation: limit dataset to top-k rows
            # --------------------------------------------------------------
            elif op.type == "limit":
                if not op.dataset or op.k is None:
                    raise HTTPException(
                        status_code=400,
                        detail="limit requires dataset and k"
                    )

                # Keep only the first k rows after prior operations
                # such as sorting.
                frames[op.dataset] = frames[op.dataset].head(op.k).copy()

            # --------------------------------------------------------------
            # Unknown operation
            # --------------------------------------------------------------
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown operation type: {op.type}"
                )

        # ------------------------------------------------------------------
        # Step 3: Build final response dataset
        # ------------------------------------------------------------------
        response_dataset = payload.response.dataset

        # Ensure the requested output dataset exists in our working frames.
        if response_dataset not in frames:
            raise HTTPException(
                status_code=400,
                detail=f"Response dataset not loaded: {response_dataset}"
            )

        result_df = frames[response_dataset]

        # If the caller requested only specific columns, validate them first.
        if payload.response.fields:
            missing = [f for f in payload.response.fields if f not in result_df.columns]
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"Response fields not found: {missing}"
                )

            result_df = result_df[payload.response.fields].copy()

        # Useful lightweight debug info:
        # how many rows are in each dataset after all processing.
        debug = {
            "loaded_datasets": {name: int(len(df)) for name, df in frames.items()}
        }

        # Final API response.
        return {
            "status": "ok",
            "row_count": int(len(result_df)),
            "debug": debug,
            "result": result_df.to_dict(orient="records"),
        }

    # Let FastAPI keep explicit HTTPExceptions as-is.
    except HTTPException:
        raise

    # Convert any unexpected Python error into a 500 API error.
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))