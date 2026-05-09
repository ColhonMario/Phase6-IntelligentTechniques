import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_SCHEMA = {
    "required_inputs": ["amt", "trans_date_trans_time", "category", "cc_num"],
    "optional_inputs": ["merchant", "zip", "lat", "long", "merch_lat", "merch_long"],
    "raw_input_importance": {
        # Stand-in importance values — replaced by notebook output in prod
        "amt": 0.245,
        "trans_date_trans_time": 0.180,
        "category": 0.150,
        "cc_num": 0.135,
        "merchant": 0.080,
        "zip": 0.060,
        "lat": 0.040, "long": 0.040,
        "merch_lat": 0.035, "merch_long": 0.035,
    },
}

SAFE_DEFAULTS = {
    "merchant": "__unknown__",   # label encoder maps unseen strings to classes_[0]
    "zip": "00000",              # treated as a rare ZIP -> freq=1
    "lat": 39.8283,              # geographic centre of the continental US
    "long": -98.5795,
    "merch_lat": 39.8283,
    "merch_long": -98.5795,
    "cc_num": 0,                 # placeholder cardholder ID -> freq=1
}

PII_STUBS = {
    "first": "_", "last": "_", "street": "_", "city": "_", "state": "_",
    "gender": "M", "dob": "1980-01-01", "job": "_", "trans_num": "_",
    "is_fraud": 0,  
}


def load_schema(artifact_dir: Optional[Path] = None) -> dict:
    """Load input schema from disk if available, otherwise return DEFAULT_SCHEMA."""
    if artifact_dir is not None:
        path = Path(artifact_dir) / "input_schema.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return DEFAULT_SCHEMA


def load_app_config(artifact_dir: Optional[Path] = None) -> dict:
    """Load app config (real categories from Sparkov, realistic scenarios).
    Returns empty fallback if file is missing.
    """
    fallback = {
        "real_categories": [
            "grocery_pos", "grocery_net", "gas_transport", "misc_pos", "misc_net",
            "shopping_pos", "shopping_net", "entertainment", "food_dining",
            "health_fitness", "home", "kids_pets", "personal_care", "travel",
        ],
        "scenarios": {},
        "top_merchants": [],
    }
    if artifact_dir is not None:
        path = Path(artifact_dir) / "app_config.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return fallback


# =============================================================================
# Helpers
# =============================================================================
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Vectorized over scalars or arrays."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2 +
         np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def is_blank(value) -> bool:
    """True if value should be considered missing (None, NaN, empty string)."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


# =============================================================================
# Validation
# =============================================================================
US_BBOX = (24.0, 49.0, -125.0, -66.0)  # lat_min, lat_max, lon_min, lon_max


def validate_inputs(form_inputs: dict, schema: dict) -> tuple[list[str], list[str]]:
    """Check required fields, return (errors, warnings).
    Errors block prediction. Warnings are informational.
    """
    errors, warnings = [], []
    required = schema["required_inputs"]
    optional = schema["optional_inputs"]

    # Required-field checks
    for field in required:
        val = form_inputs.get(field)
        if is_blank(val):
            errors.append(f"Required field '{field}' is empty")
            continue

        if field == "amt":
            if not isinstance(val, (int, float)) or val <= 0:
                errors.append(f"Amount must be a positive number (got {val})")
            elif val > 50000:
                warnings.append(f"Amount ${val:,.2f} is very large — verify it's correct")

        if field == "trans_date_trans_time":
            try:
                ts = pd.to_datetime(val)
                if ts.year < 2018 or ts.year > 2026:
                    warnings.append(
                        f"Year {ts.year} is outside the model's training window "
                        f"(2019–2020). Predictions may be less reliable."
                    )
            except Exception:
                errors.append(
                    f"Could not parse date '{val}'. Expected format: YYYY-MM-DD HH:MM:SS"
                )

    # Optional-field warnings
    n_imputed = sum(1 for f in optional if is_blank(form_inputs.get(f)))
    if n_imputed > 0:
        pct_importance = sum(schema["raw_input_importance"].get(f, 0)
                              for f in optional if is_blank(form_inputs.get(f)))
        warnings.append(
            f"{n_imputed} optional field(s) are empty and will be filled with safe "
            f"defaults. These fields account for ~{pct_importance*100:.0f}% of model importance."
        )

    # Coordinate bounds (only if all four are present and numeric)
    coord_fields = ["lat", "long", "merch_lat", "merch_long"]
    if all(not is_blank(form_inputs.get(f)) for f in coord_fields):
        lat, lon = form_inputs["lat"], form_inputs["long"]
        mlat, mlon = form_inputs["merch_lat"], form_inputs["merch_long"]

        if not (-90 <= lat <= 90):
            errors.append(f"Cardholder latitude {lat} invalid (must be -90 to 90)")
        if not (-180 <= lon <= 180):
            errors.append(f"Cardholder longitude {lon} invalid (must be -180 to 180)")
        if not (-90 <= mlat <= 90):
            errors.append(f"Merchant latitude {mlat} invalid (must be -90 to 90)")
        if not (-180 <= mlon <= 180):
            errors.append(f"Merchant longitude {mlon} invalid (must be -180 to 180)")

        if not errors:
            in_us = (US_BBOX[0] <= lat <= US_BBOX[1] and US_BBOX[2] <= lon <= US_BBOX[3])
            if not in_us:
                warnings.append(
                    "Cardholder coordinates are outside the continental US. "
                    "The model was only trained on US transactions."
                )

    return errors, warnings


# =============================================================================
# Build a Sparkov-shaped transaction dict for the pipeline
# =============================================================================
def build_transaction_dict(form_inputs: dict, schema: dict) -> tuple[dict, list[str]]:
    """Construct a Sparkov-schema row, filling in safe defaults for missing optionals.

    Returns (transaction_dict, list_of_imputed_fields).
    """
    transaction = {}
    imputed_fields = []

    # Required fields — caller should have validated these already
    for field in schema["required_inputs"]:
        transaction[field] = form_inputs[field]

    # Optional fields — fill safe defaults if missing
    for field in schema["optional_inputs"]:
        val = form_inputs.get(field)
        if is_blank(val):
            transaction[field] = SAFE_DEFAULTS.get(field, np.nan)
            imputed_fields.append(field)
        else:
            transaction[field] = val

    # PII stubs (the adapter drops them; we just need the keys to exist)
    transaction.update(PII_STUBS)
    return transaction, imputed_fields


# =============================================================================
# Batch processing
# =============================================================================
SPARKOV_REQUIRED_BATCH_COLS = ["trans_date_trans_time", "amt", "category"]
SPARKOV_OPTIONAL_BATCH_COLS = ["cc_num", "merchant", "zip",
                                "lat", "long", "merch_lat", "merch_long"]


def validate_batch_csv(df: pd.DataFrame) -> tuple[list[str], list[str], dict]:
    """Inspect an uploaded CSV. Returns (errors, warnings, quality_stats)."""
    errors, warnings = [], []

    missing_required = [c for c in SPARKOV_REQUIRED_BATCH_COLS if c not in df.columns]
    missing_optional = [c for c in SPARKOV_OPTIONAL_BATCH_COLS if c not in df.columns]

    if missing_required:
        errors.append(
            f"CSV is missing required columns: {missing_required}. "
            f"Expected at minimum: {SPARKOV_REQUIRED_BATCH_COLS}"
        )

    if missing_optional:
        warnings.append(
            f"Optional columns missing (will use defaults): {missing_optional}"
        )

    # Per-column missingness
    n_rows = len(df)
    quality_stats = {"total_rows": n_rows, "missing_per_col": {}}
    for col in SPARKOV_REQUIRED_BATCH_COLS + SPARKOV_OPTIONAL_BATCH_COLS:
        if col in df.columns:
            n_missing = df[col].isnull().sum()
            quality_stats["missing_per_col"][col] = int(n_missing)
            if n_missing > 0:
                pct = n_missing / n_rows * 100
                warnings.append(f"Column '{col}': {n_missing:,} missing values ({pct:.1f}%)")

    return errors, warnings, quality_stats


def impute_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values + missing columns with safe defaults so the pipeline runs."""
    df = df.copy()

    # Mark which rows had any missing essentials before imputation (for reporting)
    df["had_missing_fields"] = df[
        [c for c in SPARKOV_REQUIRED_BATCH_COLS + SPARKOV_OPTIONAL_BATCH_COLS
         if c in df.columns]
    ].isnull().any(axis=1).astype(int)

    # Required-column imputation (these MUST exist by validation)
    if "trans_date_trans_time" in df.columns:
        df["trans_date_trans_time"] = df["trans_date_trans_time"].fillna("2020-06-15 12:00:00")
    if "amt" in df.columns:
        median_amt = df["amt"].median() if df["amt"].notna().any() else 50.0
        df["amt"] = df["amt"].fillna(median_amt)
    if "category" in df.columns:
        most_common = df["category"].mode().iloc[0] if df["category"].notna().any() else "misc_pos"
        df["category"] = df["category"].fillna(most_common)

    # Optional-column imputation
    for col, default in SAFE_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)

    # PII stubs needed by adapter — fill in
    for col, default in PII_STUBS.items():
        if col not in df.columns:
            df[col] = default

    return df


def batch_descriptive_stats(results: pd.DataFrame, threshold: float) -> dict:
    """Compute summary statistics for a batch result table."""
    n_total = len(results)
    n_fraud = int(results["predicted_fraud"].sum())
    n_legit = n_total - n_fraud

    flagged_amount = float(results.loc[results["predicted_fraud"] == 1, "amt"].sum())
    total_amount = float(results["amt"].sum())

    stats = {
        "n_total": n_total,
        "n_fraud": n_fraud,
        "n_legit": n_legit,
        "fraud_rate": n_fraud / n_total if n_total else 0.0,
        "flagged_amount": flagged_amount,
        "total_amount": total_amount,
        "flagged_amount_pct": flagged_amount / total_amount if total_amount else 0.0,
        "mean_proba": float(results["fraud_probability"].mean()),
        "median_proba": float(results["fraud_probability"].median()),
        "max_proba": float(results["fraud_probability"].max()),
    }

    # Recall on labels if ground truth is provided
    if "is_fraud" in results.columns:
        actual = results["is_fraud"]
        tp = int(((results["predicted_fraud"] == 1) & (actual == 1)).sum())
        fn = int(((results["predicted_fraud"] == 0) & (actual == 1)).sum())
        fp = int(((results["predicted_fraud"] == 1) & (actual == 0)).sum())
        n_actual_fraud = int(actual.sum())
        stats.update({
            "has_labels": True,
            "n_actual_fraud": n_actual_fraud,
            "tp": tp, "fn": fn, "fp": fp,
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        })
    else:
        stats["has_labels"] = False

    return stats