"""Shared helper functions used across routers."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from api.schemas import ClinicalSummary, DemographicSummary
from api.state import AppState, get_state

# Columns included when returning patient member details
MEMBER_COLS = [
    "age", "gender", "status", "profession_category", "age_group", "city",
    "depression", "suicidal_thoughts", "family_history", "high_risk",
    "stress_index", "wellbeing_score", "worklife_balance", "emotional_score",
    "pressure", "satisfaction", "financial_stress", "sleep_duration",
    "workstudy_hours", "dietary_habits", "education_level",
]


def safe_float(v) -> float | None:
    """Convert value to Python float, mapping NaN/inf/None → None."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _pct(df: pd.DataFrame, col: str) -> float:
    """Mean of a binary column × 100, or 0.0 if column absent."""
    if col not in df.columns:
        return 0.0
    v = df[col].mean()
    return round(float(v) * 100, 2) if not math.isnan(float(v)) else 0.0


def _avg(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    v = df[col].mean()
    return round(float(v), 4) if not math.isnan(float(v)) else 0.0


def clinical_summary(df: pd.DataFrame) -> ClinicalSummary:
    return ClinicalSummary(
        pct_high_risk=_pct(df, "high_risk"),
        pct_depressed=_pct(df, "depression"),
        pct_suicidal=_pct(df, "suicidal_thoughts"),
        mean_stress_index=_avg(df, "stress_index"),
        mean_wellbeing_score=_avg(df, "wellbeing_score"),
        mean_pressure=_avg(df, "pressure"),
        mean_satisfaction=_avg(df, "satisfaction"),
        mean_financial_stress=_avg(df, "financial_stress"),
    )


def demographic_summary(df: pd.DataFrame) -> DemographicSummary:
    ages = df["age"].dropna() if "age" in df.columns else pd.Series([], dtype=float)
    return DemographicSummary(
        pct_professional=_pct(df, "is_professional"),
        pct_male=_pct(df, "gender_enc"),
        age_min=int(ages.min()) if len(ages) else 0,
        age_max=int(ages.max()) if len(ages) else 0,
        age_mean=round(float(ages.mean()), 2) if len(ages) else 0.0,
    )


def cluster_wcss(members_vecs: pd.DataFrame) -> tuple[float, float]:
    """Return (wcss, mean_dist_to_centroid) from a cluster's vectorised rows."""
    num_cols = members_vecs.select_dtypes(include="number").columns
    data = members_vecs[num_cols].fillna(0).values
    if len(data) == 0:
        return 0.0, 0.0
    centroid = data.mean(axis=0)
    diffs = data - centroid
    wcss = float(np.sum(diffs ** 2))
    mean_dist = float(np.linalg.norm(diffs, axis=1).mean()) if len(diffs) else 0.0
    return wcss, mean_dist


def serialise_patient(patient_id: Any, row: pd.Series) -> dict[str, Any]:
    """Convert a patient_data row to a JSON-safe dict with display columns."""
    out: dict[str, Any] = {"patient_id": patient_id}
    for col in MEMBER_COLS:
        if col not in row.index:
            continue
        v = row[col]
        if isinstance(v, float) and math.isnan(v):
            out[col] = None
        elif isinstance(v, (np.integer,)):
            out[col] = int(v)
        elif isinstance(v, (np.floating,)):
            out[col] = float(v) if not math.isnan(float(v)) else None
        elif isinstance(v, (np.bool_,)):
            out[col] = bool(v)
        elif hasattr(v, "item"):          # catch-all for numpy scalars
            out[col] = v.item()
        else:
            out[col] = v
    return out


def get_members_raw(group_id: int, state: AppState) -> pd.DataFrame:
    """Return raw patient_data rows for a given cluster."""
    member_ids = list(state.model.clusters_[group_id].index)
    available = state.patient_data.index.intersection(member_ids)
    return state.patient_data.loc[available]


def require_initialized() -> AppState:
    """Raises 503 if the model has not been initialised yet."""
    from fastapi import HTTPException
    state = get_state()
    if state.model is None:
        raise HTTPException(503, "Model not initialized. POST /initialize first.")
    return state
