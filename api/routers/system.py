"""
/initialize  — fit model on uploaded CSV
/status      — overall health check
/metrics     — system-wide clustering quality metrics
"""

from __future__ import annotations

import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy.sparse as sp
from fastapi import APIRouter, File, HTTPException, UploadFile

from modeling.agglomerative_clustering import AggloGroupModel
from modeling.evaluate import Evaluator
from modeling.vectorizer import FeatureConfig, Vectorizer
from api.schemas import InitializeResponse, SystemMetrics, SystemStatus
from api.state import get_state
from api.utils import cluster_wcss, safe_float

router = APIRouter(tags=["system"])

NUMERIC_FEATURES = [
    "age",
    "sleep_duration",
    "suicidal_thoughts",
    "workstudy_hours",
    "financial_stress",
    "family_history",
    "depression",
    "pressure",
    "satisfaction",
    "gender_enc",
    "is_professional",
    "dietary_enc",
    "education_level",
    "city_lat",
    "city_lon",
    "stress_index",
    "wellbeing_score",
    "worklife_balance",
    "high_risk",
]
CATEGORICAL_FEATURES = ["city", "profession_category", "age_group", "cgpa_band"]
BOOLEAN_FEATURES: list[str] = []

# Clinical feature weights (numeric__ prefix = post-ColumnTransformer column names)
CLINICAL_WEIGHTS: dict[str, float] = {
    "numeric__high_risk": 3.0,
    "numeric__depression": 2.5,
    "numeric__suicidal_thoughts": 2.0,
    "numeric__stress_index": 2.0,
    "numeric__financial_stress": 1.5,
    "numeric__wellbeing_score": 1.5,
    "numeric__pressure": 1.5,
    "numeric__family_history": 1.5,
}


def _build_attribute_connectivity(init_feats: pd.DataFrame) -> sp.csr_matrix:
    """
    Sparse connectivity matrix: two patients are connected if they share
    certain attributes.
    """
    connectivity_attrs = [
        "is_professional",
        "age_group",
        "suicidal_thoughts",
        "unhealthy_diet",
        "depression",
    ]
    feats = init_feats[connectivity_attrs].reset_index(drop=True).copy()
    feats["_pos"] = np.arange(len(feats))

    pairs = feats.merge(feats, on=connectivity_attrs, suffixes=("_i", "_j"))
    pairs = pairs[pairs["_pos_i"] != pairs["_pos_j"]]

    rows = pairs["_pos_i"].values
    cols = pairs["_pos_j"].values
    n = len(feats)
    return sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)
    )


@router.post("/initialize", response_model=InitializeResponse, status_code=201)
async def initialize(file: UploadFile = File(...)):
    """
    Fit the Clinical + Attribute-Matched Connectivity Agglomerative model on a
    CSV upload.  The CSV must have the same column structure as `data/data.csv`.
    Re-initializing replaces all existing groups.
    """
    state = get_state()
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(400, f"Could not parse CSV: {exc}")

    try:
        config = FeatureConfig(NUMERIC_FEATURES, CATEGORICAL_FEATURES, BOOLEAN_FEATURES)
        vec = Vectorizer(config)
        vec.fit(df)
        init_vecs: pd.DataFrame = vec.transform(df)

        # Cleaned (pre-vectorisation) features — kept for human-readable endpoints
        init_feats: pd.DataFrame = vec.pipeline.steps[0][1].transform(df)

        connectivity = _build_attribute_connectivity(init_feats)

        model = AggloGroupModel(group_size=12, connectivity=connectivity)
        model.fit(init_vecs)
        model.predict(init_vecs)

        evaluator = Evaluator(model)
    except Exception as exc:
        raise HTTPException(422, f"Model fitting failed: {exc}")

    # Persist state
    state.vectorizer = vec
    state.model = model
    state.evaluator = evaluator
    state.all_vecs = init_vecs.copy()
    state.patient_data = init_feats.copy()
    state.initialized_at = datetime.now(timezone.utc)
    state.n_initial_patients = len(init_vecs)

    stats = evaluator.group_size_stats()
    wcss_total, _ = evaluator.wcss(init_vecs)
    silhouette = evaluator.silhouette(init_vecs)

    return InitializeResponse(
        n_patients=len(init_vecs),
        n_groups=stats["n_groups"],
        group_stats=stats,
        silhouette=round(silhouette, 6),
        wcss_total=round(wcss_total, 4),
        initialized_at=state.initialized_at.isoformat(),
    )


@router.get("/status", response_model=SystemStatus)
def status():
    """Overall system health: initialization state and top-level metrics."""
    state = get_state()
    if state.model is None:
        return SystemStatus(initialized=False)

    stats = state.evaluator.group_size_stats()
    n_patients = sum(len(m) for m in state.model.clusters_.values())
    all_vecs = pd.concat(list(state.model.clusters_.values()))
    wcss_total, _ = state.evaluator.wcss(all_vecs)
    silhouette = state.evaluator.silhouette(all_vecs)
    n_groups = stats["n_groups"]

    return SystemStatus(
        initialized=True,
        initialized_at=state.initialized_at.isoformat()
        if state.initialized_at
        else None,
        n_patients=n_patients,
        n_groups=n_groups,
        group_stats=stats,
        silhouette=round(silhouette, 6),
        wcss_total=round(wcss_total, 4),
        wcss_mean_per_group=round(wcss_total / n_groups, 4) if n_groups else None,
    )


@router.get("/metrics", response_model=SystemMetrics)
def metrics():
    """
    System-wide clustering quality metrics.
    Computed over all current patients (including newcomers).
    """
    state = get_state()
    if state.model is None:
        raise HTTPException(503, "Model not initialized. POST /initialize first.")

    all_vecs = pd.concat(list(state.model.clusters_.values()))
    wcss_total, _ = state.evaluator.wcss(all_vecs)
    silhouette = state.evaluator.silhouette(all_vecs)
    stats = state.evaluator.group_size_stats()
    n_groups = stats["n_groups"]

    return SystemMetrics(
        silhouette=round(silhouette, 6),
        wcss_total=round(wcss_total, 4),
        wcss_mean_per_group=round(wcss_total / n_groups, 4) if n_groups else 0.0,
        group_stats=stats,
    )
