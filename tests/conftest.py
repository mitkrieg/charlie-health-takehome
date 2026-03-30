"""
Shared fixtures for all test modules.

Provides small synthetic DataFrames that bypass the heavy PatientDataTransformer
(geocoding, etc.) so unit tests are fast and deterministic.
"""
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Vectorised feature DataFrames (post-ColumnTransformer shape)
# ---------------------------------------------------------------------------

@pytest.fixture()
def make_vecs():
    """Factory that returns a (n × 4) numeric DataFrame with patient_id index."""
    def _make(n: int = 48, seed: int = 42) -> pd.DataFrame:
        rng = np.random.RandomState(seed)
        df = pd.DataFrame(
            rng.randn(n, 4),
            columns=["numeric__a", "numeric__b", "numeric__c", "numeric__d"],
        )
        df.index.name = "patient_id"
        return df
    return _make


@pytest.fixture()
def vecs_48(make_vecs):
    """48-row vectorised DataFrame (fits 4 groups of 12)."""
    return make_vecs(48)


@pytest.fixture()
def vecs_50(make_vecs):
    """50-row vectorised DataFrame (non-divisible by group_size)."""
    return make_vecs(50)


# ---------------------------------------------------------------------------
# Raw / cleaned feature DataFrames (pre-vectorisation shape)
# ---------------------------------------------------------------------------

@pytest.fixture()
def raw_feats_48():
    """48-row cleaned-feature DataFrame mimicking PatientDataTransformer output."""
    rng = np.random.RandomState(99)
    n = 48
    df = pd.DataFrame({
        "age": rng.randint(18, 65, n),
        "gender": rng.choice(["Male", "Female"], n),
        "status": rng.choice(["Student", "Working Professional"], n),
        "profession_category": rng.choice(["Technology", "Healthcare", "Education"], n),
        "age_group": pd.Categorical(rng.choice(["≤22", "23-30", "31-40", "41-50", "51+"], n)),
        "city": rng.choice(["Mumbai", "Delhi", "Chennai"], n),
        "depression": rng.choice([0, 1], n),
        "suicidal_thoughts": rng.choice([0, 1], n),
        "family_history": rng.choice([0, 1], n),
        "high_risk": rng.choice([0, 1], n),
        "stress_index": rng.uniform(1, 5, n).round(2),
        "wellbeing_score": rng.uniform(1, 5, n).round(2),
        "worklife_balance": rng.uniform(0, 1, n).round(2),
        "emotional_score": rng.uniform(1, 5, n).round(2),
        "pressure": rng.uniform(1, 5, n).round(1),
        "satisfaction": rng.uniform(1, 5, n).round(1),
        "financial_stress": rng.randint(1, 6, n),
        "sleep_duration": rng.randint(1, 5, n),
        "workstudy_hours": rng.randint(0, 12, n),
        "dietary_habits": rng.choice(["Healthy", "Moderate", "Unhealthy"], n),
        "education_level": rng.choice([0, 1, 2, 3], n),
        "is_professional": rng.choice([0, 1], n),
        "gender_enc": rng.choice([0, 1], n),
        "dietary_enc": rng.choice([0, 1, 2], n),
        "treatment_not_needed": rng.choice([True, False], n),
    })
    df.index.name = "patient_id"
    return df
