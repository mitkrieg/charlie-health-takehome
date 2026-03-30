# Charlie Health Takehome

Group ~2,172 mental health patients into treatment cohorts of ~12. Compares a random baseline against agglomerative clustering with clinical feature weights and demographic connectivity constraints. Includes a FastAPI service that wraps the fitted model for production use.

---

## Table of Contents

1. [Setup](#setup)
2. [Common Commands](#common-commands)
3. [Repository Layout](#repository-layout)
4. [Pipeline Overview](#pipeline-overview)
5. [Modeling](#modeling)
6. [Evaluation](#evaluation)
7. [API](#api)
8. [Notebooks](#notebooks)
9. [Tests](#tests)
10. [Feature Pipeline Reference](#feature-pipeline-reference)

---

## Setup

Requires Python 3.13.1.

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

**Dependencies:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `geopy`, `fastapi`, `uvicorn`, `python-multipart`, `httpx`, `pytest`

---

## Common Commands

```bash
# Re-run data cleaning pipeline → writes data/data_clean.csv
python prep/data_cleaning.py

# Generate EDA plots → writes 14 PNGs to eda_plots/
python prep/eda.py

# Run all tests
pytest

# Modeling tests only (skips geocoding, much faster)
pytest tests/test_modeling.py

# Single test
pytest tests/test_api.py::TestReadOnly::test_metrics

# Launch notebooks
jupyter notebook modeling.ipynb   # main analysis
jupyter notebook eda.ipynb        # inline EDA plots

# Start API server
uvicorn api.main:app --reload
# Interactive docs: http://localhost:8000/docs

# Run the full API workflow example
python examples/api_workflow.py
python examples/api_workflow.py --self-test   # in-process, no uvicorn needed
```

---

## Repository Layout

```
charlie-takehome/
├── api/
│   ├── main.py               # FastAPI app entry point
│   ├── schemas.py            # Pydantic request/response models
│   ├── state.py              # Global AppState singleton
│   ├── utils.py              # Shared helpers (summaries, WCSS, serialisation)
│   └── routers/
│       ├── system.py         # POST /initialize, GET /status, GET /metrics
│       ├── groups.py         # Group CRUD + split/similar endpoints
│       └── patients.py       # Patient lookup, removal, newcomer assignment
├── data/
│   ├── data.csv              # Raw survey data (~2,172 rows)
│   ├── data_clean.csv        # Cleaned output from data_cleaning.py
│   └── geocoding_cache.csv   # Cached city → (lat, lon) mappings
├── examples/
│   └── api_workflow.py       # End-to-end API usage example
├── modeling/
│   ├── agglomerative_clustering.py  # AggloGroupModel
│   ├── baseline.py                  # BaselineGroupModel (random assignment)
│   ├── evaluate.py                  # Evaluator (metrics, plots, drift analysis)
│   └── vectorizer.py                # Vectorizer + FeatureConfig
├── prep/
│   ├── data_cleaning.py      # PatientDataTransformer
│   └── eda.py                # EDA plot generation
├── tests/
│   ├── conftest.py           # Module-scoped fixtures, rate-limited geocoding
│   ├── test_modeling.py      # Model + evaluator unit tests
│   └── test_api.py           # API integration tests
├── eda_plots/                # 14 PNG visualisations from prep/eda.py
├── feature_pipeline.md       # Full field-level feature engineering spec
├── modeling.ipynb            # Main analysis notebook
└── eda.ipynb                 # EDA notebook (inline plots)
```

---

## Pipeline Overview

```
data/data.csv
    └─► PatientDataTransformer    (clean, impute, encode, geocode, engineer)
            └─► Vectorizer / ColumnTransformer   (scale, one-hot, passthrough)
                    └─► init_vecs  (numeric__*, categorical__*, boolean__*)
                            └─► AggloGroupModel / BaselineGroupModel
                                    └─► Evaluator
```

---

## Modeling

### `modeling/baseline.py` — `BaselineGroupModel`

Random shuffled assignment into fixed-size groups. Used as the comparison baseline. Supports online assignment of new patients to any group with remaining capacity; splits the least-loaded group if all are full.

### `modeling/agglomerative_clustering.py` — `AggloGroupModel`

Extends `GroupModel`. Key behaviours:

- **Clustering:** Fits `sklearn.AgglomerativeClustering` with `n_clusters = ceil(n / group_size)`.
- **Feature weights:** Accepts `feature_weights: dict[str, float]`. Weights are applied by scaling the transformed matrix column-wise before clustering (so clinical features like `high_risk` and `depression` influence distances more heavily).
- **Connectivity:** Accepts a sparse connectivity matrix. In the API and final notebook model, this is built so only patients sharing `is_professional × age_group × gender_enc` are eligible for direct merging.
- **Hard cap enforcement:** After initial clustering, any cluster exceeding `group_size` is recursively split via a 2-cluster `AgglomerativeClustering` sub-call.
- **Online assignment (`assign_cluster`):** New patients are placed in the nearest cluster (by weighted Euclidean distance to centroid) that still has capacity. If all clusters are full, the nearest cluster is split and the patient goes to the new half.
- **Deny-treatment routing (`separate_not_needed=True`):** When enabled, `predict(X, deny_mask=...)` separates patients whose `treatment_not_needed=1` into a reserved cluster `−1` before clustering the rest, so metrics like WCSS and silhouette aren't contaminated. `assign_cluster(X, deny_treatment=True)` routes newcomers directly to cluster `−1`. Cluster `−1` is hidden from the nearest-cluster search for normal patients.

**Clinical weights used in the API / final model:**

| Feature | Weight |
|---|---|
| `numeric__high_risk` | 3.0 |
| `numeric__depression` | 2.5 |
| `numeric__suicidal_thoughts` | 2.0 |
| `numeric__stress_index` | 2.0 |
| `numeric__financial_stress` | 1.5 |
| `numeric__wellbeing_score` | 1.5 |
| `numeric__pressure` | 1.5 |
| `numeric__family_history` | 1.5 |

### `modeling/vectorizer.py` — `Vectorizer` + `FeatureConfig`

Wraps `PatientDataTransformer` and a `ColumnTransformer` into a single sklearn `Pipeline`. `FeatureConfig` is a dataclass holding the three feature lists. The pipeline's step 0 (`preprocess`) is the `PatientDataTransformer`; step 1 (`column_transform`) is the `ColumnTransformer`.

Output columns are prefixed: `numeric__`, `categorical__`, `boolean__`.

---

## Evaluation

### `modeling/evaluate.py` — `Evaluator`

All metrics and plots exclude cluster `−1` (deny-treatment group) automatically.

**Scalar metrics:**
- `wcss(X)` → `(total_wcss, per_cluster_dict)` — Within-cluster sum of squares
- `silhouette(X)` — sklearn silhouette score (filters out label `−1`)
- `group_size_stats(X_raw=None)` → dict with `n_groups`, `min`, `max`, `mean`, `std`, `cv`. When `X_raw` (pre-vectorisation features) is passed, adds cohesion percentages: uniform gender/status/city/diet/profession/depression/suicidal/high_risk, age range ≤10yr, treatment-not-needed uniformity.
- `report(X, X_raw=None)` — Human-readable summary string combining all of the above.

**Drift analysis:**
- `assignment_drift(X_new)` → DataFrame — Assigns newcomers one-by-one and records per-cluster changes in WCSS, centroid position, and mean member distance. Populates `wcss_timeline_`.
- `drift_report(drift)` — Scalar summary of drift coverage, WCSS change, and centroid movement.

**Plots:**
- `plot_dendrogram()` — Truncated dendrogram with cut-line and oversized-cluster highlighting
- `plot_cluster_sizes()` — Bar chart of cluster sizes with `group_size` reference line
- `plot_feature_heatmap(X)` — Z-scored per-cluster feature means heatmap
- `plot_wcss_timeline(ax, label)` — Mean WCSS per cluster over newcomer assignment; overlay multiple models on one axis
- `plot_assignment_drift(drift)` — Two-panel chart: WCSS delta (top) and centroid drift (bottom), both sorted by WCSS delta descending, bars coloured by `n_assigned`
- `plot_drift_summary(drift)` — 4-panel dashboard: WCSS delta distribution, assignment load vs. cohesion impact, mean distance pre/post, top-10 most impacted clusters

---

## API

Start with `uvicorn api.main:app --reload`. Interactive docs at `http://localhost:8000/docs`.

The API wraps the fitted `AggloGroupModel`. Global state lives in `api/state.py` (`AppState` dataclass). The model must be initialised before any other endpoint is usable.

### System endpoints (`api/routers/system.py`)

| Method | Path | Description |
|---|---|---|
| `POST` | `/initialize` | Upload a CSV (same format as `data/data.csv`), fit the model, return clustering metrics. Accepts `separate_not_needed: bool` form field (default `false`). Returns `n_denied_treatment` count. |
| `GET` | `/status` | Overall health check — initialization state, patient/group counts, WCSS, silhouette. |
| `GET` | `/metrics` | System-wide clustering quality metrics. Recomputed over all current patients. |

**`POST /initialize` parameters:**
- `file` (required) — multipart CSV upload
- `separate_not_needed` (optional, default `false`) — when `true`, patients with `treatment_not_needed=1` are routed to group `−1` without affecting clustering metrics

### Group endpoints (`api/routers/groups.py`)

| Method | Path | Description |
|---|---|---|
| `GET` | `/groups` | List all groups with clinical + demographic summaries. |
| `GET` | `/groups/{group_id}` | Full group detail including member records, WCSS, mean centroid distance. |
| `DELETE` | `/groups/{group_id}` | Remove a group. `?reassign=true` redistributes members to nearest clusters; `?reassign=false` orphans them. |
| `POST` | `/groups/{group_id}/split` | Manually split a group into two halves via 2-cluster agglomerative clustering. |
| `GET` | `/groups/{group_id}/similar` | Return the `n` most similar groups by centroid distance (default `n=5`). |
| `GET` | `/groups/{group_id}/metrics` | Per-group WCSS and mean centroid distance. |

### Patient endpoints (`api/routers/patients.py`)

| Method | Path | Description |
|---|---|---|
| `GET` | `/patients` | All patients with their group assignments. |
| `GET` | `/patients/{patient_id}` | Patient detail + cleaned feature data + group assignment. |
| `GET` | `/patients/{patient_id}/group` | Lightweight: which group is this patient in? |
| `DELETE` | `/patients/{patient_id}` | Remove a patient from their group. |
| `POST` | `/newcomers` | Upload a CSV of new patients. Each is assigned online to the nearest cluster with capacity. Returns per-patient WCSS impact and centroid drift. When `separate_not_needed=true` was set at initialization, patients with `treatment_not_needed=1` are automatically routed to group `−1` with zeroed-out metrics. |

### Response schemas (selected)

`InitializeResponse` — `n_patients`, `n_groups`, `n_denied_treatment`, `group_stats`, `silhouette`, `wcss_total`, `initialized_at`

`NewcomerResult` — `patient_id`, `assigned_group_id`, `group_size_before`, `group_size_after`, `wcss_before`, `wcss_after`, `wcss_delta`, `centroid_drift`

`GroupDetail` — `group_id`, `size`, `clinical` (`ClinicalSummary`), `demographic` (`DemographicSummary`), `members`, `wcss`, `mean_dist_to_centroid`

---

## Notebooks

### `modeling.ipynb` — Main analysis

Splits data 85/15 (`initial_set` / `newcomer_set`). Builds a `FeatureConfig` with 19 numeric + 4 categorical features. Runs and compares multiple model configurations:

1. Plain `AggloGroupModel` (ward linkage)
2. Average-linkage variant
3. Weighted `AggloGroupModel` (clinical feature weights)
4. k-NN connectivity model
5. Demographic attribute-matched connectivity model
6. **Final model:** Clinical weights + attribute-matched connectivity (`is_professional × age_group × gender_enc`), with `separate_not_needed=True`

Assigns `newcomer_set` online to each model, then runs `Evaluator.assignment_drift()` and plots WCSS timelines for all models.

### `eda.ipynb` — Exploratory data analysis

Monkey-patches `eda.save` to display 14 plots inline rather than writing PNGs. Same plots as `prep/eda.py` outputs to `eda_plots/`.

---

## Tests

```bash
pytest                                   # all tests (includes geocoding — slow)
pytest tests/test_modeling.py            # model + evaluator only (fast, no geocoding)
pytest tests/test_api.py                 # API integration tests
```

`tests/conftest.py` uses `scope="module"` fixtures and a `RateLimiter(min_delay_seconds=1)` to keep geocoding calls to a minimum.

**Test classes:**

| File | Class | Covers |
|---|---|---|
| `test_modeling.py` | `TestBaselineGroupModel` | fit, predict, assign_cluster |
| `test_modeling.py` | `TestAggloGroupModel` | fit, predict, weights, connectivity, assign, split, labels |
| `test_modeling.py` | `TestEvaluator` | WCSS, silhouette, report, group stats, drift, drift report, baseline comparison |
| `test_api.py` | `TestBeforeInit` | 503 behaviour before initialization |
| `test_api.py` | `TestReadOnly` | status, metrics, groups, patients after initialization |
| `test_api.py` | `TestMutating` | newcomers, split, delete, remove patient |

---

## Feature Pipeline Reference

Full detail in [`feature_pipeline.md`](feature_pipeline.md). Summary:

### Stage 1 — `PatientDataTransformer`

**1.1 Preparation** — column normalisation, renaming (`working_professional_or_student` → `status`, etc.), typo fixes, numeric coercion, merging split student/professional columns (`academic_pressure` + `work_pressure` → `pressure`; `study_satisfaction` + `job_satisfaction` → `satisfaction`).

**1.2 Imputation** — all statistics learned from training set only: per-status medians for `pressure` and `satisfaction`; student-only median for `cgpa`; global medians for `workstudy_hours`, `financial_stress`, `sleep_duration`.

**1.3 Encoding**

| Column | Method | Output |
|---|---|---|
| `suicidal_thoughts`, `family_history`, `depression` | Yes/No → 1/0 | binary |
| `gender` | Male=1, Female=0 | `gender_enc` |
| `status` | Working Professional=1, Student=0 | `is_professional` |
| `dietary_habits` | Unhealthy=0, Moderate=1, Healthy=2 | `dietary_enc` |
| `degree` | Class 12=0, Bachelor's=1, Master's=2, PhD/MD=3 | `education_level` |
| `profession` | 8 categories + Student/Other | `profession_category` |
| `sleep_duration` | Less than 5h=1, 5–6h=2, 7–8h=3, More than 8h=4 | `sleep_duration` (ordinal) |

**1.4 Geocoding** — each unique city geocoded once via `geopy.Nominatim` (rate-limited, cached). Adds `city_lat` / `city_lon`. Cities unseen during `fit()` get `NaN`.

**1.5 Feature engineering**

| Feature | Formula |
|---|---|
| `stress_index` | `(pressure + financial_stress + (5 − sleep_duration)) / 3` |
| `wellbeing_score` | `(satisfaction + sleep_duration + dietary_enc) / 3` |
| `worklife_balance` | `1 − (workstudy_hours / max_hours) × (1 − (sleep_duration − 1) / 3)` |
| `high_risk` | `stress_index ≥ 75th pct AND (family_history OR suicidal_thoughts OR depression)` |
| `treatment_not_needed` | `depression=0 AND pressure≤2 AND satisfaction≥4 AND suicidal_thoughts=0 AND sleep_duration≥2 AND dietary_enc≥1` |
| `age_group` | 5 bins: ≤22, 23–30, 31–40, 41–50, 51+ |
| `cgpa_band` | 5 bins + `non_student` for professionals |

`treatment_not_needed` is not passed to the `ColumnTransformer`; it is used as a pre-clustering filter (and optionally to populate cluster `−1` via `separate_not_needed=True`).

**1.6 Finalisation** — drop rows missing any essential column (`age`, `pressure`, `satisfaction`, `sleep_duration`, `workstudy_hours`, `financial_stress`, `depression`), deduplicate, set index name to `patient_id`.

### Stage 2 — `ColumnTransformer`

| Step | Columns | Transformation | Output prefix |
|---|---|---|---|
| `numeric` | 19 features | `StandardScaler` | `numeric__` |
| `categorical` | `city`, `profession_category`, `age_group`, `cgpa_band` | `OneHotEncoder` | `categorical__` |
| `boolean` | (configurable) | passthrough | `boolean__` |

10 statistics are frozen from the training set to prevent leakage: pressure/satisfaction/cgpa/workstudy/financial_stress/sleep medians, `max_hours_`, `stress_75_`, `city_coords_`, plus StandardScaler mean/std and OneHotEncoder category sets.
