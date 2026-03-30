#!/usr/bin/env python3
"""
End-to-end example: split `data/data.csv` like `modeling.ipynb` (85% initial /
15% newcomers), then drive the FastAPI service the same way as `tests/test_api.py`.

  1. POST /initialize  — multipart upload of the initial cohort CSV (fits model)
  2. GET  /status, /metrics, /groups (sample)
  3. POST /newcomers   — upload the holdout CSV (online assignment; new IDs)

Run the API first:

  uvicorn api.main:app --reload

Then:

  python examples/api_workflow.py
  python examples/api_workflow.py --max-rows 120   # smaller, faster geocoding
  python examples/api_workflow.py --self-test      # TestClient, ~65 rows (like pytest slice)

The cleaning step geocodes cities (see `prep/data_cleaning.py`); use `--max-rows` for a
quicker demo or rely on cached coordinates after the first run.

Newcomers receive new sequential patient_ids (see `api/routers/patients.py`); they
do not keep their original dataframe index from the CSV.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

# Repo root on sys.path so `python examples/api_workflow.py` can import `api`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import httpx
import pandas as pd
from sklearn.model_selection import train_test_split


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _split_initial_newcomer(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same idea as modeling.ipynb: hold out a fraction as rolling admissions."""
    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )


def _reset_app_state() -> None:
    from api.state import get_state

    state = get_state()
    state.vectorizer = None
    state.model = None
    state.evaluator = None
    state.all_vecs = None
    state.patient_data = None
    state.initialized_at = None
    state.n_initial_patients = 0


def run_workflow(
    *,
    base_url: str,
    data_path: Path,
    test_size: float,
    random_state: int,
    max_rows: int | None,
    client: Any = None,
) -> None:
    df = pd.read_csv(data_path)
    if max_rows is not None:
        df = df.head(max_rows).copy()

    initial_set, newcomer_set = _split_initial_newcomer(
        df, test_size=test_size, random_state=random_state
    )

    init_csv = _csv_bytes(initial_set)
    new_csv = _csv_bytes(newcomer_set)

    own_client = client is None
    http: Any = client or httpx.Client(base_url=base_url, timeout=600.0)

    try:
        print(
            f"Split: {len(initial_set)} initial rows, {len(newcomer_set)} newcomer rows "
            f"(test_size={test_size}, random_state={random_state})\n"
        )

        r = http.post(
            "/initialize",
            files={"file": ("initial.csv", io.BytesIO(init_csv), "text/csv")},
        )
        r.raise_for_status()
        init_body = r.json()
        print("POST /initialize →", json.dumps(init_body, indent=2))

        for path in ("/status", "/metrics"):
            r = http.get(path)
            r.raise_for_status()
            print(f"GET {path} →", json.dumps(r.json(), indent=2))

        r = http.get("/groups")
        r.raise_for_status()
        groups = r.json()
        print(
            f"GET /groups → total_groups={groups['total_groups']}, "
            f"total_patients={groups['total_patients']}"
        )
        if groups["groups"]:
            g0 = groups["groups"][0]
            print(f"  (example) group {g0['group_id']}: size={g0['size']}")

        r = http.post(
            "/newcomers",
            files={"file": ("newcomers.csv", io.BytesIO(new_csv), "text/csv")},
        )
        r.raise_for_status()
        newcomers = r.json()
        print(
            f"\nPOST /newcomers → n_assigned={newcomers['n_assigned']}"
        )
        for a in newcomers["assignments"][:5]:
            print(
                f"  patient_id={a['patient_id']} → group {a['assigned_group_id']}, "
                f"Δwcss={a['wcss_delta']}, centroid_drift={a['centroid_drift']}"
            )
        if len(newcomers["assignments"]) > 5:
            print(f"  … ({len(newcomers['assignments']) - 5} more)")

        r = http.get("/patients")
        r.raise_for_status()
        pt = r.json()
        print(f"\nGET /patients → total={pt['total']}")
    finally:
        if own_client:
            http.close()


def _self_test() -> None:
    from fastapi.testclient import TestClient

    from api.main import app

    _reset_app_state()
    try:
        with TestClient(app) as client:
            run_workflow(
                base_url="http://test",
                data_path=_ROOT / "data" / "data.csv",
                test_size=0.15,
                random_state=42,
                max_rows=65,
                client=client,
            )
    finally:
        _reset_app_state()


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise the patient grouping API.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="API root (default: local uvicorn)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=_ROOT / "data" / "data.csv",
        help="Raw survey CSV (same columns as shipped data/data.csv)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Fraction held out for POST /newcomers (default: 0.15, like modeling.ipynb)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="RNG seed for the split (notebook uses random_state=True; int is reproducible)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        metavar="N",
        help="Use only the first N rows before splitting (faster; for demos)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run against in-process TestClient (no uvicorn; resets app state)",
    )
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return

    if not args.data.is_file():
        print(f"Missing data file: {args.data}", file=sys.stderr)
        sys.exit(1)

    run_workflow(
        base_url=args.base_url.rstrip("/"),
        data_path=args.data,
        test_size=args.test_size,
        random_state=args.random_state,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
