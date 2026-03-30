from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from modeling.vectorizer import Vectorizer
    from modeling.agglomerative_clustering import AggloGroupModel
    from modeling.evaluate import Evaluator


@dataclass
class AppState:
    vectorizer: "Vectorizer | None" = None
    model: "AggloGroupModel | None" = None
    evaluator: "Evaluator | None" = None
    # Vectorized features for all current patients (model input space)
    all_vecs: "pd.DataFrame | None" = None
    # Cleaned pre-vectorization features (human-readable output)
    patient_data: "pd.DataFrame | None" = None
    initialized_at: datetime | None = None
    n_initial_patients: int = 0


_state = AppState()


def get_state() -> AppState:
    return _state
