from sklearn.base import BaseEstimator
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from prep.data_cleaning import PatientDataTransformer


@dataclass
class FeatureConfig:
    numeric_features: list[str]
    categorical_features: list[str]
    boolean_features: list[str]
    random_state: int = 123


class Vectorizer:
    def __init__(
        self, config: FeatureConfig, model: BaseEstimator | None = None
    ) -> None:
        self.config = config
        self.model = model
        self.pipeline = self._build_pipeline(model)

    def _build_pipeline(self, model: BaseEstimator | None) -> Pipeline:
        data_transform = PatientDataTransformer()
        numeric = Pipeline([("scaler", StandardScaler())])
        categorical = Pipeline(
            [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
        )
        column_transformer = ColumnTransformer(
            [
                ("numeric", numeric, self.config.numeric_features),
                ("categorical", categorical, self.config.categorical_features),
                ("boolean", "passthrough", self.config.boolean_features),
            ]
        ).set_output(transform="pandas")

        if model:
            pipeline = Pipeline(
                [
                    ("preprocess", data_transform),
                    ("column_transform", column_transformer),
                    ("model", model),
                ]
            )
        else:
            pipeline = Pipeline(
                [
                    ("preprocess", data_transform),
                    ("column_transform", column_transformer),
                ]
            )

        return pipeline

    def fit(self, X, y=None):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("requires model. pass a model or use transform")
        return self.pipeline.predict(X)

    def transform(self, X):
        if self.model:
            return self.pipeline.predict(X)

        return self.pipeline.transform(X)
