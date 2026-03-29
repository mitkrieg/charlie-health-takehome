from sklearn.base import BaseEstimator
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class GroupModel(BaseEstimator, ABC):
    def __init__(self, group_size: int = 12, random_state=123):
        self.group_size = group_size
        self.random_state = random_state
        self.clusters_ = {}

    @abstractmethod
    def fit(self, X: pd.DataFrame, y=None):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def assign_cluster(self, X: pd.DataFrame):
        pass

    @property
    def labels_(self) -> pd.Series:
        if not hasattr(self, "clusters_"):
            raise RuntimeError("No clusters yet — call predict() first.")

        s = pd.Series(
            {
                pid: cid
                for cid, members in self.clusters_.items()
                for pid in members.index
            }
        )
        s.index.name = "patient_id"
        return s



class BaselineGroupModel(GroupModel):
    def fit(self, X: pd.DataFrame, y=None):
        self.n_fitted_samples_ = len(X)
        self.rng_ = np.random.RandomState(self.random_state)
        return self

    def predict(self, X) -> np.ndarray:
        shuffled_idx = X.sample(frac=1, random_state=self.random_state).index
        labels = np.empty(len(X), dtype=int)
        for cid, start in enumerate(range(0, self.n_fitted_samples_, self.group_size)):
            for idx in shuffled_idx[start : start + self.group_size]:
                labels[X.index.get_loc(idx)] = cid
        self.clusters_ = {cid: X.iloc[labels == cid] for cid in range(labels.max() + 1)}
        self.n_groups = len(self.clusters_)
        return labels

    def assign_cluster(self, X) -> int:
        available = [
            cid for cid in self.clusters_ if len(self.clusters_[cid]) < self.group_size
        ]
        if available:
            cid = int(self.rng_.choice(available))
            self.clusters_[cid] = pd.concat([self.clusters_[cid], X])
            return cid
        else:
            cid = int(self.rng_.choice(list(self.clusters_.keys())))
            members = self.clusters_[cid]
            mid = len(members) // 2
            new_cid = max(self.clusters_.keys()) + 1
            self.clusters_[cid] = members.iloc[:mid]
            self.clusters_[new_cid] = pd.concat([members.iloc[mid:], X])
            return new_cid
