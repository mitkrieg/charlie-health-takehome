from sklearn.base import BaseEstimator
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class GroupModel(BaseEstimator, ABC):
    def __init__(self, group_size: int = 12, random_state=123):
        self.group_size = group_size
        self.random_state = random_state

    @abstractmethod
    def fit(self, X: pd.DataFrame, y=None):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame):
        pass

    @abstractmethod
    def assign_cluster(self, X: pd.DataFrame):
        pass


class BaselineGroupModel(GroupModel):
    def fit(self, X: pd.DataFrame, y=None):
        self.n_fitted_samples_ = len(X)
        self.n_fitted_samples = len(X)
        self.rng_ = np.random.RandomState(self.random_state)
        return self

    def predict(self, X) -> list[pd.DataFrame]:
        shuffled = (
            X.copy()
            .sample(frac=1, random_state=self.random_state)
            .reset_index(drop=True)
        )
        groups = [
            shuffled.iloc[i : i + self.group_size]
            for i in range(0, self.n_fitted_samples_, self.group_size)
        ]
        self.n_groups = len(groups)
        self.clusters_ = {i: g for i, g in enumerate(groups)}
        return groups

    def assign_cluster(self, X) -> int:
        available = [cid for cid in self.clusters_ if len(self.clusters_[cid]) < self.group_size]
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
