from .baseline import GroupModel
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
import math


class AggloGroupModel(GroupModel):
    def __init__(
        self,
        group_size: int = 12,
        random_state: int = 123,
        connectivity=None,
        feature_weights: dict | None = None,
        linkage="ward",
        metric="euclidean",
        separate_not_needed: bool = False,
    ) -> None:
        super().__init__(group_size=group_size, random_state=random_state)
        self.connectivity = connectivity  # sparse/dense matrix or None
        self.feature_weights = feature_weights  # dict {feature_name: float} or None
        self.linkage = linkage
        self.metric = metric
        self.separate_not_needed = separate_not_needed

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_cols_ = X.select_dtypes(include="number").columns.tolist()
        self.rng_ = np.random.RandomState(self.random_state)
        if self.feature_weights is not None:
            self.weights_ = np.array(
                [self.feature_weights.get(col, 1.0) for col in self.feature_cols_]
            )
        else:
            self.weights_ = np.ones(len(self.feature_cols_))
        return self

    def _apply_weights(self, X_num: pd.DataFrame) -> np.ndarray:
        return X_num[self.feature_cols_].values * self.weights_

    def predict(self, X: pd.DataFrame, deny_mask=None) -> np.ndarray:
        # Separate denied patients (treatment_not_needed) into group -1
        if deny_mask is not None and deny_mask.any():
            X_cluster = X[~deny_mask]
            X_denied = X[deny_mask]
        else:
            X_cluster = X
            X_denied = None

        n_clusters = math.ceil(len(X_cluster) / self.group_size)
        agglo = AgglomerativeClustering(
            n_clusters=n_clusters,
            compute_distances=True,
            connectivity=self.connectivity,
            metric=self.metric,
            linkage=self.linkage,
        )
        labels = agglo.fit_predict(self._apply_weights(X_cluster))
        self.agglo_model_ = agglo  # exposes .children_, .distances_, .labels_
        self.clusters_ = {int(cid): X_cluster[labels == cid] for cid in set(labels)}

        if X_denied is not None and len(X_denied) > 0:
            self.clusters_[-1] = X_denied

        self._update_cluster_means()

        # Enforce hard cap: split any cluster that exceeds group_size
        oversized = [
            cid for cid, m in self.clusters_.items()
            if cid != -1 and len(m) > self.group_size
        ]
        while oversized:
            self._split_cluster(oversized.pop())
            oversized = [
                cid for cid, m in self.clusters_.items()
                if cid != -1 and len(m) > self.group_size
            ]

        return self.labels_.loc[X.index].values

    def assign_cluster(self, X: pd.DataFrame) -> int:
        patient_vec = self._apply_weights(X[self.feature_cols_].iloc[[0]])[0]
        active_cids = [cid for cid in self.clusters_ if cid != -1]
        distances = {
            cid: np.linalg.norm(
                patient_vec - self.cluster_means_[cid].values * self.weights_
            )
            for cid in active_cids
        }
        sorted_clusters = sorted(distances, key=lambda cid: distances[cid])

        cid = None
        for candidate in sorted_clusters:
            if len(self.clusters_[candidate]) < self.group_size:
                cid = candidate
                break

        if cid is None:
            closest_cid = sorted_clusters[0]
            self._split_cluster(closest_cid)
            new_cid = max(self.clusters_.keys())
            cid = new_cid

        self.clusters_[cid] = pd.concat([self.clusters_[cid], X])
        self._update_cluster_means()
        return cid

    def _split_cluster(self, cid: int):
        members = self.clusters_[cid]
        labels = AgglomerativeClustering(n_clusters=2).fit_predict(
            self._apply_weights(members)
        )
        new_cid = max(self.clusters_.keys()) + 1
        self.clusters_[cid] = members[labels == 0]
        self.clusters_[new_cid] = members[labels == 1]
        self._update_cluster_means()

    def _update_cluster_means(self):
        self.cluster_means_ = {
            cid: members[self.feature_cols_].mean()
            for cid, members in self.clusters_.items()
        }
