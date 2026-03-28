from .baseline import GroupModel
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
import math


class AggloGroupModel(GroupModel):
    def fit(self, X: pd.DataFrame, y=None):
        self.feature_cols_ = X.select_dtypes(include="number").columns.tolist()
        self.rng_ = np.random.RandomState(self.random_state)
        return self

    def predict(self, X: pd.DataFrame) -> list[pd.DataFrame]:
        n_clusters = math.ceil(len(X) / self.group_size)
        agglo = AgglomerativeClustering(n_clusters=n_clusters, compute_distances=True)
        labels = agglo.fit_predict(X[self.feature_cols_].fillna(0))
        self.agglo_model_ = agglo  # exposes .children_, .distances_, .labels_
        self.clusters_ = {int(cid): X[labels == cid] for cid in set(labels)}
        self._update_cluster_means()
        return list(self.clusters_.values())

    def assign_cluster(self, X: pd.DataFrame) -> int:
        patient_vec = X[self.feature_cols_].fillna(0).iloc[0]
        distances = {
            cid: np.linalg.norm(patient_vec.values - self.cluster_means_[cid].values)
            for cid in self.clusters_
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
            members[self.feature_cols_].fillna(0)
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
