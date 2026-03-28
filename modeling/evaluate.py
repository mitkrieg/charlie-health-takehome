import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram


class Evaluator:
    def __init__(self, model, random_state=123) -> None:
        self.model = model
        self.random_state = random_state

    def _get_labels(self, X: pd.DataFrame) -> pd.Series:
        """Reconstruct integer label array from model.clusters_ aligned to X's index."""
        labels = pd.Series(index=X.index, dtype=float)
        for cid, members in self.model.clusters_.items():
            labels.loc[members.index.intersection(X.index)] = cid
        return labels.dropna().astype(int)

    def wcss(self, X: pd.DataFrame) -> tuple[float, dict]:
        """Within cluster sum of squares"""
        feature_cols = X.select_dtypes(include="number").columns
        per_cluster = {}
        wcss = 0.0
        for cid, members in self.model.clusters_.items():
            cluster = members[feature_cols].fillna(0)
            centroid = cluster.mean(axis=0)
            ss = float(np.sum((cluster - centroid) ** 2))
            per_cluster[cid] = ss
            wcss += ss
        return wcss, per_cluster

    def silhouette(self, X: pd.DataFrame, metric: str = "euclidean") -> float:
        feature_cols = X.select_dtypes(include="number").columns
        labels = self._get_labels(X)
        X_aligned = X.loc[labels.index][feature_cols].fillna(0)
        return silhouette_score(X_aligned, labels, metric=metric, random_state=self.random_state)

    def plot_dendrogram(self, **kwargs):
        if not hasattr(self.model, "agglo_model_"):
            raise AttributeError(
                "plot_dendrogram() requires AggloGroupModel — call predict() first "
                "to populate model.agglo_model_."
            )
        inner = self.model.agglo_model_
        counts = np.zeros(inner.children_.shape[0])
        n_samples = len(inner.labels_)
        for i, merge in enumerate(inner.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [inner.children_, inner.distances_, counts]
        ).astype(float)

        dendrogram(linkage_matrix, **kwargs)

    def plot_cluster_sizes(self, ax=None):
        """Bar chart of cluster sizes with max-size reference line."""
        if ax is None:
            _, ax = plt.subplots()
        sizes = pd.Series({cid: len(m) for cid, m in self.model.clusters_.items()})
        sizes.sort_index().plot.bar(ax=ax)
        ax.axhline(y=self.model.group_size, color="red", linestyle="--", label="max size")
        ax.legend()
        return ax

    def group_size_stats(self) -> dict:
        """Summary statistics for cluster sizes (balance check)."""
        sizes = np.array([len(m) for m in self.model.clusters_.values()], dtype=float)
        mean = sizes.mean()
        std = sizes.std()
        return {
            "n_groups": len(sizes),
            "min": int(sizes.min()),
            "max": int(sizes.max()),
            "mean": float(mean),
            "std": float(std),
            "cv": float(std / mean) if mean > 0 else float("nan"),
        }

    def feature_homogeneity(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-cluster means for each numeric feature."""
        feature_cols = X.select_dtypes(include="number").columns
        rows = {}
        for cid, members in self.model.clusters_.items():
            rows[cid] = members[feature_cols].mean()
        return pd.DataFrame(rows).T.sort_index()

    def risk_distribution(self, X: pd.DataFrame, risk_col: str = "high_risk") -> pd.DataFrame:
        """Per-cluster size, risk count, and risk rate for a binary risk column."""
        records = []
        for cid, members in self.model.clusters_.items():
            aligned = members.loc[members.index.intersection(X.index)]
            size = len(aligned)
            risk_count = int(aligned[risk_col].sum()) if risk_col in aligned.columns else 0
            records.append({
                "cluster": cid,
                "size": size,
                "risk_count": risk_count,
                "risk_rate": risk_count / size if size > 0 else float("nan"),
            })
        return pd.DataFrame(records).set_index("cluster").sort_index()

    def plot_feature_heatmap(self, X: pd.DataFrame, ax=None):
        """Heatmap of z-scored per-cluster feature means."""
        if ax is None:
            _, ax = plt.subplots()
        hom = self.feature_homogeneity(X)
        scaled = hom.apply(zscore, axis=0)
        sns.heatmap(scaled, center=0, cmap="RdBu_r", ax=ax)
        return ax
