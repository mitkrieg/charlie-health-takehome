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
        labels = self.model.labels_
        return silhouette_score(
            X.loc[labels.index, feature_cols].fillna(0),
            labels,
            metric=metric,
            random_state=self.random_state,
        )

    def report(self, X: pd.DataFrame, metric: str = "euclidean") -> str:
        wcss, per_cluster_wcss = self.wcss(X)
        sil = self.silhouette(X, metric)
        stats = self.group_size_stats()

        lines = [
            "Metrics report:",
            f"  Groups:     {stats['n_groups']}",
            f"  Sizes:      min={stats['min']}  max={stats['max']}  mean={stats['mean']:.1f}  std={stats['std']:.1f}  cv={stats['cv']:.3f}",
            f"  WCSS:       total={wcss:.2f} mean={wcss / len(per_cluster_wcss): .2f}",
            f"  Silhouette: {sil:.4f}  (metric={metric})",
        ]
        return "\n".join(lines)

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
        ax.axhline(
            y=self.model.group_size, color="red", linestyle="--", label="max size"
        )
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

    def plot_feature_heatmap(self, X: pd.DataFrame, ax=None):
        """Heatmap of z-scored per-cluster feature means."""
        if ax is None:
            _, ax = plt.subplots()
        hom = self.feature_homogeneity(X)
        scaled = hom.apply(zscore, axis=0)
        sns.heatmap(scaled, center=0, cmap="RdBu_r", ax=ax)
        return ax
