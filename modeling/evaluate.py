import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram


def _cluster_stats(members_df, feature_cols):
    """Returns (wcss, mean_dist, centroid_vec) for a cluster DataFrame."""
    data = members_df[feature_cols].fillna(0).values
    centroid = data.mean(axis=0)
    diffs = data - centroid
    wcss = float(np.sum(diffs**2))
    mean_dist = float(np.linalg.norm(diffs, axis=1).mean()) if len(data) > 0 else 0.0
    return wcss, mean_dist, centroid


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

    def report(
        self,
        X: pd.DataFrame,
        X_raw: pd.DataFrame | None = None,
        metric: str = "euclidean",
    ) -> str:
        wcss, per_cluster_wcss = self.wcss(X)
        sil = self.silhouette(X, metric)
        stats = self.group_size_stats(X_raw=X_raw)

        lines = [
            "Metrics report:",
            f"  Groups:     {stats['n_groups']}",
            f"  Sizes:      min={stats['min']}  max={stats['max']}  mean={stats['mean']:.1f}  std={stats['std']:.1f}  cv={stats['cv']:.3f}",
            f"  WCSS:       total={wcss:.2f} mean={wcss / len(per_cluster_wcss): .2f}",
            f"  Silhouette: {sil:.4f}  (metric={metric})",
        ]

        homogeneity_fields = [
            ("pct_uniform_gender", "uniform gender"),
            ("pct_uniform_status", "uniform status"),
            ("pct_uniform_city", "uniform city"),
            ("pct_age_range_10yr", "age range ≤10yr"),
            ("pct_uniform_dietary_habits", "uniform diet"),
            ("pct_uniform_profession", "uniform profession"),
            ("pct_uniform_depression", "uniform depression"),
            ("pct_uniform_suicidal_thoughts", "uniform suicidal"),
            ("pct_uniform_high_risk", "uniform high_risk"),
            ("pct_possible_non_treatment", "uniform treatment not needed"),
        ]
        cohesion = [
            (label, stats[key]) for key, label in homogeneity_fields if key in stats
        ]
        if cohesion:
            lines.append("  Cohesion:   % of groups that are homogeneous")
            for label, val in cohesion:
                lines.append(f"    {label:<26} {val:.1f}%")

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

    def group_size_stats(self, X_raw: pd.DataFrame | None = None) -> dict:
        """Summary statistics for cluster sizes and within-group homogeneity.

        X_raw — optional cleaned (pre-vectorization) DataFrame (e.g. output of
        ``vec.pipeline.steps[0][1].transform(initial_set)``).  When provided,
        cohesion stats are computed by joining cluster member indices against
        X_raw so that original-scale values (city names, age in years, etc.)
        are used.  Without it the cohesion section is omitted.
        """
        clusters = list(self.model.clusters_.values())
        n = len(clusters)
        sizes = np.array([len(m) for m in clusters], dtype=float)
        mean = sizes.mean()
        std = sizes.std()

        stats = {
            "n_groups": n,
            "min": int(sizes.min()),
            "max": int(sizes.max()),
            "mean": float(mean),
            "std": float(std),
            "cv": float(std / mean) if mean > 0 else float("nan"),
        }

        if X_raw is None:
            return stats

        # Build per-cluster raw-feature slices using shared index
        raw_clusters = [X_raw.loc[X_raw.index.intersection(m.index)] for m in clusters]

        def _pct_uniform(col):
            """% of groups where all members share the same value for col."""
            eligible = [rc for rc in raw_clusters if col in rc.columns]
            if not eligible:
                return None
            count = sum(1 for rc in eligible if rc[col].dropna().nunique() <= 1)
            return round(100.0 * count / len(eligible), 1)

        def _pct_age_range(max_range=10):
            eligible = [
                rc for rc in raw_clusters if "age" in rc.columns and len(rc) > 0
            ]
            if not eligible:
                return None
            count = sum(
                1 for rc in eligible if (rc["age"].max() - rc["age"].min()) <= max_range
            )
            return round(100.0 * count / len(eligible), 1)

        for col, key in [
            ("gender", "pct_uniform_gender"),
            ("status", "pct_uniform_status"),
            ("city", "pct_uniform_city"),
            ("unhealthy_diet", "pct_uniform_dietary_habits"),
            ("profession_category", "pct_uniform_profession"),
            ("depression", "pct_uniform_depression"),
            ("suicidal_thoughts", "pct_uniform_suicidal_thoughts"),
            ("high_risk", "pct_uniform_high_risk"),
            ("treatment_not_needed", "pct_possible_non_treatment"),
        ]:
            val = _pct_uniform(col)
            if val is not None:
                stats[key] = val

        val = _pct_age_range(10)
        if val is not None:
            stats["pct_age_range_10yr"] = val

        return stats

    def print_clusters(self, X: pd.DataFrame, max_rows: int = 5) -> None:
        """Print a compact per-cluster summary using original cleaned data X."""
        DISPLAY_COLS = [
            "age",
            "gender",
            "city",
            "status",
            "profession_category",
            "dietary_habits",
            "education_level",
            "stress_index",
            "high_risk",
            "depression",
            "suicidal_thoughts",
            "family_history",
            "pressure",
            "satisfaction",
            "sleep_duration",
            "financial_stress",
            "treatment_not_needed",
        ]

        labels = self.model.labels_
        for cid in sorted(self.model.clusters_):
            member_ids = labels[labels == cid].index
            members = X.loc[X.index.intersection(member_ids)]
            print(f"── Cluster {cid}  ({len(members)} members) {'─' * 50}")

            # Age
            if "age" in members.columns:
                age = members["age"].dropna()
                print(
                    f"  Age          mean={age.mean():.1f}  (range {int(age.min())}–{int(age.max())})"
                )

            # Status split
            if "status" in members.columns:
                pct_prof = (
                    members["status"]
                    .str.contains("Professional", case=False, na=False)
                    .mean()
                    * 100
                )
                pct_stud = 100 - pct_prof
                print(
                    f"  Status       {pct_prof:.0f}% Professional  {pct_stud:.0f}% Student"
                )

            # Risk flags
            risk_parts = []
            for col, label in [
                ("high_risk", "high_risk"),
                ("depression", "depression"),
                ("suicidal_thoughts", "suicidal"),
            ]:
                if col in members.columns:
                    risk_parts.append(f"{label}={members[col].mean() * 100:.0f}%")
            if risk_parts:
                print(f"  Risk         {'  '.join(risk_parts)}")

            # Composite scores
            score_parts = []
            for col, label in [
                ("stress_index", "stress"),
                ("wellbeing_score", "wellbeing"),
                ("worklife_balance", "worklife"),
            ]:
                if col in members.columns:
                    score_parts.append(f"{label}={members[col].mean():.2f}")
            if score_parts:
                print(f"  Scores       {'  '.join(score_parts)}")

            # Raw stressors
            stressor_parts = []
            for col, label in [
                ("pressure", "pressure"),
                ("financial_stress", "financial"),
                ("sleep_duration", "sleep"),
            ]:
                if col in members.columns:
                    stressor_parts.append(f"{label}={members[col].mean():.1f}")
            if stressor_parts:
                print(f"  Stressors    {'  '.join(stressor_parts)}")

            # Sample rows
            display_cols = [c for c in DISPLAY_COLS if c in members.columns]
            print()
            print(members[display_cols].head(max_rows).to_string())
            if len(members) > max_rows:
                print(f"   ... {len(members) - max_rows} more rows")
            print()

    def feature_homogeneity(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-cluster means for each numeric feature."""
        feature_cols = X.select_dtypes(include="number").columns
        rows = {}
        for cid, members in self.model.clusters_.items():
            rows[cid] = members[feature_cols].mean()
        return pd.DataFrame(rows).T.sort_index()

    def assignment_drift(self, X_new: pd.DataFrame) -> pd.DataFrame:
        """Assign new patients one-by-one and report per-cluster drift metrics."""
        feature_cols = self.model.feature_cols_

        # 1. Snapshot pre-assignment state
        pre_sizes = {}
        pre_means = {}
        pre_wcss = {}
        pre_mean_dist = {}
        for cid, members in self.model.clusters_.items():
            pre_sizes[cid] = len(members)
            wcss, mean_dist, centroid = _cluster_stats(members, feature_cols)
            pre_wcss[cid] = wcss
            pre_mean_dist[cid] = mean_dist
            pre_means[cid] = centroid

        # 2. Assign all new rows one at a time
        assigned_to: dict = {}
        for idx in X_new.index:
            cid = self.model.assign_cluster(X_new.loc[[idx]])
            assigned_to[idx] = cid

        n_assigned_per_cluster: dict = {}
        for cid in assigned_to.values():
            n_assigned_per_cluster[cid] = n_assigned_per_cluster.get(cid, 0) + 1

        # 3. Compute post-assignment state for all current clusters
        all_cids = sorted(self.model.clusters_.keys())
        rows = []
        for cid in all_cids:
            members = self.model.clusters_[cid]
            post_wcss, post_mean_dist, post_centroid = _cluster_stats(
                members, feature_cols
            )

            is_new = cid not in pre_sizes
            size_pre = 0 if is_new else pre_sizes[cid]
            size_post = len(members)
            n_assigned = n_assigned_per_cluster.get(cid, 0)

            if is_new:
                wcss_pre = float("nan")
                mean_dist_pre = float("nan")
                centroid_drift = float("nan")
            else:
                wcss_pre = pre_wcss[cid]
                mean_dist_pre = pre_mean_dist[cid]
                centroid_drift = float(np.linalg.norm(post_centroid - pre_means[cid]))

            rows.append(
                {
                    "cluster_id": cid,
                    "size_pre": size_pre,
                    "size_post": size_post,
                    "n_assigned": n_assigned,
                    "wcss_pre": wcss_pre,
                    "wcss_post": post_wcss,
                    "wcss_delta": post_wcss
                    - (wcss_pre if not np.isnan(wcss_pre) else post_wcss),
                    "centroid_drift": centroid_drift,
                    "mean_dist_pre": mean_dist_pre,
                    "mean_dist_post": post_mean_dist,
                }
            )

        result = pd.DataFrame(rows).set_index("cluster_id")
        # wcss_delta for new clusters should be NaN, not 0
        result.loc[result["size_pre"] == 0, "wcss_delta"] = float("nan")
        return result

    def plot_assignment_drift(self, drift: pd.DataFrame, ax=None) -> plt.Axes:
        """Grouped bar chart of WCSS pre/post per cluster with centroid drift on a secondary axis."""
        if ax is None:
            _, ax = plt.subplots(figsize=(14, 5))

        cids = drift.index.tolist()
        x = np.arange(len(cids))
        width = 0.35

        for i, cid in enumerate(cids):
            alpha = 1.0 if drift.loc[cid, "n_assigned"] > 0 else 0.4
            wcss_pre = drift.loc[cid, "wcss_pre"]
            wcss_post = drift.loc[cid, "wcss_post"]
            ax.bar(
                x[i] - width / 2,
                wcss_pre if not np.isnan(wcss_pre) else 0,
                width,
                color="steelblue",
                alpha=alpha,
                label="WCSS pre" if i == 0 else "_nolegend_",
            )
            ax.bar(
                x[i] + width / 2,
                wcss_post,
                width,
                color="navy",
                alpha=alpha,
                label="WCSS post" if i == 0 else "_nolegend_",
            )

        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("WCSS")
        ax.set_xticks(x)
        ax.set_xticklabels(cids)
        ax.set_title("Cluster Assignment Drift")

        ax2 = ax.twinx()
        drift_vals = drift["centroid_drift"].values
        ax2.plot(
            x,
            drift_vals,
            color="orange",
            marker="o",
            linewidth=1.5,
            label="Centroid drift",
        )
        ax2.set_ylabel("Centroid drift (Euclidean)", color="orange")
        ax2.tick_params(axis="y", labelcolor="orange")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        return ax

    def plot_feature_heatmap(self, X: pd.DataFrame, ax=None):
        """Heatmap of z-scored per-cluster feature means."""
        if ax is None:
            _, ax = plt.subplots()
        hom = self.feature_homogeneity(X)
        scaled = hom.apply(zscore, axis=0)
        sns.heatmap(scaled, center=0, cmap="RdBu_r", ax=ax)
        return ax
