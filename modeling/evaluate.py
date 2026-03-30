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
        if hasattr(self.model, "feature_cols_"):
            feature_cols = self.model.feature_cols_
        else:
            # BaselineGroupModel stores the raw input; infer numeric columns from clusters
            sample = next(iter(self.model.clusters_.values()))
            feature_cols = sample.select_dtypes(include="number").columns.tolist()

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

        # 2. Assign all new rows one at a time, recording mean WCSS after each step
        assigned_to: dict = {}
        self.wcss_timeline_ = []
        for idx in X_new.index:
            cid = self.model.assign_cluster(X_new.loc[[idx]])
            assigned_to[idx] = cid
            total = sum(
                _cluster_stats(m, feature_cols)[0]
                for m in self.model.clusters_.values()
            )
            self.wcss_timeline_.append(total / len(self.model.clusters_))

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

    def drift_report(self, drift: pd.DataFrame) -> str:
        """Scalar summary of newcomer assignment drift (mirrors report() style).

        drift — DataFrame returned by assignment_drift().
        """
        total_clusters = len(drift)
        aff = drift[drift["n_assigned"] > 0]
        n_affected = len(aff)
        n_new = int((drift["size_pre"] == 0).sum())
        n_newcomers = int(drift["n_assigned"].sum())

        pre_total = float(drift["wcss_pre"].sum(skipna=True))
        post_total = float(drift["wcss_post"].sum(skipna=True))
        wcss_abs = post_total - pre_total
        wcss_pct = (wcss_abs / pre_total * 100) if pre_total != 0 else float("nan")

        wcss_mean_delta = float(aff["wcss_delta"].mean(skipna=True))
        worst_cid = drift["wcss_delta"].idxmax(skipna=True)
        worst_val = float(drift.loc[worst_cid, "wcss_delta"])

        centroid_mean = float(aff["centroid_drift"].mean(skipna=True))
        centroid_max_cid = aff["centroid_drift"].idxmax(skipna=True)
        centroid_max_val = float(aff.loc[centroid_max_cid, "centroid_drift"])

        pre_dist = float(aff["mean_dist_pre"].mean(skipna=True))
        post_dist = float(aff["mean_dist_post"].mean(skipna=True))
        dist_pct = ((post_dist - pre_dist) / pre_dist * 100) if pre_dist != 0 else float("nan")

        sign = "+" if wcss_abs >= 0 else ""
        sign_d = "+" if wcss_mean_delta >= 0 else ""
        sign_w = "+" if worst_val >= 0 else ""
        sign_dist = "+" if (post_dist - pre_dist) >= 0 else ""

        lines = [
            f"Drift report ({n_newcomers} newcomers → {total_clusters} clusters):",
            f"  Coverage:       {n_affected} of {total_clusters} clusters affected ({100 * n_affected / total_clusters:.1f}%)  |  {n_new} new clusters from splits",
            f"  WCSS total:     pre={pre_total:,.0f}  post={post_total:,.0f}  Δ={sign}{wcss_abs:,.1f}  ({sign}{wcss_pct:.1f}%)",
            f"  WCSS delta:     mean={sign_d}{wcss_mean_delta:.1f}  worst=cluster {worst_cid} ({sign_w}{worst_val:.1f})",
            f"  Centroid drift: mean={centroid_mean:.1f}  max={centroid_max_val:.1f} (cluster {centroid_max_cid})",
            f"  Cohesion:       mean dist pre={pre_dist:.2f}  post={post_dist:.2f}  ({sign_dist}{dist_pct:.1f}%)  [affected clusters]",
        ]
        return "\n".join(lines)

    def plot_assignment_drift(self, drift: pd.DataFrame) -> plt.Figure:
        """Two-panel chart for cluster assignment drift (affected clusters only).

        Only clusters that received at least one newcomer are shown.

        Top panel    — WCSS increase (wcss_delta) per affected cluster, coloured
                       by n_assigned so high-traffic clusters stand out.
        Bottom panel — Centroid drift per affected cluster.

        Returns the Figure for further customisation.
        """
        affected = drift[drift["n_assigned"] > 0].copy()
        n_total = len(drift)
        n_affected = len(affected)

        fig, (ax_wcss, ax_drift) = plt.subplots(
            2, 1, figsize=(max(10, n_affected * 0.4), 8), sharex=True,
            gridspec_kw={"hspace": 0.08},
        )

        x = np.arange(n_affected)
        cids = affected.index.tolist()

        # Colour bars by n_assigned (darker = more newcomers)
        norm = plt.Normalize(vmin=1, vmax=max(affected["n_assigned"].max(), 2))
        cmap = plt.cm.Blues
        colours = [cmap(0.4 + 0.6 * norm(v)) for v in affected["n_assigned"]]

        ax_wcss.bar(x, affected["wcss_delta"].clip(lower=0), color=colours)
        ax_wcss.axhline(0, color="black", linewidth=0.8)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, ax=ax_wcss, label="n_assigned", pad=0.01)
        ax_wcss.set_ylabel("WCSS increase (Δ)")
        ax_wcss.set_title(
            f"Assignment Drift — {n_affected} of {n_total} clusters received newcomers"
        )

        ax_drift.bar(x, affected["centroid_drift"].fillna(0), color=colours)
        ax_drift.set_ylabel("Centroid drift (Euclidean)")
        ax_drift.set_xlabel("Cluster ID")
        ax_drift.set_xticks(x)
        ax_drift.set_xticklabels(cids, rotation=90, fontsize=8)

        fig.tight_layout()
        return fig

    def plot_drift_summary(self, drift: pd.DataFrame, fig=None) -> plt.Figure:
        """4-panel summary dashboard of assignment drift health.

        Parameters
        ----------
        drift : DataFrame returned by ``assignment_drift()``.
        fig   : optional pre-created Figure (must have room for 4 subplots).

        Returns the Figure.
        """
        affected = drift[drift["n_assigned"] > 0].copy()
        n_affected = len(affected)
        n_total = len(drift)
        n_newcomers = int(drift["n_assigned"].sum())

        # --- edge case: nothing to show ---
        if n_affected == 0:
            if fig is None:
                fig = plt.figure(figsize=(14, 10))
            fig.text(0.5, 0.5, "No clusters received newcomers",
                     ha="center", va="center", fontsize=16)
            return fig

        if fig is None:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                                     constrained_layout=True)
        else:
            axes = np.array(fig.subplots(2, 2))
        ax_a, ax_b = axes[0]
        ax_c, ax_d = axes[1]

        wcss_delta = affected["wcss_delta"].dropna()
        median_val = float(wcss_delta.median())
        p90_val = float(np.nanpercentile(wcss_delta, 90))

        # ── Panel A: WCSS Delta Distribution ──
        sns.histplot(wcss_delta, kde=True, ax=ax_a, color="steelblue")
        ax_a.axvline(median_val, color="navy", linestyle="--", linewidth=1.2,
                     label=f"median {median_val:.1f}")
        ax_a.axvline(p90_val, color="firebrick", linestyle="--", linewidth=1.2,
                     label=f"P90 {p90_val:.1f}")
        ax_a.legend(fontsize=9)
        ax_a.set_title("WCSS Change Distribution (affected clusters)")
        ax_a.set_xlabel("wcss_delta")

        # ── Panel B: Assignment Load vs. Cohesion Impact ──
        sc = ax_b.scatter(affected["n_assigned"], affected["wcss_delta"],
                          c=affected["size_pre"], cmap="YlOrRd",
                          edgecolors="grey", linewidths=0.5, s=40)
        fig.colorbar(sc, ax=ax_b, label="Pre-assignment size")
        # annotate outliers above P90
        outliers = affected[affected["wcss_delta"] > p90_val]
        for cid, row in outliers.iterrows():
            ax_b.annotate(str(cid), (row["n_assigned"], row["wcss_delta"]),
                          fontsize=7, color="firebrick",
                          textcoords="offset points", xytext=(4, 4))
        ax_b.set_xlabel("n_assigned")
        ax_b.set_ylabel("wcss_delta")
        ax_b.set_title("Assignment Load vs. Cohesion Impact")

        # ── Panel C: Mean Distance Pre vs. Post ──
        existing = affected[affected["size_pre"] > 0].copy()
        sc_c = ax_c.scatter(existing["mean_dist_pre"], existing["mean_dist_post"],
                            c=existing["n_assigned"], cmap="Blues",
                            edgecolors="grey", linewidths=0.5, s=40)
        fig.colorbar(sc_c, ax=ax_c, label="n_assigned")
        # y = x reference line
        lo = min(existing["mean_dist_pre"].min(), existing["mean_dist_post"].min())
        hi = max(existing["mean_dist_pre"].max(), existing["mean_dist_post"].max())
        margin = (hi - lo) * 0.05
        ax_c.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                  color="grey", linestyle="--", linewidth=0.8)
        n_worsened = int((existing["mean_dist_post"] > existing["mean_dist_pre"]).sum())
        n_existing = len(existing)
        ax_c.text(0.05, 0.95, f"{n_worsened} of {n_existing} clusters worsened",
                  transform=ax_c.transAxes, fontsize=9, verticalalignment="top",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
        ax_c.set_xlabel("mean_dist_pre")
        ax_c.set_ylabel("mean_dist_post")
        ax_c.set_title("Mean Distance Pre vs. Post")

        # ── Panel D: Top 10 Most Impacted Clusters ──
        top_n = min(10, n_affected)
        top = affected.nlargest(top_n, "wcss_delta")
        y_labels = [
            f"C{cid}  (n={int(row['n_assigned'])}, drift={row['centroid_drift']:.1f})"
            for cid, row in top.iterrows()
        ]
        bars = ax_d.barh(range(top_n), top["wcss_delta"], color="steelblue")
        ax_d.set_yticks(range(top_n))
        ax_d.set_yticklabels(y_labels, fontsize=8)
        ax_d.invert_yaxis()
        ax_d.set_xlabel("wcss_delta")
        ax_d.set_title("Top 10 Most Impacted Clusters")

        # ── Suptitle ──
        pct_worsened = (100 * n_worsened / n_existing) if n_existing > 0 else 0
        fig.suptitle(
            f"{n_newcomers} newcomers → {n_affected}/{n_total} clusters affected  |  "
            f"median WCSS Δ = {median_val:.1f}  |  "
            f"{pct_worsened:.0f}% of affected clusters worsened",
            fontsize=12, fontweight="bold",
        )

        return fig

    _TIMELINE_COLORS = ["steelblue", "darkorange", "seagreen", "firebrick",
                         "mediumpurple", "goldenrod", "deeppink", "teal"]

    def plot_wcss_timeline(self, ax=None, label=None) -> plt.Axes:
        """Line chart of mean WCSS across all clusters after each newcomer assignment.

        Requires ``assignment_drift()`` to have been called first (populates
        ``self.wcss_timeline_``).

        Parameters
        ----------
        ax    : pass an existing Axes to overlay multiple models on one chart.
        label : legend label; defaults to the model's class name.
        """
        if not hasattr(self, "wcss_timeline_"):
            raise RuntimeError("Call assignment_drift() before plot_wcss_timeline().")

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 4))

        if label is None:
            label = type(self.model).__name__

        # pick the next colour that hasn't been used on this axes
        idx = len(ax.lines)
        color = self._TIMELINE_COLORS[idx % len(self._TIMELINE_COLORS)]

        steps = np.arange(1, len(self.wcss_timeline_) + 1)
        tl = self.wcss_timeline_
        slope = (tl[-1] - tl[0]) / len(tl) if len(tl) > 1 else 0.0
        sign = "+" if slope >= 0 else ""
        ax.plot(steps, tl, linewidth=1.5, color=color,
                label=f"{label}  ({sign}{slope:.3f}/newcomer)")
        ax.set_xlabel("Newcomers assigned")
        ax.set_ylabel("Mean WCSS per cluster")
        ax.set_title("Mean WCSS over newcomer assignment")
        ax.legend()

        return ax

    def plot_feature_heatmap(self, X: pd.DataFrame, ax=None):
        """Heatmap of z-scored per-cluster feature means."""
        if ax is None:
            _, ax = plt.subplots()
        hom = self.feature_homogeneity(X)
        scaled = hom.apply(zscore, axis=0)
        sns.heatmap(scaled, center=0, cmap="RdBu_r", ax=ax)
        return ax
