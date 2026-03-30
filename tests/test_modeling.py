"""
Tests for modeling/baseline.py, modeling/agglomerative_clustering.py,
and modeling/evaluate.py.
"""
import math

import numpy as np
import pandas as pd
import pytest

from modeling.baseline import BaselineGroupModel, GroupModel
from modeling.agglomerative_clustering import AggloGroupModel
from modeling.evaluate import Evaluator


# ═══════════════════════════════════════════════════════════════════════════════
# GroupModel / BaselineGroupModel
# ═══════════════════════════════════════════════════════════════════════════════

class TestBaselineGroupModel:
    def test_fit_returns_self(self, vecs_48):
        m = BaselineGroupModel()
        assert m.fit(vecs_48) is m

    def test_predict_assigns_all_patients(self, vecs_48):
        m = BaselineGroupModel()
        m.fit(vecs_48)
        labels = m.predict(vecs_48)
        assert len(labels) == 48
        # Every patient should appear in exactly one cluster
        all_pids = set()
        for members in m.clusters_.values():
            all_pids.update(members.index)
        assert all_pids == set(vecs_48.index)

    def test_groups_have_correct_size(self, vecs_48):
        m = BaselineGroupModel(group_size=12)
        m.fit(vecs_48)
        m.predict(vecs_48)
        # 48 / 12 = 4 groups, each exactly 12
        assert len(m.clusters_) == 4
        for members in m.clusters_.values():
            assert len(members) == 12

    def test_non_divisible_size(self, vecs_50):
        m = BaselineGroupModel(group_size=12)
        m.fit(vecs_50)
        m.predict(vecs_50)
        total = sum(len(m) for m in m.clusters_.values())
        assert total == 50

    def test_labels_property(self, vecs_48):
        m = BaselineGroupModel(group_size=12)
        m.fit(vecs_48)
        m.predict(vecs_48)
        labels = m.labels_
        assert isinstance(labels, pd.Series)
        assert labels.index.name == "patient_id"
        assert set(labels.index) == set(vecs_48.index)

    def test_assign_cluster_adds_to_existing(self, vecs_48, make_vecs):
        m = BaselineGroupModel(group_size=12)
        m.fit(vecs_48)
        m.predict(vecs_48)
        newcomer = make_vecs(1, seed=999)
        cid = m.assign_cluster(newcomer)
        assert isinstance(cid, int)
        assert len(m.clusters_[cid]) <= 12

    def test_assign_cluster_splits_when_full(self, vecs_48, make_vecs):
        """When all clusters are full, assigning should trigger a split."""
        m = BaselineGroupModel(group_size=12)
        m.fit(vecs_48)
        m.predict(vecs_48)
        # All 4 groups are full (12 each). Assigning 12 newcomers should cause splits.
        pre_n_groups = len(m.clusters_)
        for i in range(12):
            newcomer = make_vecs(1, seed=1000 + i)
            m.assign_cluster(newcomer)
        # Total patients should be 48 + 12 = 60
        total = sum(len(mem) for mem in m.clusters_.values())
        assert total == 60
        # At least one split should have occurred
        assert len(m.clusters_) > pre_n_groups

    def test_deterministic_with_same_seed(self, vecs_48):
        m1 = BaselineGroupModel(random_state=42)
        m1.fit(vecs_48)
        l1 = m1.predict(vecs_48)
        m2 = BaselineGroupModel(random_state=42)
        m2.fit(vecs_48)
        l2 = m2.predict(vecs_48)
        np.testing.assert_array_equal(l1, l2)


# ═══════════════════════════════════════════════════════════════════════════════
# AggloGroupModel
# ═══════════════════════════════════════════════════════════════════════════════

class TestAggloGroupModel:
    def test_fit_returns_self(self, vecs_48):
        m = AggloGroupModel(group_size=12)
        assert m.fit(vecs_48) is m

    def test_predict_assigns_all(self, vecs_48):
        m = AggloGroupModel(group_size=12)
        m.fit(vecs_48)
        labels = m.predict(vecs_48)
        assert len(labels) == 48
        total = sum(len(mem) for mem in m.clusters_.values())
        assert total == 48

    def test_no_cluster_exceeds_group_size(self, vecs_50):
        m = AggloGroupModel(group_size=12)
        m.fit(vecs_50)
        m.predict(vecs_50)
        for cid, members in m.clusters_.items():
            assert len(members) <= 12, f"Cluster {cid} has {len(members)} members"

    def test_feature_weights_applied(self, vecs_48):
        weights = {"numeric__a": 5.0, "numeric__b": 0.1}
        m = AggloGroupModel(group_size=12, feature_weights=weights)
        m.fit(vecs_48)
        assert m.weights_[0] == 5.0   # numeric__a
        assert m.weights_[1] == 0.1   # numeric__b
        assert m.weights_[2] == 1.0   # numeric__c (default)

    def test_predict_stores_agglo_model(self, vecs_48):
        m = AggloGroupModel(group_size=12)
        m.fit(vecs_48)
        m.predict(vecs_48)
        assert hasattr(m, "agglo_model_")
        assert hasattr(m.agglo_model_, "children_")

    def test_cluster_means_populated(self, vecs_48):
        m = AggloGroupModel(group_size=12)
        m.fit(vecs_48)
        m.predict(vecs_48)
        assert len(m.cluster_means_) == len(m.clusters_)
        for cid in m.clusters_:
            assert cid in m.cluster_means_

    def test_assign_cluster_returns_int(self, vecs_48, make_vecs):
        m = AggloGroupModel(group_size=12)
        m.fit(vecs_48)
        m.predict(vecs_48)
        newcomer = make_vecs(1, seed=777)
        cid = m.assign_cluster(newcomer)
        assert isinstance(cid, (int, np.integer))

    def test_assign_cluster_adds_patient(self, vecs_48, make_vecs):
        m = AggloGroupModel(group_size=12)
        m.fit(vecs_48)
        m.predict(vecs_48)
        pre_total = sum(len(mem) for mem in m.clusters_.values())
        newcomer = make_vecs(1, seed=777)
        m.assign_cluster(newcomer)
        post_total = sum(len(mem) for mem in m.clusters_.values())
        assert post_total == pre_total + 1

    def test_assign_cluster_respects_capacity(self, vecs_48, make_vecs):
        m = AggloGroupModel(group_size=12)
        m.fit(vecs_48)
        m.predict(vecs_48)
        newcomer = make_vecs(1, seed=777)
        cid = m.assign_cluster(newcomer)
        assert len(m.clusters_[cid]) <= 12

    def test_split_cluster(self, vecs_48):
        m = AggloGroupModel(group_size=12)
        m.fit(vecs_48)
        m.predict(vecs_48)
        cid = list(m.clusters_.keys())[0]
        pre_size = len(m.clusters_[cid])
        pre_n_clusters = len(m.clusters_)

        m._split_cluster(cid)

        assert len(m.clusters_) == pre_n_clusters + 1
        # Original + new should contain the same number of patients
        new_cid = max(m.clusters_.keys())
        assert len(m.clusters_[cid]) + len(m.clusters_[new_cid]) == pre_size

    def test_connectivity_matrix(self, vecs_48):
        """Passing a connectivity matrix should not error."""
        import scipy.sparse as sp
        n = len(vecs_48)
        # Simple k-nearest-neighbour style: connect adjacent indices
        rows = list(range(n - 1)) + list(range(1, n))
        cols = list(range(1, n)) + list(range(n - 1))
        conn = sp.csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(n, n)
        )
        m = AggloGroupModel(group_size=12, connectivity=conn)
        m.fit(vecs_48)
        labels = m.predict(vecs_48)
        assert len(labels) == 48

    def test_labels_property(self, vecs_48):
        m = AggloGroupModel(group_size=12)
        m.fit(vecs_48)
        m.predict(vecs_48)
        labels = m.labels_
        assert isinstance(labels, pd.Series)
        assert set(labels.index) == set(vecs_48.index)


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluator
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluator:
    @pytest.fixture()
    def fitted_agglo(self, vecs_48):
        m = AggloGroupModel(group_size=12)
        m.fit(vecs_48)
        m.predict(vecs_48)
        return m, vecs_48

    @pytest.fixture()
    def fitted_baseline(self, vecs_48):
        m = BaselineGroupModel(group_size=12)
        m.fit(vecs_48)
        m.predict(vecs_48)
        return m, vecs_48

    # ── WCSS ──

    def test_wcss_returns_tuple(self, fitted_agglo):
        model, X = fitted_agglo
        ev = Evaluator(model)
        total, per_cluster = ev.wcss(X)
        assert isinstance(total, float)
        assert isinstance(per_cluster, dict)
        assert total >= 0

    def test_wcss_total_equals_sum_of_parts(self, fitted_agglo):
        model, X = fitted_agglo
        ev = Evaluator(model)
        total, per_cluster = ev.wcss(X)
        assert abs(total - sum(per_cluster.values())) < 1e-8

    # ── Silhouette ──

    def test_silhouette_in_range(self, fitted_agglo):
        model, X = fitted_agglo
        ev = Evaluator(model)
        sil = ev.silhouette(X)
        assert -1 <= sil <= 1

    # ── Report ──

    def test_report_is_string(self, fitted_agglo):
        model, X = fitted_agglo
        ev = Evaluator(model)
        r = ev.report(X)
        assert isinstance(r, str)
        assert "WCSS" in r
        assert "Silhouette" in r

    def test_report_with_raw(self, fitted_agglo, raw_feats_48):
        model, X = fitted_agglo
        ev = Evaluator(model)
        r = ev.report(X, X_raw=raw_feats_48)
        assert isinstance(r, str)
        # Should contain cohesion info when X_raw is provided
        assert "Cohesion" in r or "uniform" in r.lower() or "Groups" in r

    # ── group_size_stats ──

    def test_group_size_stats_keys(self, fitted_agglo):
        model, X = fitted_agglo
        ev = Evaluator(model)
        stats = ev.group_size_stats()
        for key in ("n_groups", "min", "max", "mean", "std", "cv"):
            assert key in stats

    def test_group_size_stats_with_raw(self, fitted_agglo, raw_feats_48):
        model, X = fitted_agglo
        ev = Evaluator(model)
        stats = ev.group_size_stats(X_raw=raw_feats_48)
        assert "n_groups" in stats
        # Should have at least one cohesion key when raw data provided
        cohesion_keys = [k for k in stats if k.startswith("pct_")]
        assert len(cohesion_keys) > 0

    # ── feature_homogeneity ──

    def test_feature_homogeneity_shape(self, fitted_agglo):
        model, X = fitted_agglo
        ev = Evaluator(model)
        hom = ev.feature_homogeneity(X)
        assert isinstance(hom, pd.DataFrame)
        assert len(hom) == len(model.clusters_)

    # ── assignment_drift ──

    def test_assignment_drift(self, fitted_agglo, make_vecs):
        model, X = fitted_agglo
        ev = Evaluator(model)
        newcomers = make_vecs(5, seed=888)
        drift = ev.assignment_drift(newcomers)
        assert isinstance(drift, pd.DataFrame)
        assert "size_pre" in drift.columns
        assert "wcss_delta" in drift.columns
        assert drift["n_assigned"].sum() == 5

    def test_assignment_drift_timeline(self, fitted_agglo, make_vecs):
        model, X = fitted_agglo
        ev = Evaluator(model)
        newcomers = make_vecs(3, seed=888)
        ev.assignment_drift(newcomers)
        assert hasattr(ev, "wcss_timeline_")
        assert len(ev.wcss_timeline_) == 3

    # ── drift_report ──

    def test_drift_report(self, fitted_agglo, make_vecs):
        model, X = fitted_agglo
        ev = Evaluator(model)
        newcomers = make_vecs(5, seed=888)
        drift = ev.assignment_drift(newcomers)
        report = ev.drift_report(drift)
        assert isinstance(report, str)
        assert "Drift report" in report

    # ── Works with baseline model too ──

    def test_wcss_baseline(self, fitted_baseline):
        model, X = fitted_baseline
        ev = Evaluator(model)
        total, per_cluster = ev.wcss(X)
        assert total >= 0

    def test_assignment_drift_baseline(self, fitted_baseline, make_vecs):
        model, X = fitted_baseline
        ev = Evaluator(model)
        newcomers = make_vecs(3, seed=888)
        drift = ev.assignment_drift(newcomers)
        assert drift["n_assigned"].sum() == 3
