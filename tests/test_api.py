"""
Tests for the FastAPI application.

The /initialize endpoint runs the full pipeline including geocoding, which is
slow.  To keep the suite fast we initialise ONCE in a module-scoped fixture
and run the non-mutating tests first.  Mutating tests (delete, newcomer, split)
run last and each re-initialise as needed.
"""
import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.state import get_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _reset():
    state = get_state()
    state.vectorizer = None
    state.model = None
    state.evaluator = None
    state.all_vecs = None
    state.patient_data = None
    state.initialized_at = None
    state.n_initial_patients = 0


def _init(client: TestClient, csv_bytes: bytes):
    resp = client.post("/initialize", files={"file": ("data.csv", csv_bytes, "text/csv")})
    assert resp.status_code == 201, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# Module-scoped raw data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_df():
    return pd.read_csv("data/data.csv")


@pytest.fixture(scope="module")
def csv_60(raw_df) -> bytes:
    """First 60 rows — enough for several groups, one geocode run."""
    return _csv_bytes(raw_df.head(60))


@pytest.fixture(scope="module")
def newcomer_csv(raw_df) -> bytes:
    """Rows 60-64 for newcomer uploads."""
    return _csv_bytes(raw_df.iloc[60:65])


# ---------------------------------------------------------------------------
# Pre-init tests (no geocoding needed)
# ---------------------------------------------------------------------------

class TestBeforeInit:
    """Endpoints that should work (or return 503) before any initialisation."""

    @pytest.fixture(autouse=True)
    def _clean(self):
        _reset()
        yield
        _reset()

    def test_status_not_initialized(self):
        client = TestClient(app)
        resp = client.get("/status")
        assert resp.status_code == 200
        assert resp.json()["initialized"] is False

    @pytest.mark.parametrize("path", ["/metrics", "/groups", "/patients"])
    def test_503_before_init(self, path):
        client = TestClient(app)
        assert client.get(path).status_code == 503

    def test_bad_csv_returns_error(self):
        client = TestClient(app)
        resp = client.post(
            "/initialize",
            files={"file": ("bad.csv", b"not,valid\x00csv\x01data", "text/csv")},
        )
        assert resp.status_code in (400, 422)


# ---------------------------------------------------------------------------
# Read-only tests against a single initialised state (fast — no re-geocoding)
# ---------------------------------------------------------------------------

class TestReadOnly:
    """
    These tests share one initialisation to avoid repeated geocoding.
    They MUST NOT mutate state (no DELETE, no POST /newcomers, no POST /split).
    """

    @pytest.fixture(scope="class", autouse=True)
    def _init_once(self, csv_60):
        _reset()
        client = TestClient(app)
        _init(client, csv_60)
        yield
        _reset()

    @pytest.fixture()
    def client(self):
        return TestClient(app)

    # /status
    def test_status_initialized(self, client):
        body = client.get("/status").json()
        assert body["initialized"] is True
        assert body["n_patients"] > 0
        assert body["n_groups"] > 0
        assert body["silhouette"] is not None

    # /metrics
    def test_metrics(self, client):
        body = client.get("/metrics").json()
        assert body["wcss_total"] > 0
        assert -1 <= body["silhouette"] <= 1
        assert body["group_stats"]["n_groups"] > 0

    # /groups listing
    def test_list_groups(self, client):
        body = client.get("/groups").json()
        assert body["total_groups"] > 0
        g = body["groups"][0]
        for key in ("group_id", "size", "clinical", "demographic"):
            assert key in g

    # /groups/{id}
    def test_get_group_detail(self, client):
        groups = client.get("/groups").json()["groups"]
        gid = groups[0]["group_id"]
        body = client.get(f"/groups/{gid}").json()
        assert body["group_id"] == gid
        assert body["size"] == len(body["members"])
        assert body["wcss"] >= 0

    def test_get_group_not_found(self, client):
        assert client.get("/groups/999999").status_code == 404

    # /groups/{id}/metrics
    def test_group_metrics(self, client):
        gid = client.get("/groups").json()["groups"][0]["group_id"]
        body = client.get(f"/groups/{gid}/metrics").json()
        assert body["wcss"] >= 0

    # /groups/{id}/similar
    def test_similar_groups(self, client):
        gid = client.get("/groups").json()["groups"][0]["group_id"]
        body = client.get(f"/groups/{gid}/similar?n=3").json()
        assert body["group_id"] == gid
        dists = [sg["centroid_distance"] for sg in body["similar_groups"]]
        assert dists == sorted(dists)

    # /patients listing
    def test_list_patients(self, client):
        body = client.get("/patients").json()
        assert body["total"] > 0
        p = body["patients"][0]
        assert "patient_id" in p and "group_id" in p

    # /patients/{id}/group
    def test_patient_group_lookup(self, client):
        patients = client.get("/patients").json()["patients"]
        pid = patients[0]["patient_id"]
        body = client.get(f"/patients/{pid}/group").json()
        assert body["group_id"] == patients[0]["group_id"]

    # /patients/{id}
    def test_patient_detail(self, client):
        pid = client.get("/patients").json()["patients"][0]["patient_id"]
        body = client.get(f"/patients/{pid}").json()
        assert body["patient_id"] == pid
        assert "data" in body

    def test_patient_not_found(self, client):
        assert client.get("/patients/999999/group").status_code == 404

    # /initialize (re-init)
    def test_reinitialize(self, client, csv_60):
        """Re-initialising with the same data should succeed (uses cached geocodes)."""
        resp = client.post(
            "/initialize", files={"file": ("data.csv", csv_60, "text/csv")}
        )
        assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Mutating tests — each re-initialises from the same cached vectoriser state
# ---------------------------------------------------------------------------

class TestMutating:
    """
    Tests that change state (delete, split, newcomers).
    Each test re-initialises so mutations don't interfere.
    """

    @pytest.fixture(autouse=True)
    def _init_each(self, csv_60):
        _reset()
        client = TestClient(app)
        _init(client, csv_60)
        yield
        _reset()

    @pytest.fixture()
    def client(self):
        return TestClient(app)

    # ── Split ──

    def test_split_group(self, client):
        groups = client.get("/groups").json()["groups"]
        gid = next(g["group_id"] for g in groups if g["size"] >= 2)
        pre_size = next(g["size"] for g in groups if g["group_id"] == gid)

        body = client.post(f"/groups/{gid}/split").json()
        assert body["original_group_id"] == gid
        assert body["new_group_id"] != gid
        assert body["original_group_size"] + body["new_group_size"] == pre_size

    def test_split_not_found(self, client):
        assert client.post("/groups/999999/split").status_code == 404

    # ── Delete group with reassign ──

    def test_delete_group_reassign(self, client):
        groups = client.get("/groups").json()
        pre_total = groups["total_patients"]
        gid = groups["groups"][0]["group_id"]
        n_members = groups["groups"][0]["size"]

        body = client.delete(f"/groups/{gid}?reassign=true").json()
        assert body["removed_group_id"] == gid
        assert body["n_members"] == n_members
        assert body["reassigned_to"] is not None

        # Total patients preserved
        assert client.get("/groups").json()["total_patients"] == pre_total

    # ── Delete group without reassign ──

    def test_delete_group_orphan(self, client):
        groups = client.get("/groups").json()
        gid = groups["groups"][-1]["group_id"]
        n_members = groups["groups"][-1]["size"]
        pre_total = groups["total_patients"]

        body = client.delete(f"/groups/{gid}?reassign=false").json()
        assert body["n_members_orphaned"] == n_members
        assert client.get("/groups").json()["total_patients"] == pre_total - n_members

    def test_delete_group_not_found(self, client):
        assert client.delete("/groups/999999").status_code == 404

    # ── Remove patient ──

    def test_remove_patient(self, client):
        patients = client.get("/patients").json()
        pre_total = patients["total"]
        pid = patients["patients"][0]["patient_id"]
        gid = patients["patients"][0]["group_id"]

        body = client.delete(f"/patients/{pid}").json()
        assert body["patient_id"] == pid
        assert body["removed_from_group"] == gid

        # Patient should be gone
        assert client.get(f"/patients/{pid}/group").status_code == 404
        assert client.get("/patients").json()["total"] == pre_total - 1

    def test_remove_patient_not_found(self, client):
        assert client.delete("/patients/999999").status_code == 404

    # ── Newcomers ──

    def test_add_newcomers(self, client, newcomer_csv):
        pre_total = client.get("/patients").json()["total"]

        resp = client.post(
            "/newcomers", files={"file": ("new.csv", newcomer_csv, "text/csv")}
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["n_assigned"] > 0
        for a in body["assignments"]:
            assert a["group_size_after"] >= a["group_size_before"]
            assert "wcss_delta" in a
            assert "centroid_drift" in a

        assert client.get("/patients").json()["total"] == pre_total + body["n_assigned"]

    def test_newcomer_bad_csv(self, client):
        resp = client.post(
            "/newcomers", files={"file": ("bad.csv", b"garbage", "text/csv")}
        )
        assert resp.status_code in (400, 422)
