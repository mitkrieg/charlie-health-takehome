"""
GET    /patients                   — all patients with their group assignments
GET    /patients/{id}              — single patient detail + group
GET    /patients/{id}/group        — lightweight: which group is this patient in?
DELETE /patients/{id}              — remove a patient from their group

POST   /newcomers                  — add one or more patients from a CSV upload;
                                     returns per-patient assignment + group impact
"""
from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas import (
    NewcomerResponse,
    NewcomerResult,
    PatientDetail,
    PatientGroupInfo,
    PatientListResponse,
    RemovePatientResponse,
)
from api.utils import (
    cluster_wcss,
    require_initialized,
    serialise_patient,
)

router = APIRouter(tags=["patients"])


# ── List all patients ──────────────────────────────────────────────────────────

@router.get("/patients", response_model=PatientListResponse)
def list_patients():
    state = require_initialized()
    labels = state.model.labels_
    patients = [PatientGroupInfo(patient_id=pid, group_id=int(gid)) for pid, gid in labels.items()]
    return PatientListResponse(patients=patients, total=len(patients))


# ── Patient group lookup ───────────────────────────────────────────────────────

@router.get("/patients/{patient_id}/group", response_model=PatientGroupInfo)
def get_patient_group(patient_id: int):
    state = require_initialized()
    labels = state.model.labels_
    if patient_id not in labels.index:
        raise HTTPException(404, f"Patient {patient_id} not found.")
    return PatientGroupInfo(patient_id=patient_id, group_id=int(labels[patient_id]))


# ── Patient detail ─────────────────────────────────────────────────────────────

@router.get("/patients/{patient_id}", response_model=PatientDetail)
def get_patient(patient_id: int):
    state = require_initialized()
    labels = state.model.labels_
    if patient_id not in labels.index:
        raise HTTPException(404, f"Patient {patient_id} not found.")
    group_id = int(labels[patient_id])
    data: dict = {}
    if patient_id in state.patient_data.index:
        data = serialise_patient(patient_id, state.patient_data.loc[patient_id])
        data.pop("patient_id", None)
    return PatientDetail(patient_id=patient_id, group_id=group_id, data=data)


# ── Remove patient ─────────────────────────────────────────────────────────────

@router.delete("/patients/{patient_id}", response_model=RemovePatientResponse)
def remove_patient(patient_id: int):
    state = require_initialized()
    labels = state.model.labels_
    if patient_id not in labels.index:
        raise HTTPException(404, f"Patient {patient_id} not found.")

    group_id = int(labels[patient_id])
    members = state.model.clusters_[group_id]

    # Drop the patient from their cluster
    state.model.clusters_[group_id] = members.drop(index=patient_id)

    if len(state.model.clusters_[group_id]) == 0:
        # Cluster is now empty — remove it entirely
        del state.model.clusters_[group_id]
        state.model.cluster_means_.pop(group_id, None)
    else:
        state.model._update_cluster_means()

    # Remove from human-readable store
    if patient_id in state.patient_data.index:
        state.patient_data = state.patient_data.drop(index=patient_id)

    remaining = len(state.model.clusters_.get(group_id, pd.DataFrame()))
    return RemovePatientResponse(
        patient_id=patient_id,
        removed_from_group=group_id,
        group_size_after=remaining,
    )


# ── Add newcomers ──────────────────────────────────────────────────────────────

@router.post("/newcomers", response_model=NewcomerResponse, status_code=201)
async def add_newcomers(file: UploadFile = File(...)):
    """
    Upload a CSV of new patients (same column format as the training data).
    Each patient is assigned online to the nearest cluster with capacity.
    Returns per-patient assignment info and the impact on their cluster.
    """
    state = require_initialized()
    content = await file.read()
    try:
        df_new = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(400, f"Could not parse CSV: {exc}")

    try:
        newcomer_vecs: pd.DataFrame = state.vectorizer.transform(df_new)
        newcomer_feats: pd.DataFrame = state.vectorizer.pipeline.steps[0][1].transform(df_new)
    except Exception as exc:
        raise HTTPException(422, f"Feature transformation failed: {exc}")

    # Reindex newcomers so patient_ids don't collide with existing patients
    existing_ids = set()
    for members in state.model.clusters_.values():
        existing_ids.update(members.index)
    next_id = max(existing_ids, default=-1) + 1
    new_ids = list(range(next_id, next_id + len(newcomer_vecs)))
    newcomer_vecs.index = pd.Index(new_ids, name="patient_id")
    newcomer_feats.index = pd.Index(new_ids, name="patient_id")

    assignments: list[NewcomerResult] = []

    for pid in newcomer_vecs.index:
        row_vec = newcomer_vecs.loc[[pid]]

        # Snapshot the target cluster BEFORE assignment (we don't know it yet)
        # Capture all cluster means so we can measure drift after
        pre_means = {
            cid: state.model.cluster_means_[cid].copy()
            for cid in state.model.clusters_
        }
        pre_sizes = {cid: len(m) for cid, m in state.model.clusters_.items()}
        pre_wcss = {
            cid: cluster_wcss(state.model.clusters_[cid])[0]
            for cid in state.model.clusters_
        }

        # assign_cluster mutates model.clusters_ and model.cluster_means_
        assigned_cid = state.model.assign_cluster(row_vec)

        post_wcss, _ = cluster_wcss(state.model.clusters_[assigned_cid])
        post_size = len(state.model.clusters_[assigned_cid])

        wcss_before = pre_wcss.get(assigned_cid, 0.0)
        size_before = pre_sizes.get(assigned_cid, 0)

        if assigned_cid in pre_means:
            post_mean = state.model.cluster_means_[assigned_cid].values
            centroid_drift = float(
                np.linalg.norm(post_mean - pre_means[assigned_cid].values)
            )
        else:
            centroid_drift = 0.0  # brand-new cluster from a split

        # Store the newcomer's raw features
        if pid in newcomer_feats.index:
            state.patient_data = pd.concat(
                [state.patient_data, newcomer_feats.loc[[pid]]]
            )

        wcss_delta = post_wcss - wcss_before
        assignments.append(
            NewcomerResult(
                patient_id=pid,
                assigned_group_id=assigned_cid,
                group_size_before=size_before,
                group_size_after=post_size,
                wcss_before=round(wcss_before, 4),
                wcss_after=round(post_wcss, 4),
                wcss_delta=round(wcss_delta, 4),
                centroid_drift=round(centroid_drift, 6),
            )
        )

    return NewcomerResponse(assignments=assignments, n_assigned=len(assignments))
