"""
GET    /groups                           — list all groups
GET    /groups/{id}                      — group detail + member list
DELETE /groups/{id}?reassign=true|false  — remove group (optionally reassign members)
POST   /groups/{id}/split                — manually split a group in two
GET    /groups/{id}/similar?n=5          — n most similar groups by centroid distance
GET    /groups/{id}/metrics              — per-group quality metrics
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from api.schemas import (
    DeleteGroupResponse,
    GroupDetail,
    GroupListResponse,
    GroupMetrics,
    GroupSummary,
    SimilarGroup,
    SimilarGroupsResponse,
    SplitGroupResponse,
)
from api.state import AppState
from api.utils import (
    clinical_summary,
    cluster_wcss,
    demographic_summary,
    get_members_raw,
    require_initialized,
    serialise_patient,
)

router = APIRouter(prefix="/groups", tags=["groups"])


def _require(group_id: int, state: AppState) -> None:
    if group_id not in state.model.clusters_:
        raise HTTPException(404, f"Group {group_id} not found.")


@router.get("", response_model=GroupListResponse)
def list_groups():
    state = require_initialized()
    summaries = []
    for gid in sorted(state.model.clusters_):
        members_raw = get_members_raw(gid, state)
        summaries.append(
            GroupSummary(
                group_id=gid,
                size=len(state.model.clusters_[gid]),
                clinical=clinical_summary(members_raw),
                demographic=demographic_summary(members_raw),
            )
        )
    total_patients = sum(len(m) for m in state.model.clusters_.values())
    return GroupListResponse(
        groups=summaries,
        total_groups=len(summaries),
        total_patients=total_patients,
    )


@router.get("/{group_id}", response_model=GroupDetail)
def get_group(group_id: int):
    state = require_initialized()
    _require(group_id, state)

    members_vecs = state.model.clusters_[group_id]
    members_raw = get_members_raw(group_id, state)
    wcss, mean_dist = cluster_wcss(members_vecs)

    members_out = []
    for pid in members_vecs.index:
        if pid in state.patient_data.index:
            members_out.append(serialise_patient(pid, state.patient_data.loc[pid]))

    return GroupDetail(
        group_id=group_id,
        size=len(members_vecs),
        clinical=clinical_summary(members_raw),
        demographic=demographic_summary(members_raw),
        members=members_out,
        wcss=round(wcss, 4),
        mean_dist_to_centroid=round(mean_dist, 4),
    )


@router.delete("/{group_id}", response_model=DeleteGroupResponse)
def delete_group(
    group_id: int,
    reassign: bool = Query(
        True, description="Reassign members to nearest remaining group"
    ),
):
    state = require_initialized()
    _require(group_id, state)

    members_vecs = state.model.clusters_.pop(group_id)
    state.model.cluster_means_.pop(group_id, None)
    n_members = len(members_vecs)

    # Also remove from patient_data tracking
    orphaned_ids = list(members_vecs.index)

    if not reassign or len(state.model.clusters_) == 0:
        # Remove from patient_data too
        ids_to_drop = state.patient_data.index.intersection(orphaned_ids)
        state.patient_data = state.patient_data.drop(index=ids_to_drop)
        return DeleteGroupResponse(
            removed_group_id=group_id,
            n_members=n_members,
            n_members_orphaned=n_members,
        )

    # Reassign members one-by-one to nearest cluster
    reassigned: dict[str, int] = {}
    for pid in members_vecs.index:
        row = members_vecs.loc[[pid]]
        new_cid = state.model.assign_cluster(row)
        reassigned[str(pid)] = new_cid

    return DeleteGroupResponse(
        removed_group_id=group_id,
        n_members=n_members,
        reassigned_to=reassigned,
    )


@router.post("/{group_id}/split", response_model=SplitGroupResponse)
def split_group(group_id: int):
    state = require_initialized()
    _require(group_id, state)

    if len(state.model.clusters_[group_id]) < 2:
        raise HTTPException(
            400, f"Group {group_id} has fewer than 2 members; cannot split."
        )

    pre_cids = set(state.model.clusters_.keys())
    state.model._split_cluster(group_id)
    post_cids = set(state.model.clusters_.keys())

    new_cids = post_cids - pre_cids
    new_cid = new_cids.pop() if new_cids else group_id  # fallback (shouldn't happen)

    return SplitGroupResponse(
        original_group_id=group_id,
        new_group_id=new_cid,
        original_group_size=len(state.model.clusters_[group_id]),
        new_group_size=len(state.model.clusters_[new_cid]),
    )


@router.get("/{group_id}/similar", response_model=SimilarGroupsResponse)
def similar_groups(
    group_id: int,
    n: int = Query(5, ge=1, le=50, description="Number of similar groups to return"),
):
    state = require_initialized()
    _require(group_id, state)

    target_mean = state.model.cluster_means_[group_id].values * state.model.weights_
    distances = []
    for other_id, other_mean in state.model.cluster_means_.items():
        if other_id == group_id:
            continue
        dist = float(
            np.linalg.norm(target_mean - other_mean.values * state.model.weights_)
        )
        distances.append(
            SimilarGroup(group_id=other_id, centroid_distance=round(dist, 4))
        )

    distances.sort(key=lambda x: x.centroid_distance)
    return SimilarGroupsResponse(group_id=group_id, similar_groups=distances[:n])


@router.get("/{group_id}/metrics", response_model=GroupMetrics)
def group_metrics(group_id: int):
    state = require_initialized()
    _require(group_id, state)

    members_vecs = state.model.clusters_[group_id]
    wcss, mean_dist = cluster_wcss(members_vecs)
    return GroupMetrics(
        group_id=group_id,
        size=len(members_vecs),
        wcss=round(wcss, 4),
        mean_dist_to_centroid=round(mean_dist, 4),
    )
