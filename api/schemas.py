from __future__ import annotations

from typing import Any
from pydantic import BaseModel


class ClinicalSummary(BaseModel):
    pct_high_risk: float
    pct_depressed: float
    pct_suicidal: float
    mean_stress_index: float
    mean_wellbeing_score: float
    mean_pressure: float
    mean_satisfaction: float
    mean_financial_stress: float


class DemographicSummary(BaseModel):
    pct_professional: float
    pct_male: float
    age_min: int
    age_max: int
    age_mean: float


class InitializeResponse(BaseModel):
    n_patients: int
    n_groups: int
    group_stats: dict[str, Any]
    silhouette: float
    wcss_total: float
    initialized_at: str


class SystemStatus(BaseModel):
    initialized: bool
    initialized_at: str | None = None
    n_patients: int = 0
    n_groups: int = 0
    group_stats: dict[str, Any] | None = None
    silhouette: float | None = None
    wcss_total: float | None = None
    wcss_mean_per_group: float | None = None


class SystemMetrics(BaseModel):
    silhouette: float
    wcss_total: float
    wcss_mean_per_group: float
    group_stats: dict[str, Any]


class GroupSummary(BaseModel):
    group_id: int
    size: int
    clinical: ClinicalSummary
    demographic: DemographicSummary


class GroupListResponse(BaseModel):
    groups: list[GroupSummary]
    total_groups: int
    total_patients: int


class GroupDetail(BaseModel):
    group_id: int
    size: int
    clinical: ClinicalSummary
    demographic: DemographicSummary
    members: list[dict[str, Any]]
    wcss: float
    mean_dist_to_centroid: float


class GroupMetrics(BaseModel):
    group_id: int
    size: int
    wcss: float
    mean_dist_to_centroid: float


class SplitGroupResponse(BaseModel):
    original_group_id: int
    new_group_id: int
    original_group_size: int
    new_group_size: int


class SimilarGroup(BaseModel):
    group_id: int
    centroid_distance: float


class SimilarGroupsResponse(BaseModel):
    group_id: int
    similar_groups: list[SimilarGroup]


class DeleteGroupResponse(BaseModel):
    removed_group_id: int
    n_members: int
    # present when reassign=true
    reassigned_to: dict[str, int] | None = None
    # present when reassign=false
    n_members_orphaned: int | None = None


class PatientGroupInfo(BaseModel):
    patient_id: Any
    group_id: int


class PatientListResponse(BaseModel):
    patients: list[PatientGroupInfo]
    total: int


class PatientDetail(BaseModel):
    patient_id: Any
    group_id: int
    data: dict[str, Any]


class RemovePatientResponse(BaseModel):
    patient_id: Any
    removed_from_group: int
    group_size_after: int


class NewcomerResult(BaseModel):
    patient_id: Any
    assigned_group_id: int
    group_size_before: int
    group_size_after: int
    wcss_before: float
    wcss_after: float
    wcss_delta: float
    centroid_drift: float


class NewcomerResponse(BaseModel):
    assignments: list[NewcomerResult]
    n_assigned: int
