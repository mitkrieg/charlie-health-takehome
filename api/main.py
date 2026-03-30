"""
Patient Group Assignment API
----------------------------
Wraps the Clinical + Attribute-Matched Connectivity Agglomerative Model.

Start with:
    uvicorn api.main:app --reload

Interactive docs: http://localhost:8000/docs
"""
from fastapi import FastAPI

from api.routers.groups import router as groups_router
from api.routers.patients import router as patients_router
from api.routers.system import router as system_router

app = FastAPI(
    title="Patient Group Assignment API",
    description=(
        "Assigns mental health patients to treatment cohorts of ~12 using an "
        "agglomerative clustering model with clinical feature weights and "
        "demographic attribute-matched connectivity constraints."
    ),
    version="1.0.0",
)

app.include_router(system_router)
app.include_router(groups_router)
app.include_router(patients_router)
