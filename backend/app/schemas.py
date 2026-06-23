"""Pydantic request/response schemas for the CausalIQ API."""

from typing import Dict, List, Optional

from pydantic import BaseModel


class CampaignCreate(BaseModel):
    name: str
    description: Optional[str] = None
    treatment_type: str  # "ad", "email", "discount"
    sample_size: int = 500000


class CampaignResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    treatment_type: str
    sample_size: int
    status: str  # "training", "complete", "failed"
    created_at: str


class ModelResult(BaseModel):
    model_name: str
    auuc: float
    qini: float
    mean_uplift: float


class CampaignResults(BaseModel):
    campaign_id: str
    best_model: str
    model_results: List[ModelResult]
    persuadables_count: int
    total_evaluated: int


class PersuadableCustomer(BaseModel):
    rank: int
    uplift_score: float
    features: Dict[str, float]


class PredictRequest(BaseModel):
    features: List[List[float]]


class PredictResponse(BaseModel):
    uplift_scores: List[float]
    persuadable: List[bool]
    model_used: str
