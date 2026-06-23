"""Campaign endpoints for CausalIQ — training, evaluation, and prediction."""

import uuid
from datetime import datetime
from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException
from sklearn.model_selection import train_test_split

from app.ml.data_loader import load_criteo_data
from app.ml.evaluation import evaluate_all_models
from app.ml.models import UpliftEnsemble
from app.schemas import (
    CampaignCreate,
    CampaignResponse,
    CampaignResults,
    ModelResult,
    PersuadableCustomer,
    PredictRequest,
    PredictResponse,
)

FEATURE_NAMES = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11']

campaigns_store: dict = {}
results_store: dict = {}
persuadables_store: dict = {}

router = APIRouter(prefix="/api/campaigns", tags=["campaigns"])


@router.post("", response_model=CampaignResponse)
def create_campaign(body: CampaignCreate) -> CampaignResponse:
    """Train all four meta-learners on Criteo data and store evaluation results.

    Loads the Criteo dataset, splits 70/30, trains the UpliftEnsemble, runs
    full AUUC/Qini evaluation, and persists the fitted models for later prediction.
    """
    campaign_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()

    campaigns_store[campaign_id] = {
        'id': campaign_id,
        'name': body.name,
        'description': body.description,
        'treatment_type': body.treatment_type,
        'sample_size': body.sample_size,
        'status': 'training',
        'created_at': created_at,
    }

    try:
        X, treatment, y, feature_names = load_criteo_data(
            '../data/processed/criteo_sample.csv', seed=42
        )

        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42)
        X_train, X_test = X[train_idx], X[test_idx]
        treatment_train, treatment_test = treatment[train_idx], treatment[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        ensemble = UpliftEnsemble()
        ensemble.fit_all(X_train, treatment_train, y_train)

        predictions = ensemble.predict_all(X_test)

        eval_results = evaluate_all_models(
            predictions, y_test, treatment_test, X_test, FEATURE_NAMES
        )

        best_model_name = eval_results['best_model']
        persuadable_idx = eval_results['persuadable_idx']
        persuadable_scores = eval_results['persuadable_scores']

        results_store[campaign_id] = {
            'eval': eval_results,
            'best_model_name': best_model_name,
            'best_model_obj': ensemble.models[best_model_name],
            'predictions': predictions,
            'X_test': X_test,
            'y_test': y_test,
            'treatment_test': treatment_test,
        }

        persuadables_store[campaign_id] = [
            PersuadableCustomer(
                rank=rank + 1,
                uplift_score=float(persuadable_scores[rank]),
                features={
                    name: float(X_test[persuadable_idx[rank]][i])
                    for i, name in enumerate(FEATURE_NAMES)
                },
            )
            for rank in range(len(persuadable_idx))
        ]

        campaigns_store[campaign_id]['status'] = 'complete'

    except Exception as exc:
        campaigns_store[campaign_id]['status'] = 'failed'
        raise HTTPException(status_code=500, detail=str(exc))

    c = campaigns_store[campaign_id]
    return CampaignResponse(
        id=c['id'],
        name=c['name'],
        description=c['description'],
        treatment_type=c['treatment_type'],
        sample_size=c['sample_size'],
        status=c['status'],
        created_at=c['created_at'],
    )


@router.get("", response_model=List[CampaignResponse])
def list_campaigns() -> List[CampaignResponse]:
    """Return a list of all campaigns in the in-memory store."""
    return [
        CampaignResponse(
            id=c['id'],
            name=c['name'],
            description=c['description'],
            treatment_type=c['treatment_type'],
            sample_size=c['sample_size'],
            status=c['status'],
            created_at=c['created_at'],
        )
        for c in campaigns_store.values()
    ]


@router.get("/{campaign_id}", response_model=CampaignResponse)
def get_campaign(campaign_id: str) -> CampaignResponse:
    """Return a single campaign by ID. Raises 404 if not found."""
    c = campaigns_store.get(campaign_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return CampaignResponse(
        id=c['id'],
        name=c['name'],
        description=c['description'],
        treatment_type=c['treatment_type'],
        sample_size=c['sample_size'],
        status=c['status'],
        created_at=c['created_at'],
    )


@router.get("/{campaign_id}/results", response_model=CampaignResults)
def get_campaign_results(campaign_id: str) -> CampaignResults:
    """Return AUUC, Qini, and persuadable summary for a completed campaign."""
    if campaign_id not in campaigns_store:
        raise HTTPException(status_code=404, detail="Campaign not found")
    stored = results_store.get(campaign_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Results not available yet")

    eval_results = stored['eval']['results']
    best_model = stored['best_model_name']
    X_test = stored['X_test']
    persuadable_idx = stored['eval']['persuadable_idx']

    model_results = [
        ModelResult(
            model_name=name,
            auuc=res['auuc'],
            qini=res['qini'],
            mean_uplift=float(res['uplift_scores'].mean()),
        )
        for name, res in eval_results.items()
    ]

    return CampaignResults(
        campaign_id=campaign_id,
        best_model=best_model,
        model_results=model_results,
        persuadables_count=len(persuadable_idx),
        total_evaluated=len(X_test),
    )


@router.get("/{campaign_id}/persuadables", response_model=List[PersuadableCustomer])
def get_persuadables(campaign_id: str, limit: int = 20) -> List[PersuadableCustomer]:
    """Return the top N persuadable customers ranked by predicted uplift score."""
    if campaign_id not in campaigns_store:
        raise HTTPException(status_code=404, detail="Campaign not found")
    persuadables = persuadables_store.get(campaign_id)
    if persuadables is None:
        raise HTTPException(status_code=404, detail="Persuadables not available yet")
    return persuadables[:limit]


@router.post("/{campaign_id}/predict", response_model=PredictResponse)
def predict(campaign_id: str, body: PredictRequest) -> PredictResponse:
    """Run uplift prediction on new feature vectors using the campaign's best model."""
    if campaign_id not in campaigns_store:
        raise HTTPException(status_code=404, detail="Campaign not found")
    stored = results_store.get(campaign_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Model not ready")

    try:
        X_input = np.array(body.features, dtype=np.float32)
        best_model_obj = stored['best_model_obj']
        scores = best_model_obj.predict(X_input)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return PredictResponse(
        uplift_scores=[float(s) for s in scores],
        persuadable=[bool(s > 0) for s in scores],
        model_used=stored['best_model_name'],
    )
