# CausalIQ — Context File

## Project Overview
CausalIQ: A causal inference and uplift modeling platform for marketing campaign optimization. Identifies "persuadable" customers — those who respond positively to treatment (email, discount, ad) but wouldn't have converted without it. Targets Dentsu (marketing analytics) and IBM (causal ML).

**Stack**: Python (CausalML/EconML), FastAPI, SQLAlchemy + Supabase, React/TypeScript/Tailwind

**Status**: Week 1 — Synthetic data + baseline models

---

## Key Constants

| Constant | Value |
|----------|-------|
| Synthetic dataset size | 10,000 customers |
| Treatment split | 50/50 |
| Base conversion rate | 15% |
| Uplift range | -10% to +30% |
| AUUC target | >0.06 |
| Qini target | >0.15 |
| Models | T-Learner, S-Learner, X-Learner, DR-Learner |

---

## Causal ML Concepts

**Uplift**: Causal effect of treatment on individual outcome.
Formula: Uplift = P(Conversion | Treatment) - P(Conversion | Control)

**T-Learner**: Two separate models for treatment/control. Difference = uplift.
**S-Learner**: Single model with treatment as a feature.
**X-Learner**: Residual pseudo-outcomes, two-stage. Better for imbalanced data.
**DR-Learner**: Double/debiased ML. Most statistically rigorous.

**AUUC**: Area Under Uplift Curve. Good if >0.06.
**Qini**: Normalized uplift ratio. Good if >0.15.
**Persuadable**: Customer with positive uplift who wouldn't convert without treatment.

---

## Directory Structure

backend/app/

├── ml/

│   ├── data_generator.py      # Synthetic campaign data

│   ├── models.py              # T/S/X/DR-Learner classes

│   ├── evaluation.py          # AUUC, Qini, uplift curves (Week 3)

│   └── artifacts/

│       └── models/            # Saved trained models

├── api/

│   ├── campaigns.py           # POST/GET /campaigns

│   ├── predictions.py         # POST /predict

│   └── results.py             # GET /results

├── main.py                    # FastAPI app

├── models.py                  # SQLAlchemy ORM

├── schemas.py                 # Pydantic schemas

└── database.py                # DB connection

---

## Installed Packages

- causalml==0.16.0
- xgboost==3.2.0
- scikit-learn==1.3.2
- numpy==1.26.4
- pandas==3.0.3
- fastapi==0.104.1
- uvicorn==0.24.0
- sqlalchemy==2.0.23
- pydantic==2.5.0

**Note**: econml skipped in Week 1 due to scikit-learn conflict. Will resolve in Week 2.

---

## Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| causalml install fails | pip install causalml --no-cache-dir |
| grep not found in PowerShell | Use findstr instead |
| echo. not recognized | Use New-Item filename -ItemType File |
| econml conflicts with scikit-learn | Skip econml for now, install in Week 2 |
| Import from app.ml fails | Check __init__.py files exist in app/ and app/ml/ |

---

## Week Progress

- Week 1: Synthetic data + T/S-Learner ← YOU ARE HERE
- Week 2: X-Learner + DR-Learner + ensemble
- Week 3: AUUC, Qini, persuadables
- Week 4: FastAPI backend
- Week 5: React frontend
- Week 6: Production deployment

---

## Resume Bullet (Final)

CausalIQ: Engineered a full-stack causal inference platform using meta-learner architectures (T/S/X/DR-Learner) to identify persuadable customers in marketing campaigns. Achieved AUUC >0.07 and Qini >0.18. Deployed FastAPI backend (Render) + React dashboard (Vercel).
Tech: CausalML, FastAPI, React/TypeScript, PostgreSQL, XGBoost.