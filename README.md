# CausalIQ — Causal Inference & Uplift Modeling Platform

> Identifying **persuadable customers** in marketing campaigns using causal ML — going beyond correlation to measure true treatment effects.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB)](https://react.dev)
[![Deployed](https://img.shields.io/badge/Deployed-Render%20%2B%20Vercel-brightgreen)](https://causal-iq.vercel.app)

**Live Demo**: https://causal-iq.vercel.app
**API Docs**: https://causaliq-backend.onrender.com/docs
**GitHub**: https://github.com/DharshanKumar1010/CausalIQ

---

## The Problem

Traditional marketing analytics measures *who converted* — not *why they converted*. A customer who would have bought anyway doesn't need a discount. Sending ads to "sure things" wastes budget. The real target is the **persuadable**: someone who converts *because of* the intervention.

CausalIQ uses causal inference to estimate **Individual Treatment Effects (ITE)** — the true causal impact of showing an ad on each customer's conversion probability.

---

## Dataset

**Criteo Uplift Dataset v2.1** — Industry standard benchmark for uplift modeling

| Metric | Value |
|--------|-------|
| Total rows | 13,979,592 |
| Features | 12 anonymized ad features (f0-f11) |
| Treatment | Binary (ad shown / not shown) |
| Outcome | Conversion (visit / purchase) |
| Treatment ratio | 85% treated |
| Conversion rate | 0.29% |
| Source | Criteo AI Lab (AdKDD 2018 Workshop) |

Training uses a balanced 500k sample (250k treated, 250k control) for computational efficiency while preserving realistic signal.

---

## ML Architecture

### Meta-Learner Models

| Model | Approach | Strength |
|-------|----------|----------|
| **T-Learner** | Separate models per group | Simple, interpretable baseline |
| **S-Learner** | Single model + treatment feature | Shared representations |
| **X-Learner** | Residual pseudo-outcomes | Better heterogeneity detection |
| **DR-Learner** | Doubly robust estimation | Most statistically rigorous ✅ Best |

### Evaluation Metrics

| Metric | Description | Best Model |
|--------|-------------|------------|
| **AUUC** | Area Under Uplift Curve | DR-Learner: 0.000026 |
| **Qini Coefficient** | Normalized uplift ratio | DR-Learner: 0.063626 |
| **Persuadables** | Top 10% by uplift score | 15,000 customers |

### Why DR-Learner Wins
The Doubly Robust Learner combines outcome modeling with propensity weighting — if either component is correctly specified, the estimator remains unbiased. On real ad-tech data with low conversion rates (0.245%), this robustness is critical.

---

## Results
```
============================================================
  CAUSALIQ: All 4 Meta-Learners on Criteo
  Train: 349,999 | Test: 150,001

  Treatment ratio: 50.0% | Conversion rate: 0.245%

  T_LEARNER:   Mean uplift: 0.000420 | Qini: -0.198416
  S_LEARNER:   Mean uplift: 0.000352 | Qini:  0.049152
  X_LEARNER:   Mean uplift: 0.000552 | Qini: -0.051753
  DR_LEARNER:  Mean uplift: 0.000599 | Qini:  0.063626 ← BEST
  ENSEMBLE:    Mean uplift: 0.000481

  Persuadables identified: 15,000 customers (top 10%)
  Key features: f10 (+0.15), f9 (+0.14), f7 (+0.07)
============================================================
```

---

## Tech Stack

### Backend
- **FastAPI** — REST API with automatic Swagger docs
- **CausalML** — T/S/X-Learner implementations
- **XGBoost** — Base learner for all meta-learners
- **scikit-learn** — Data preprocessing, train/test split
- **pandas / numpy** — Data manipulation
- **matplotlib** — Uplift curve visualization

### Frontend
- **React 18 + TypeScript** — Component-based UI
- **Tailwind CSS** — Dark theme styling
- **Recharts** — Qini coefficient bar chart
- **Axios** — API communication
- **React Router** — Client-side routing

### Infrastructure
- **Render** — FastAPI backend hosting
- **Vercel** — React frontend hosting
- **GitHub** — Version control with weekly commits

---

## Architecture
```
┌─────────────────────────────────────────────────────┐
│                   React Frontend                     │
│         (Vercel — causal-iq.vercel.app)             │
│  Dashboard → Create Campaign → View Results         │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP (Axios)
┌──────────────────────▼──────────────────────────────┐
│                  FastAPI Backend                     │
│      (Render — causaliq-backend.onrender.com)       │
│                                                     │
│  POST /api/campaigns  →  Train 4 models             │
│  GET  /api/campaigns/{id}/results  →  AUUC/Qini     │
│  GET  /api/campaigns/{id}/persuadables  →  Top 10%  │
│  POST /api/campaigns/{id}/predict  →  New customers │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                   ML Pipeline                        │
│                                                     │
│  Criteo v2.1 (13.9M rows)                          │
│       ↓ Balance sample (500k)                       │
│       ↓ Train/Test split (70/30)                    │
│       ↓ T/S/X/DR-Learner training                  │
│       ↓ AUUC + Qini evaluation                      │
│       ↓ Persuadable identification (top 10%)        │
└─────────────────────────────────────────────────────┘
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/campaigns` | Create campaign & train all 4 models |
| `GET` | `/api/campaigns` | List all campaigns |
| `GET` | `/api/campaigns/{id}` | Get campaign details |
| `GET` | `/api/campaigns/{id}/results` | AUUC, Qini, model comparison |
| `GET` | `/api/campaigns/{id}/persuadables` | Top persuadable customers |
| `POST` | `/api/campaigns/{id}/predict` | Predict uplift for new customers |

Full interactive docs: https://causaliq-backend.onrender.com/docs

---

## Running Locally

### Prerequisites
- Python 3.11
- Node.js 18+
- Git

### Backend Setup
```bash
git clone https://github.com/DharshanKumar1010/CausalIQ.git
cd CausalIQ

# Create virtual environment
python -m venv .venv
.venv/Scripts/Activate.ps1  # Windows
source .venv/bin/activate    # Mac/Linux

# Install dependencies
pip install -r backend/requirements_deploy.txt

# Download Criteo dataset (13.9M rows)
python prepare_data.py

# Run backend
cd backend
uvicorn app.main:app --reload
# API running at http://localhost:8000
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
# Frontend running at http://localhost:5173
```

---

## Project Structure
```
CausalIQ/
├── backend/
│   ├── app/
│   │   ├── ml/
│   │   │   ├── data_loader.py      # Criteo data loading
│   │   │   ├── data_generator.py   # Synthetic data (testing)
│   │   │   ├── models.py           # T/S/X/DR-Learner + Ensemble
│   │   │   └── evaluation.py       # AUUC, Qini, persuadables
│   │   ├── api/
│   │   │   └── campaigns.py        # REST endpoints
│   │   ├── main.py                 # FastAPI app
│   │   └── schemas.py              # Pydantic models
│   └── requirements_deploy.txt
├── frontend/
│   └── src/
│       ├── pages/
│       │   ├── Dashboard.tsx       # Campaign creation
│       │   └── CampaignResults.tsx # Results visualization
│       └── services/
│           └── api.ts              # API calls
├── data/
│   └── processed/
│       └── criteo_deploy_sample.csv
└── prepare_data.py                 # Dataset preparation
```

---

## Key Concepts

**Uplift Modeling**: Predicting the *causal effect* of treatment rather than just outcome probability. A customer with 80% purchase probability regardless of ads is less valuable to target than one who goes from 2% to 20% when shown an ad.

**Heterogeneous Treatment Effects (HTE)**: The insight that treatment effects vary by individual. CausalIQ's meta-learners capture this heterogeneity to rank customers by their personal response to advertising.

**Persuadables**: The top 10% of customers ranked by predicted uplift — those most likely to convert *because of* the ad. Targeting persuadables instead of likely-converters improves ROI by eliminating wasted spend on sure-things and sleeping dogs.

---

## Portfolio Context

CausalIQ is the third project in a three-part ML portfolio targeting Dentsu and IBM internships:

| Project | ML Domain | Dataset | Key Metric |
|---------|-----------|---------|------------|
| [ChurnIQ](https://github.com/DharshanKumar1010/ChurnIQ) | Supervised (XGBoost) | Telco Churn (7k) | AUC-ROC: 0.9179 |
| [SensorIQ](https://github.com/DharshanKumar1010/SensorIQ) | Unsupervised (LSTM) | NASA C-MAPSS | F1: 0.575 |
| **CausalIQ** | **Causal Inference** | **Criteo (13.9M)** | **Qini: 0.064** |

---

## References

- Künzel et al. (2019): *Metalearners for Estimating Heterogeneous Treatment Effects*
- Diemert et al. (2018): *A Large Scale Benchmark for Uplift Modeling* (AdKDD 2018)
- Nie & Wager (2021): *Quasi-oracle Estimation of Heterogeneous Treatment Effects*

---

## Author

**Dharshan Kumar** — B.Tech CSE (Data Science), SRM IST Kattankulatham, 2027
- GitHub: [@DharshanKumar1010](https://github.com/DharshanKumar1010)

---

*Built over 6 weeks as part of a structured ML portfolio targeting marketing analytics and causal ML roles.*
