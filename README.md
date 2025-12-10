# 🏦 Credit Risk Prediction API

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![GCP](https://img.shields.io/badge/GCP-Cloud_Run-4285F4?logo=google-cloud&logoColor=white)](https://cloud.google.com/run)
[![Kaggle](https://img.shields.io/badge/Kaggle-77.2%25-20BEFF?logo=kaggle)](https://www.kaggle.com/c/home-credit-default-risk)
[![ROC AUC](https://img.shields.io/badge/ROC_AUC-0.785-success)](/)

An end-to-end machine learning system for credit default prediction, featuring a production-ready REST API. Built with LightGBM + XGBoost ensemble model, deployed on Google Cloud Run.

## 🌐 Live Demo

**API Endpoint:** [https://credit-risk-api-xxxxx.europe-west1.run.app](https://credit-risk-api-xxxxx.europe-west1.run.app)

**Swagger Docs:** [https://credit-risk-api-xxxxx.europe-west1.run.app/docs](https://credit-risk-api-xxxxx.europe-west1.run.app/docs)

> *Update the URLs after redeployment*

---

## 📋 Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Project Architecture](#project-architecture)
- [API Documentation](#api-documentation)
- [Model Performance](#model-performance)
- [Feature Engineering](#feature-engineering)
- [Key Findings](#key-findings)
- [Installation](#installation)
- [Deployment](#deployment)
- [Project Structure](#project-structure)

---

## Overview

This project tackles the **Home Credit Default Risk** Kaggle competition challenge: predicting which loan applicants are likely to default. The solution includes:

- **Comprehensive EDA** across 8 interconnected datasets
- **580 engineered features** from customer behavioral and financial data
- **Ensemble model** combining LightGBM and XGBoost
- **Production REST API** with FastAPI
- **Cloud deployment** on Google Cloud Run with Docker

### Business Impact

The model helps financial institutions:
- Identify high-risk loan applicants early
- Reduce default rates and financial losses
- Make data-driven lending decisions
- Automate credit scoring at scale

---

## Tech Stack

### Machine Learning
| Tool | Purpose |
|------|---------|
| **LightGBM** | Gradient boosting, feature importance |
| **XGBoost** | Gradient boosting, ensemble component |
| **scikit-learn** | Preprocessing, metrics, VotingClassifier |
| **pandas / numpy** | Data manipulation |
| **Optuna** | Hyperparameter optimization |

### API & Deployment
| Tool | Purpose |
|------|---------|
| **FastAPI** | REST API framework |
| **Uvicorn** | ASGI server |
| **Docker** | Containerization |
| **Google Cloud Run** | Serverless deployment |

### Data Analysis
| Tool | Purpose |
|------|---------|
| **matplotlib / seaborn** | Visualization |
| **scipy** | Statistical analysis |

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA SOURCES                                   │
│  application_train.csv │ bureau.csv │ previous_application.csv │ etc.      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXPLORATORY DATA ANALYSIS                           │
│         • application_eda.ipynb - Core applicant analysis                   │
│         • bureau_eda.ipynb - External credit history                        │
│         • previous_application_eda.ipynb - Historical loan patterns         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FEATURE ENGINEERING                                 │
│         • 580 engineered features across all datasets                       │
│         • Aggregations, ratios, temporal features                           │
│         • Missing value handling, outlier treatment                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODEL TRAINING                                    │
│         • LightGBM + XGBoost individual models                              │
│         • Feature selection (top 198 features)                              │
│         • VotingClassifier ensemble (soft voting)                           │
│         • Cross-validation & hyperparameter tuning                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         API DEVELOPMENT                                     │
│         • 43 input features for real-time prediction                        │
│         • FastAPI REST endpoints                                            │
│         • Request validation with Pydantic                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DEPLOYMENT                                        │
│         • Docker containerization                                           │
│         • Google Cloud Run deployment                                       │
│         • Auto-scaling & HTTPS                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check, API info |
| `GET` | `/health` | Service health status |
| `POST` | `/predict` | Credit risk prediction |
| `GET` | `/docs` | Swagger UI documentation |

### Prediction Request

**Endpoint:** `POST /predict`

**Request Body:**
```json
{
  "CODE_GENDER": "M",
  "FLAG_OWN_CAR": "Y",
  "FLAG_OWN_REALTY": "Y",
  "CNT_CHILDREN": 0,
  "AMT_INCOME_TOTAL": 135000.0,
  "AMT_CREDIT": 568800.0,
  "AMT_ANNUITY": 20560.0,
  "AMT_GOODS_PRICE": 450000.0,
  "NAME_INCOME_TYPE": "Working",
  "NAME_EDUCATION_TYPE": "Higher education",
  "NAME_FAMILY_STATUS": "Married",
  "NAME_HOUSING_TYPE": "House / apartment",
  "DAYS_BIRTH": -12005,
  "DAYS_EMPLOYED": -1456,
  "EXT_SOURCE_1": 0.5,
  "EXT_SOURCE_2": 0.6,
  "EXT_SOURCE_3": 0.7
}
```

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.12,
  "risk_level": "LOW",
  "message": "Loan application is likely to be repaid"
}
```

### Risk Levels

| Probability | Risk Level | Recommendation |
|-------------|------------|----------------|
| 0.00 - 0.30 | LOW | Approve |
| 0.30 - 0.50 | MEDIUM | Review required |
| 0.50 - 1.00 | HIGH | Decline or additional verification |

### Example Usage

**cURL:**
```bash
curl -X POST "https://your-api-url.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{"CODE_GENDER": "M", "AMT_INCOME_TOTAL": 135000, ...}'
```

**Python:**
```python
import requests

response = requests.post(
    "https://your-api-url.run.app/predict",
    json={
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "Y",
        "AMT_INCOME_TOTAL": 135000.0,
        "AMT_CREDIT": 568800.0,
        # ... other features
    }
)
print(response.json())
```

---

## Model Performance

### Ensemble Results (VotingClassifier)

| Metric | Score |
|--------|-------|
| **Kaggle Score** | 77.20% |
| **ROC AUC** | 0.785 |
| **Precision (Default)** | 47.00% |
| **Recall (Default)** | 11.00% |
| **F1-Score (Default)** | 18.00% |
| **PR AUC** | 0.28 |

### Model Comparison

| Model | ROC AUC | Training Time |
|-------|---------|---------------|
| LightGBM | 0.780 | Fast |
| XGBoost | 0.778 | Medium |
| **Ensemble** | **0.785** | Combined |

### Confusion Matrix Analysis

- **Strengths:** High accuracy on majority class, good class separation
- **Limitations:** Lower recall for minority class (defaulters)
- **Trade-off:** Model optimized for ROC AUC, balancing precision and recall

---

## Feature Engineering

### Dataset Integration

| Dataset | Features Created | Description |
|---------|-----------------|-------------|
| `application_train.csv` | 43 raw + derived | Core applicant data |
| `bureau.csv` | ~100 | External credit history |
| `previous_application.csv` | ~150 | Historical loan applications |
| `installments_payments.csv` | ~80 | Payment behavior |
| `credit_card_balance.csv` | ~70 | Credit card usage |
| `POS_CASH_balance.csv` | ~60 | Cash loan behavior |
| **Total** | **580 features** | After aggregation |

### Top 10 Most Important Features

| Rank | Feature | Description | Importance |
|------|---------|-------------|------------|
| 1 | `CREDIT_DIV_ANNUITY` | Credit-to-annuity ratio | 112.50 |
| 2 | `EXT_SOURCE_MEAN` | Mean of external scores | 97.55 |
| 3 | `DAYS_BIRTH` | Applicant age | 89.50 |
| 4 | `PREV_CNT_PAYMENT_MEAN` | Avg previous payments | 70.50 |
| 5 | `DAYS_PAYMENT_RATIO_MAX` | Payment timing ratio | 65.01 |
| 6 | `EXT_SOURCE_1_BIRTH_RATIO` | External score / age | 59.50 |
| 7 | `EXT_SOURCE_3_BIRTH_RATIO` | External score / age | 49.00 |
| 8 | `EXT_SOURCE_2` | External credit score | 48.51 |
| 9 | `DAYS_BEFORE_DUE_SUM` | Payment timing sum | 45.50 |
| 10 | `DAYS_REGISTRATION` | Registration duration | 44.50 |

### Feature Engineering Insights

- **External credit scores** (`EXT_SOURCE_*`) are the strongest predictors
- **Age-related features** consistently important across models
- **Engineered ratios** outperform raw features
- **Payment behavior** signals from historical data are highly predictive

---

## Key Findings

### Who is More Likely to Default?

📈 **Higher Risk Indicators:**
- Lower external credit scores
- Higher credit-to-income ratio
- Shorter employment history
- More previous loan applications
- Irregular payment patterns

📉 **Lower Risk Indicators:**
- Higher external credit scores
- Stable employment (longer `DAYS_EMPLOYED`)
- Lower debt-to-income ratio
- Consistent payment history
- Home ownership

---

## Installation

### Prerequisites
- Python 3.11+
- Docker (optional, for containerized deployment)

### Local Setup

1. **Clone the repository:**
```bash
git clone https://github.com/steellsas/credit-risk-prediction-api.git
cd credit-risk-prediction-api
```

2. **Create virtual environment (using uv - recommended):**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate venv
uv venv
source .venv/bin/activate  # Unix/macOS
# or
.venv\Scripts\activate     # Windows
```

3. **Install dependencies:**
```bash
uv pip install -r requirements.txt
```

4. **Run the API locally:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

5. **Access the API:**
- Swagger UI: http://localhost:8000/docs
- API: http://localhost:8000

### Docker Setup

```bash
# Build image
docker build -t credit-risk-api .

# Run container
docker run -p 8000:8000 credit-risk-api
```

---

## Deployment

### Google Cloud Run Deployment

1. **Authenticate with GCP:**
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

2. **Build and push to Container Registry:**
```bash
# Enable required APIs
gcloud services enable containerregistry.googleapis.com
gcloud services enable run.googleapis.com

# Build and push
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/credit-risk-api
```

3. **Deploy to Cloud Run:**
```bash
gcloud run deploy credit-risk-api \
  --image gcr.io/YOUR_PROJECT_ID/credit-risk-api \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

4. **Get the service URL:**
```bash
gcloud run services describe credit-risk-api --region europe-west1 --format="value(status.url)"
```

---

## Project Structure

```
credit-risk-prediction-api/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic schemas
│   ├── prediction.py        # Prediction logic
│   └── preprocessing.py     # Feature preprocessing
├── data/
│   ├── raw/                 # Original datasets
│   └── processed/           # Processed features
├── notebooks/
│   ├── application_eda.ipynb
│   ├── bureau_eda.ipynb
│   ├── previous_application_eda.ipynb
│   ├── feature_engineering.ipynb
│   └── modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── features.py          # Feature engineering functions
│   └── utils.py             # Utility functions
├── models/                  # Saved model files
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── .dockerignore
├── .gitignore
└── README.md
```

---

## Notebooks Overview

| Notebook | Description |
|----------|-------------|
| `application_eda.ipynb` | Core applicant data analysis, target distribution, missing values |
| `bureau_eda.ipynb` | External credit history patterns, bureau features |
| `previous_application_eda.ipynb` | Historical loan application analysis |
| `feature_engineering.ipynb` | Feature creation, aggregation, selection |
| `modeling.ipynb` | Model training, tuning, ensemble creation |

---

## Future Improvements

- [ ] **Model Monitoring** - Add prediction logging and drift detection
- [ ] **A/B Testing** - Compare model versions in production
- [ ] **Explainability** - SHAP values for individual predictions
- [ ] **Threshold Optimization** - Business-driven cutoff tuning
- [ ] **CI/CD Pipeline** - Automated testing and deployment
- [ ] **Model Retraining** - Scheduled retraining pipeline

---

## License

This project is for educational and portfolio purposes, based on the Kaggle Home Credit Default Risk competition.

---

## Author

**Andrius**  
[GitHub](https://github.com/steellsas)

---

*If you found this project helpful, please consider giving it a ⭐!*