# 🏦 Credit Risk Prediction API

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.5-orange.svg)](https://lightgbm.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-red.svg)](https://xgboost.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![GCP Cloud Run](https://img.shields.io/badge/GCP-Cloud%20Run-yellow.svg)](https://cloud.google.com/run)
[![ROC AUC](https://img.shields.io/badge/ROC%20AUC-0.785-success.svg)]()

End-to-end machine learning pipeline for predicting credit default risk, from exploratory data analysis to production deployment.

## 🌐 Live Demo

| Resource | URL |
|----------|-----|
| **🔗 Live API** | [credit-risk-api-753367264972.europe-west1.run.app](https://credit-risk-api-753367264972.europe-west1.run.app) |
| **📚 Swagger UI** | [/docs](https://credit-risk-api-753367264972.europe-west1.run.app/docs) |
| **❤️ Health Check** | [/health](https://credit-risk-api-753367264972.europe-west1.run.app/health) |
| **ℹ️ Model Info** | [/model-info](https://credit-risk-api-753367264972.europe-west1.run.app/model-info) |

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [API Documentation](#-api-documentation)
- [Model Performance](#-model-performance)
- [Feature Engineering](#-feature-engineering)
- [Key Findings](#-key-findings)
- [Installation](#-installation)
- [Deployment](#-deployment)
- [Project Structure](#-project-structure)

---

## 🎯 Overview

This project predicts whether a loan applicant will default on their credit, using data from the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) Kaggle competition. The solution includes:

- **Comprehensive EDA** across 8 interconnected datasets
- **Advanced feature engineering** with 580+ engineered features
- **Ensemble model** combining LightGBM and XGBoost
- **Production-ready API** with FastAPI and Docker
- **Cloud deployment** on Google Cloud Run

---

## 🛠️ Tech Stack

### Machine Learning
| Tool | Purpose |
|------|---------|
| **LightGBM** | Gradient boosting (primary model) |
| **XGBoost** | Gradient boosting (ensemble member) |
| **scikit-learn** | Preprocessing, metrics, voting classifier |
| **Optuna** | Hyperparameter optimization |
| **Polars** | Fast data processing |

### API & Deployment
| Tool | Purpose |
|------|---------|
| **FastAPI** | REST API framework |
| **Uvicorn** | ASGI server |
| **Docker** | Containerization |
| **GCP Cloud Run** | Serverless deployment |
| **GCR** | Container registry |

### Data Analysis
| Tool | Purpose |
|------|---------|
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computing |
| **Matplotlib/Seaborn** | Visualization |
| **SciPy** | Statistical analysis |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                 │
│  │ application  │   │   bureau     │   │   previous   │                 │
│  │   _train     │   │   + bureau   │   │ _application │                 │
│  │   (307K)     │   │   _balance   │   │    (1.6M)    │                 │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                 │
│         │                  │                  │                          │
│         │     ┌────────────┴────────────┐     │                          │
│         │     │                         │     │                          │
│  ┌──────┴─────┴─────┐           ┌───────┴─────┴──────┐                  │
│  │  POS_CASH_balance│           │ installments       │                  │
│  │      (10M)       │           │  _payments (13M)   │                  │
│  └──────────────────┘           └────────────────────┘                  │
│         │                                │                               │
│         └────────────┬───────────────────┘                              │
│                      ▼                                                   │
│         ┌────────────────────────┐                                      │
│         │   FEATURE ENGINEERING  │                                      │
│         │   580+ Features        │                                      │
│         └───────────┬────────────┘                                      │
│                     ▼                                                    │
│         ┌────────────────────────┐                                      │
│         │   MODEL TRAINING       │                                      │
│         │   VotingClassifier     │                                      │
│         │   (LightGBM + XGBoost) │                                      │
│         └───────────┬────────────┘                                      │
│                     ▼                                                    │
│         ┌────────────────────────┐                                      │
│         │   FastAPI + Docker     │                                      │
│         └───────────┬────────────┘                                      │
│                     ▼                                                    │
│         ┌────────────────────────┐                                      │
│         │   GCP Cloud Run        │                                      │
│         │   Production API       │                                      │
│         └────────────────────────┘                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📡 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Welcome message |
| `GET` | `/health` | Health check with uptime |
| `GET` | `/model-info` | Model metadata and top features |
| `POST` | `/predict` | Get default probability prediction |

### Prediction Request

```bash
curl -X POST "https://credit-risk-api-753367264972.europe-west1.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "SK_ID_CURR": 100002,
    "NAME_CONTRACT_TYPE": "Cash loans",
    "CODE_GENDER": "M",
    "FLAG_OWN_CAR": "N",
    "FLAG_OWN_REALTY": "Y",
    "CNT_CHILDREN": 0,
    "AMT_INCOME_TOTAL": 270000.0,
    "AMT_CREDIT": 1293502.5,
    "AMT_ANNUITY": 35698.5,
    "AMT_GOODS_PRICE": 1129500.0,
    "REGION_POPULATION_RELATIVE": 0.035792,
    "DAYS_BIRTH": -12005,
    "DAYS_EMPLOYED": -4542,
    "DAYS_REGISTRATION": -3393.0,
    "DAYS_ID_PUBLISH": -2531,
    "FLAG_MOBIL": 1,
    "FLAG_EMP_PHONE": 1,
    "FLAG_WORK_PHONE": 0,
    "FLAG_CONT_MOBILE": 1,
    "FLAG_PHONE": 1,
    "FLAG_EMAIL": 0,
    "CNT_FAM_MEMBERS": 1.0,
    "REGION_RATING_CLIENT": 2,
    "REGION_RATING_CLIENT_W_CITY": 2,
    "HOUR_APPR_PROCESS_START": 9,
    "REG_REGION_NOT_LIVE_REGION": 0,
    "REG_REGION_NOT_WORK_REGION": 0,
    "LIVE_REGION_NOT_WORK_REGION": 0,
    "REG_CITY_NOT_LIVE_CITY": 0,
    "REG_CITY_NOT_WORK_CITY": 0,
    "LIVE_CITY_NOT_WORK_CITY": 0,
    "EXT_SOURCE_1": 0.5,
    "EXT_SOURCE_2": 0.6,
    "EXT_SOURCE_3": 0.5,
    "APARTMENTS_AVG": 0.5,
    "BASEMENTAREA_AVG": 0.5,
    "YEARS_BEGINEXPLUATATION_AVG": 0.98,
    "YEARS_BUILD_AVG": 0.62,
    "COMMONAREA_AVG": 0.04,
    "ELEVATORS_AVG": 0.08,
    "ENTRANCES_AVG": 0.14,
    "FLOORSMAX_AVG": 0.17,
    "FLOORSMIN_AVG": 0.21,
    "LANDAREA_AVG": 0.07,
    "OBS_30_CNT_SOCIAL_CIRCLE": 2.0,
    "DEF_30_CNT_SOCIAL_CIRCLE": 0.0
  }'
```

### Prediction Response

```json
{
  "prediction": 0,
  "probability": 0.1234,
  "risk_level": "LOW",
  "risk_color": "🟢",
  "message": "Low risk - Applicant shows strong repayment indicators"
}
```

### Risk Levels

| Level | Probability | Color | Description |
|-------|-------------|-------|-------------|
| **LOW** | 0.00 - 0.30 | 🟢 | Strong repayment indicators |
| **MEDIUM** | 0.30 - 0.50 | 🟡 | Moderate risk, review recommended |
| **HIGH** | 0.50 - 1.00 | 🔴 | High default risk |

### Python Example

```python
import requests

url = "https://credit-risk-api-753367264972.europe-west1.run.app/predict"

applicant = {
    "SK_ID_CURR": 100002,
    "NAME_CONTRACT_TYPE": "Cash loans",
    "CODE_GENDER": "M",
    "AMT_INCOME_TOTAL": 270000.0,
    "AMT_CREDIT": 1293502.5,
    "EXT_SOURCE_1": 0.5,
    "EXT_SOURCE_2": 0.6,
    "EXT_SOURCE_3": 0.5,
    # ... other features
}

response = requests.post(url, json=applicant)
result = response.json()

print(f"Default Probability: {result['probability']:.2%}")
print(f"Risk Level: {result['risk_level']} {result['risk_color']}")
```

---

## 📊 Model Performance

### Ensemble Model (VotingClassifier)

| Metric | Score |
|--------|-------|
| **ROC AUC** | 0.785 |
| **Kaggle Score** | 77.2% |
| **Precision (Class 1)** | 47% |
| **Recall (Class 1)** | 11% |

### Model Comparison

| Model | ROC AUC | Training Time |
|-------|---------|---------------|
| **LightGBM** | 0.782 | ~2 min |
| **XGBoost** | 0.779 | ~5 min |
| **VotingClassifier** | **0.785** | ~7 min |
| Logistic Regression | 0.742 | ~30 sec |
| Random Forest | 0.731 | ~10 min |

### Confusion Matrix Analysis

```
                 Predicted
              |  No  |  Yes |
    Actual ---|------|------|
       No     | 47K  | 1.5K |
       Yes    | 3.5K | 450  |
```

- **True Negatives:** 47,000 (correctly identified non-defaulters)
- **False Positives:** 1,500 (non-defaulters flagged as risky)
- **False Negatives:** 3,500 (defaulters missed)
- **True Positives:** 450 (correctly identified defaulters)

---

## 🔧 Feature Engineering

### Dataset Integration

| Dataset | Records | Features Created |
|---------|---------|------------------|
| application_train | 307,511 | Base features |
| bureau | 1,716,428 | Credit history aggregations |
| bureau_balance | 27,299,925 | Monthly balance patterns |
| previous_application | 1,670,214 | Previous loan features |
| POS_CASH_balance | 10,001,358 | POS loan aggregations |
| installments_payments | 13,605,401 | Payment behavior features |
| credit_card_balance | 3,840,312 | Credit card patterns |

### Top 10 Most Important Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | EXT_SOURCE_2 | 0.142 |
| 2 | EXT_SOURCE_3 | 0.128 |
| 3 | EXT_SOURCE_1 | 0.095 |
| 4 | DAYS_BIRTH | 0.067 |
| 5 | DAYS_EMPLOYED | 0.054 |
| 6 | AMT_CREDIT | 0.043 |
| 7 | AMT_ANNUITY | 0.038 |
| 8 | AMT_GOODS_PRICE | 0.035 |
| 9 | DAYS_ID_PUBLISH | 0.032 |
| 10 | DAYS_REGISTRATION | 0.028 |

---

## 🔍 Key Findings

### Who Defaults More?

Based on exploratory data analysis:

1. **Lower External Scores**
   - Applicants with EXT_SOURCE scores below 0.3 have 3x higher default rate
   - External sources are the strongest predictors

2. **Higher Credit-to-Income Ratio**
   - Credit amount > 8x annual income increases default risk by 45%
   - Debt burden is a critical factor

3. **Shorter Employment History**
   - Less than 1 year employed: 12% default rate
   - More than 5 years employed: 6% default rate

4. **Younger Applicants**
   - Age 20-30: 10% default rate
   - Age 50+: 5% default rate

5. **Previous Loan Issues**
   - Applicants with previous application rejections: 15% default rate
   - Clean history applicants: 7% default rate

---

## 💻 Installation

### Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)
- UV package manager (recommended) or pip

### Local Setup with UV

```bash
# Clone repository
git clone https://github.com/steellsas/credit-risk-prediction-api.git
cd credit-risk-prediction-api

# Create virtual environment
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements-prod.txt

# Run API
uvicorn app.main:app --reload --port 8000
```

### Local Setup with pip

```bash
# Clone repository
git clone https://github.com/steellsas/credit-risk-prediction-api.git
cd credit-risk-prediction-api

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements-prod.txt

# Run API
uvicorn app.main:app --reload --port 8000
```

### Docker Setup

```bash
# Build image
docker build -t credit-risk-api .

# Run container
docker run -p 8080:8080 credit-risk-api

# Access API
open http://localhost:8080/docs
```

---

## 🚀 Deployment

### GCP Cloud Run (Current Production)

The API is deployed on Google Cloud Run with the following configuration:

| Parameter | Value |
|-----------|-------|
| Region | europe-west1 (Belgium) |
| Memory | 2 GB |
| CPU | 2 vCPU |
| Min Instances | 0 (scale to zero) |
| Max Instances | 10 |
| Timeout | 60 seconds |

### Deploy Your Own Instance

```bash
# 1. Configure GCP
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Enable APIs
gcloud services enable containerregistry.googleapis.com run.googleapis.com cloudbuild.googleapis.com

# 3. Build and push
docker build -t credit-risk-api .
docker tag credit-risk-api gcr.io/YOUR_PROJECT_ID/credit-risk-api
docker push gcr.io/YOUR_PROJECT_ID/credit-risk-api

# 4. Deploy
gcloud run deploy credit-risk-api \
  --image gcr.io/YOUR_PROJECT_ID/credit-risk-api \
  --platform managed \
  --region europe-west1 \
  --memory 2Gi \
  --allow-unauthenticated
```

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

---

## 📁 Project Structure

```
credit-risk-prediction-api/
├── app/
│   ├── __init__.py
│   └── main.py              # FastAPI application
├── data/
│   ├── cache/               # Aggregated feature cache
│   ├── encoders/            # Label encoders
│   └── models/              # Trained model files
├── notebooks/
│   └── credit_risk_eda.ipynb  # Exploratory analysis
├── src/
│   └── utils/
│       └── app_utils.py     # Utility functions
├── .dockerignore
├── .gitignore
├── DEPLOYMENT.md            # GCP deployment guide
├── Dockerfile
├── Makefile
├── README.md
├── requirements-dev.txt     # Development dependencies
├── requirements-prod.txt    # Production dependencies
└── pyproject.toml
```

---

## 📄 License

This project is for educational and portfolio purposes.

---

## 👤 Author

**Andrius**

- GitHub: [@steellsas](https://github.com/steellsas)

---

## 🙏 Acknowledgments

- [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) Kaggle competition
- [FastAPI](https://fastapi.tiangolo.com/) documentation
- [LightGBM](https://lightgbm.readthedocs.io/) and [XGBoost](https://xgboost.readthedocs.io/) teams
