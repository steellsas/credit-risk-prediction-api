"""
Credit Risk Prediction API
==========================
Production-ready REST API for credit default risk prediction.

This API uses an ensemble model (LightGBM + XGBoost) trained on the 
Home Credit Default Risk dataset to predict loan default probability.

Author: Andrius
GitHub: https://github.com/steellsas/credit-risk-prediction-api
"""

import joblib
import pickle
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import polars as pl
import pandas as pd

from src.utils.app_utils import (
    clean_applicans_data,
    create_application_df,
    replace_infinite_with_nan,
    prepare_data
)

# ============================================
# Logging Configuration
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================
# API Metadata for Swagger Documentation
# ============================================
API_TITLE = "Credit Risk Prediction API"
API_DESCRIPTION = """
## 🏦 Credit Default Risk Prediction

This API predicts the probability of a loan applicant defaulting on their loan 
using an ensemble machine learning model.

### 🎯 Model Information
- **Algorithm**: VotingClassifier (LightGBM + XGBoost, Soft Voting)
- **Training Data**: Home Credit Default Risk Dataset
- **Features**: 198 selected features from 580 engineered features
- **Performance**: ROC AUC 0.785 | Kaggle Score: 77.2%

### 📊 Risk Levels
| Probability | Risk Level | Recommendation |
|-------------|------------|----------------|
| 0.00 - 0.30 | 🟢 LOW | Likely to approve |
| 0.30 - 0.50 | 🟡 MEDIUM | Manual review recommended |
| 0.50 - 1.00 | 🔴 HIGH | High default risk |

### 🔗 Links
- [GitHub Repository](https://github.com/steellsas/credit-risk-prediction-api)
- [Kaggle Competition](https://www.kaggle.com/c/home-credit-default-risk)
"""

API_TAGS = [
    {
        "name": "Prediction",
        "description": "Credit risk prediction endpoints"
    },
    {
        "name": "Health",
        "description": "API health and status endpoints"
    },
    {
        "name": "Model Info",
        "description": "Model metadata and feature information"
    }
]

# ============================================
# FastAPI Application
# ============================================
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version="2.0.0",
    openapi_tags=API_TAGS,
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Andrius",
        "url": "https://github.com/steellsas"
    },
    license_info={
        "name": "MIT License"
    }
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Global Variables
# ============================================
model = None
encoder_map = None
model_features = None
bureau_balance_agg = None
previous_application_agg = None
installmens_agg = None
pos_cash_agg_df = None
creadit_card_df = None
startup_time = None

# ============================================
# Pydantic Models (Request/Response Schemas)
# ============================================

class ApplicantData(BaseModel):
    """Input schema for loan applicant data."""
    
    SK_ID_CURR: int = Field(..., description="Unique applicant ID", example=100001)
    NAME_CONTRACT_TYPE: str = Field(..., description="Type of loan contract", example="Cash loans")
    NAME_EDUCATION_TYPE: str = Field(..., description="Education level", example="Higher education")
    ORGANIZATION_TYPE: str = Field(..., description="Type of organization where client works", example="Business Entity Type 3")
    OCCUPATION_TYPE: str = Field(..., description="Client's occupation", example="Laborers")
    NAME_INCOME_TYPE: str = Field(..., description="Type of income", example="Working")
    AMT_CREDIT: float = Field(..., description="Credit amount of the loan", example=568800.0)
    FLOORSMAX_MODE: float = Field(..., description="Max floors in building (mode)", example=0.0)
    EXT_SOURCE_2: float = Field(..., description="External data source score 2", example=0.6)
    EXT_SOURCE_3: float = Field(..., description="External data source score 3", example=0.5)
    NAME_FAMILY_STATUS: str = Field(..., description="Family status", example="Married")
    EXT_SOURCE_1: float = Field(..., description="External data source score 1", example=0.5)
    AMT_GOODS_PRICE: float = Field(..., description="Price of goods for which loan is given", example=450000.0)
    FLOORSMIN_MODE: float = Field(..., description="Min floors in building (mode)", example=0.0)
    ENTRANCES_MODE: float = Field(..., description="Number of entrances (mode)", example=0.0)
    YEARS_BUILD_AVG: float = Field(..., description="Year when building was built (avg)", example=0.0)
    LIVINGAREA_AVG: float = Field(..., description="Living area (avg)", example=0.0)
    DEF_30_CNT_SOCIAL_CIRCLE: float = Field(..., description="Defaults in social circle (30 days)", example=0.0)
    FLAG_DOCUMENT_18: int = Field(..., description="Document 18 provided flag", example=0)
    DAYS_LAST_PHONE_CHANGE: float = Field(..., description="Days since last phone change", example=-1000.0)
    FLOORSMAX_MEDI: float = Field(..., description="Max floors in building (median)", example=0.0)
    APARTMENTS_MEDI: float = Field(..., description="Number of apartments (median)", example=0.0)
    AMT_REQ_CREDIT_BUREAU_YEAR: float = Field(..., description="Credit bureau inquiries (year)", example=1.0)
    APARTMENTS_AVG: float = Field(..., description="Number of apartments (avg)", example=0.0)
    TOTALAREA_MODE: float = Field(..., description="Total area (mode)", example=0.0)
    REG_CITY_NOT_LIVE_CITY: int = Field(..., description="Registration city differs from living city", example=0)
    REGION_RATING_CLIENT_W_CITY: int = Field(..., description="Region rating with city", example=2)
    AMT_REQ_CREDIT_BUREAU_MON: float = Field(..., description="Credit bureau inquiries (month)", example=0.0)
    APARTMENTS_MODE: float = Field(..., description="Number of apartments (mode)", example=0.0)
    FLOORSMAX_AVG: float = Field(..., description="Max floors in building (avg)", example=0.0)
    FLAG_DOCUMENT_3: int = Field(..., description="Document 3 provided flag", example=1)
    DAYS_ID_PUBLISH: int = Field(..., description="Days since ID was published", example=-3000)
    AMT_ANNUITY: float = Field(..., description="Loan annuity amount", example=25000.0)
    DAYS_BIRTH: float = Field(..., description="Client's age in days (negative)", example=-12000)
    DAYS_EMPLOYED: int = Field(..., description="Days employed (negative)", example=-2000)
    LIVINGAREA_MEDI: float = Field(..., description="Living area (median)", example=0.0)
    REGION_RATING_CLIENT: int = Field(..., description="Region rating", example=2)
    FLOORSMIN_MEDI: float = Field(..., description="Min floors in building (median)", example=0.0)
    DAYS_REGISTRATION: float = Field(..., description="Days since registration", example=-5000.0)
    YEARS_BUILD_MEDI: float = Field(..., description="Year when building was built (median)", example=0.0)
    YEARS_BUILD_MODE: float = Field(..., description="Year when building was built (mode)", example=0.0)
    DEF_60_CNT_SOCIAL_CIRCLE: float = Field(..., description="Defaults in social circle (60 days)", example=0.0)
    CODE_GENDER: str = Field(..., description="Gender (M/F)", example="M")
    OWN_CAR_AGE: Optional[float] = Field(None, description="Age of car (if owned)", example=5.0)
    AMT_INCOME_TOTAL: float = Field(..., description="Total income", example=150000.0)
    CNT_FAM_MEMBERS: float = Field(..., description="Number of family members", example=2.0)

    class Config:
        json_schema_extra = {
            "example": {
                "SK_ID_CURR": 100001,
                "NAME_CONTRACT_TYPE": "Cash loans",
                "NAME_EDUCATION_TYPE": "Higher education",
                "ORGANIZATION_TYPE": "Business Entity Type 3",
                "OCCUPATION_TYPE": "Laborers",
                "NAME_INCOME_TYPE": "Working",
                "AMT_CREDIT": 568800.0,
                "FLOORSMAX_MODE": 0.0,
                "EXT_SOURCE_2": 0.6,
                "EXT_SOURCE_3": 0.5,
                "NAME_FAMILY_STATUS": "Married",
                "EXT_SOURCE_1": 0.5,
                "AMT_GOODS_PRICE": 450000.0,
                "FLOORSMIN_MODE": 0.0,
                "ENTRANCES_MODE": 0.0,
                "YEARS_BUILD_AVG": 0.0,
                "LIVINGAREA_AVG": 0.0,
                "DEF_30_CNT_SOCIAL_CIRCLE": 0.0,
                "FLAG_DOCUMENT_18": 0,
                "DAYS_LAST_PHONE_CHANGE": -1000.0,
                "FLOORSMAX_MEDI": 0.0,
                "APARTMENTS_MEDI": 0.0,
                "AMT_REQ_CREDIT_BUREAU_YEAR": 1.0,
                "APARTMENTS_AVG": 0.0,
                "TOTALAREA_MODE": 0.0,
                "REG_CITY_NOT_LIVE_CITY": 0,
                "REGION_RATING_CLIENT_W_CITY": 2,
                "AMT_REQ_CREDIT_BUREAU_MON": 0.0,
                "APARTMENTS_MODE": 0.0,
                "FLOORSMAX_AVG": 0.0,
                "FLAG_DOCUMENT_3": 1,
                "DAYS_ID_PUBLISH": -3000,
                "AMT_ANNUITY": 25000.0,
                "DAYS_BIRTH": -12000,
                "DAYS_EMPLOYED": -2000,
                "LIVINGAREA_MEDI": 0.0,
                "REGION_RATING_CLIENT": 2,
                "FLOORSMIN_MEDI": 0.0,
                "DAYS_REGISTRATION": -5000.0,
                "YEARS_BUILD_MEDI": 0.0,
                "YEARS_BUILD_MODE": 0.0,
                "DEF_60_CNT_SOCIAL_CIRCLE": 0.0,
                "CODE_GENDER": "M",
                "OWN_CAR_AGE": 5.0,
                "AMT_INCOME_TOTAL": 150000.0,
                "CNT_FAM_MEMBERS": 2.0
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    
    prediction: int = Field(..., description="Binary prediction (0=No Default, 1=Default)")
    probability: float = Field(..., description="Probability of default (0.0 - 1.0)")
    risk_level: str = Field(..., description="Risk level category (LOW/MEDIUM/HIGH)")
    risk_color: str = Field(..., description="Risk level color indicator")
    message: str = Field(..., description="Human-readable interpretation")
    applicant_id: int = Field(..., description="Applicant ID from request")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 0,
                "probability": 0.15,
                "risk_level": "LOW",
                "risk_color": "🟢",
                "message": "Low default risk - Loan likely to be repaid",
                "applicant_id": 100001
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    data_loaded: bool = Field(..., description="Whether aggregate data is loaded")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    timestamp: str = Field(..., description="Current timestamp")


class ModelInfoResponse(BaseModel):
    """Response schema for model information endpoint."""
    
    model_name: str
    model_type: str
    algorithm: str
    n_features: int
    performance: dict
    training_data: str
    top_features: list


# ============================================
# Helper Functions
# ============================================

def get_risk_level(probability: float) -> tuple:
    """
    Determine risk level based on default probability.
    
    Returns:
        tuple: (risk_level, risk_color, message)
    """
    if probability < 0.30:
        return ("LOW", "🟢", "Low default risk - Loan likely to be repaid")
    elif probability < 0.50:
        return ("MEDIUM", "🟡", "Medium default risk - Manual review recommended")
    else:
        return ("HIGH", "🔴", "High default risk - Elevated probability of default")


def enrich_with_aggregates(input_df: pl.DataFrame) -> pl.DataFrame:
    """
    Enriches the input applicant DataFrame with aggregated historical features.
    """
    enriched = input_df.join(bureau_balance_agg, on="SK_ID_CURR", how="left") \
                       .join(previous_application_agg, on="SK_ID_CURR", how="left") \
                       .join(installmens_agg, on="SK_ID_CURR", how="left") \
                       .join(pos_cash_agg_df, on="SK_ID_CURR", how="left") \
                       .join(creadit_card_df, on="SK_ID_CURR", how="left")
    return enriched


# ============================================
# Startup Event
# ============================================

@app.on_event("startup")
async def load_model_and_data():
    """Load model and aggregated data at startup."""
    global model, encoder_map, model_features, startup_time
    global bureau_balance_agg, previous_application_agg, installmens_agg
    global pos_cash_agg_df, creadit_card_df
    
    startup_time = datetime.now()
    logger.info("🚀 Starting Credit Risk Prediction API...")
    
    try:
        # Load model
        logger.info("Loading ML model...")
        model = joblib.load("data/models/voting_model_2.pkl")
        logger.info("✅ Model loaded successfully")
        
        # Load encoders
        logger.info("Loading encoders...")
        encoder_map = joblib.load("data/encoders/appl_encoder_map.pkl")
        logger.info("✅ Encoders loaded successfully")
        
        # Load feature list
        logger.info("Loading feature list...")
        with open('data/models/voting_features.txt', 'rb') as fp:
            model_features = pickle.load(fp)
        logger.info(f"✅ Feature list loaded ({len(model_features)} features)")
        
        # Load aggregated data
        logger.info("Loading aggregated data...")
        bureau_balance_agg = pl.read_parquet("data/cache/bureau_balance_agg.parquet")
        previous_application_agg = pl.read_parquet("data/cache/previous_application_agg.parquet")
        installmens_agg = pl.read_parquet("data/cache/installments_agg.parquet")
        pos_cash_agg_df = pl.read_parquet("data/cache/pos_cash_agg.parquet")
        creadit_card_df = pl.read_parquet("data/cache/credit_card_agg.parquet")
        logger.info("✅ Aggregated data loaded successfully")
        
        logger.info("🎉 API startup complete!")
        
    except Exception as e:
        logger.critical(f"❌ Startup failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Startup failed: {str(e)}")


# ============================================
# API Endpoints
# ============================================

@app.get("/", tags=["Health"])
async def root():
    """
    Root endpoint - API welcome message.
    """
    return {
        "message": "Welcome to Credit Risk Prediction API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring and container orchestration.
    
    Returns the current status of the API including model and data loading status.
    """
    uptime = (datetime.now() - startup_time).total_seconds() if startup_time else 0
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        data_loaded=bureau_balance_agg is not None,
        uptime_seconds=round(uptime, 2),
        timestamp=datetime.now().isoformat()
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model Info"])
async def get_model_info():
    """
    Get information about the deployed model.
    
    Returns model metadata, performance metrics, and top features.
    """
    return ModelInfoResponse(
        model_name="Credit Default Risk Predictor",
        model_type="VotingClassifier (Soft Voting)",
        algorithm="LightGBM + XGBoost Ensemble",
        n_features=len(model_features) if model_features else 0,
        performance={
            "roc_auc": 0.785,
            "kaggle_score": "77.2%",
            "precision_class1": 0.47,
            "recall_class1": 0.11,
            "f1_class1": 0.18
        },
        training_data="Home Credit Default Risk (Kaggle)",
        top_features=[
            "CREDIT_DIV_ANNUITY",
            "EXT_SOURCE_MEAN",
            "DAYS_BIRTH",
            "PREV_CNT_PAYMENT_MEAN",
            "DAYS_PAYMENT_RATIO_MAX",
            "EXT_SOURCE_1_BIRTH_RATIO",
            "EXT_SOURCE_3_BIRTH_RATIO",
            "EXT_SOURCE_2",
            "DAYS_BEFORE_DUE_SUM",
            "DAYS_REGISTRATION"
        ]
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(data: ApplicantData):
    """
    Predict credit default risk for a loan applicant.
    
    This endpoint receives applicant data and returns:
    - Binary prediction (0 = No Default, 1 = Default)
    - Probability of default
    - Risk level classification (LOW/MEDIUM/HIGH)
    - Human-readable interpretation
    
    **Model**: VotingClassifier (LightGBM + XGBoost)  
    **Performance**: ROC AUC 0.785
    """
    try:
        logger.info(f"📥 Prediction request for SK_ID_CURR: {data.SK_ID_CURR}")
        
        # Convert to DataFrame
        try:
            input_df = pl.DataFrame([data.model_dump()])
        except Exception as e:
            logger.error(f"Failed to convert input data: {e}")
            raise HTTPException(status_code=400, detail="Invalid input format")
        
        # Clean data
        try:
            cleaned_df = clean_applicans_data(input_df)
        except Exception as e:
            logger.error(f"Data cleaning failed: {e}")
            raise HTTPException(status_code=500, detail="Error during data cleaning")
        
        # Encode features
        try:
            selected_features = [
                'NAME_CONTRACT_TYPE', 'NAME_EDUCATION_TYPE', 'ORGANIZATION_TYPE',
                'OCCUPATION_TYPE', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'CODE_GENDER'
            ]
            selected_encoders = {k: encoder_map[k] for k in selected_features if k in encoder_map}
            main_df = create_application_df(cleaned_df, encoder_map=selected_encoders)
        except Exception as e:
            logger.error(f"Feature encoding failed: {e}")
            raise HTTPException(status_code=500, detail="Error during feature encoding")
        
        # Enrich with aggregates
        try:
            enriched_df = enrich_with_aggregates(main_df)
        except Exception as e:
            logger.error(f"Data enrichment failed: {e}")
            raise HTTPException(status_code=500, detail="Error enriching data")
        
        # Final preprocessing
        try:
            enriched_df = prepare_data(enriched_df)
            final_df = enriched_df[model_features].to_pandas()
            final_df = replace_infinite_with_nan(final_df)
        except Exception as e:
            logger.error(f"Final preprocessing failed: {e}")
            raise HTTPException(status_code=500, detail="Error during preprocessing")
        
        # Make prediction
        try:
            prediction = int(model.predict(final_df)[0])
            probability = float(model.predict_proba(final_df)[:, 1][0])
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            raise HTTPException(status_code=500, detail="Model prediction error")
        
        # Determine risk level
        risk_level, risk_color, message = get_risk_level(probability)
        
        logger.info(f"✅ Prediction complete: {prediction} (prob: {probability:.4f}, risk: {risk_level})")
        
        return PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            risk_level=risk_level,
            risk_color=risk_color,
            message=message,
            applicant_id=data.SK_ID_CURR
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unexpected server error")
