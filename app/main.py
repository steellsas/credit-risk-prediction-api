
import joblib
import pickle
import logging

from fastapi import FastAPI, Request,  HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel
from typing import Optional

import polars as pl

from src.utils.app_utils import (
    clean_applicans_data,
    create_application_df,
    replace_infinite_with_nan,
    prepare_data
)



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app/logs/app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def enrich_with_aggregates(input_df: pl.DataFrame) -> pl.DataFrame:
    """
    Enriches the input applicant DataFrame with aggregated historical features from various sources.

    Parameters:
        input_df (pl.DataFrame): The applicant data as a polars DataFrame.

    Returns:
        pl.DataFrame: The enriched DataFrame with additional aggregate features.
    """
    sk_id = input_df["SK_ID_CURR"][0]

    enriched = input_df.join(bureau_balance_agg, on="SK_ID_CURR", how="left") \
                       .join(previous_application_agg, on="SK_ID_CURR", how="left") \
                       .join(installmens_agg, on="SK_ID_CURR", how="left") \
                       .join(pos_cash_agg_df, on="SK_ID_CURR", how="left") \
                       .join(creadit_card_df, on="SK_ID_CURR", how="left")

    return enriched


app = FastAPI()

model = joblib.load("data/models/voting_model_2.pkl")
encoder_map = joblib.load("data/encoders/appl_encoder_map.pkl")
with open('data/models/voting_features.txt', 'rb') as fp:
    model_features = pickle.load(fp)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


class InputData(BaseModel):

    SK_ID_CURR: int
    NAME_CONTRACT_TYPE: str
    NAME_EDUCATION_TYPE: str
    ORGANIZATION_TYPE:str
    OCCUPATION_TYPE: str
    NAME_INCOME_TYPE: str
    AMT_CREDIT: float
    FLOORSMAX_MODE: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    NAME_FAMILY_STATUS: str
    EXT_SOURCE_1: float
    AMT_GOODS_PRICE: float
    FLOORSMIN_MODE: float
    ENTRANCES_MODE: float
    YEARS_BUILD_AVG: float
    LIVINGAREA_AVG: float
    DEF_30_CNT_SOCIAL_CIRCLE: float
    FLAG_DOCUMENT_18: int
    DAYS_LAST_PHONE_CHANGE: float
    FLOORSMAX_MEDI: float
    APARTMENTS_MEDI: float
    AMT_REQ_CREDIT_BUREAU_YEAR: float
    APARTMENTS_AVG: float
    TOTALAREA_MODE: float
    REG_CITY_NOT_LIVE_CITY: int
    REGION_RATING_CLIENT_W_CITY: int
    AMT_REQ_CREDIT_BUREAU_MON: float
    APARTMENTS_MODE: float
    FLOORSMAX_AVG: float
    FLAG_DOCUMENT_3: int
    DAYS_ID_PUBLISH: int
    AMT_ANNUITY: float
    DAYS_BIRTH: float
    DAYS_EMPLOYED: int
    LIVINGAREA_MEDI: float
    REGION_RATING_CLIENT: int
    FLOORSMIN_MEDI: float
    DAYS_REGISTRATION: float
    YEARS_BUILD_MEDI: float
    YEARS_BUILD_MODE: float
    DEF_60_CNT_SOCIAL_CIRCLE: float
    CODE_GENDER: str
    OWN_CAR_AGE: Optional[float]
    AMT_INCOME_TOTAL: float
    CNT_FAM_MEMBERS: float


@app.on_event("startup")
def load_aggregated_data():
    """
    Loads all required aggregated historical data into global variables at API startup.

    Raises:
        RuntimeError: If any required data file is missing or corrupt.
    """
    logger.info("Loading aggregated data...")
    try:
        global bureau_balance_agg, previous_application_agg, installmens_agg, pos_cash_agg_df, creadit_card_df

        bureau_balance_agg = pl.read_parquet("data/cache/bureau_balance_agg.parquet")
        previous_application_agg = pl.read_parquet("data/cache/previous_application_agg.parquet")
        installmens_agg = pl.read_parquet("data/cache/installments_agg.parquet")
        pos_cash_agg_df = pl.read_parquet("data/cache/pos_cash_agg.parquet")
        creadit_card_df = pl.read_parquet("data/cache/credit_card_agg.parquet")
        logger.info("Aggregated data loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load aggregated data: {str(e)}", exc_info=True)
        raise RuntimeError("Startup failed due to missing or corrupt data.")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):

    """
    Serves the main HTML page for the application.

    Parameters:
        request (Request): The incoming HTTP request object.

    Returns:
        TemplateResponse: The rendered index.html template.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(data: InputData):
    """
    Predicts the credit risk outcome for a given applicant using a pre-trained voting model.

    This endpoint receives structured applicant data, performs a series of preprocessing steps—
    including cleaning, encoding, enrichment with aggregated historical data—and returns a binary
    prediction along with its probability score.

    Parameters:
    ----------
    data : InputData
        A Pydantic model containing all required applicant features.

    Returns:
    -------
    dict
        A dictionary containing:
        - "prediction": int (0 or 1) indicating the model's classification.
        - "probability": float representing the confidence score for the positive class.

    Raises:
    -------
    HTTPException
        - 400: If input data is malformed or cannot be parsed.
        - 500: If any step in the preprocessing or prediction pipeline fails, including:
            - Data cleaning
            - Feature encoding
            - Aggregation enrichment
            - Final preprocessing
            - Model inference
    """
    try:
        logger.info(f"Received prediction request for SK_ID_CURR: {data.SK_ID_CURR}")
        try:
            input_df = pl.DataFrame([data])
            logger.debug("Input data converted to polars DataFrame")
        except Exception as e:
            logger.error(f"Failed to convert input data: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail="Invalid input format")
        try:
            cleaned_df = clean_applicans_data(input_df)
            logger.debug("Data cleaned successfully")
        except Exception as e:
            logger.error(f"Data cleaning failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error during data cleaning")
        try:
            selected_features = [
                'NAME_CONTRACT_TYPE', 'NAME_EDUCATION_TYPE', 'ORGANIZATION_TYPE',
                'OCCUPATION_TYPE', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'CODE_GENDER'
            ]
            selected_encoders = {k: encoder_map[k] for k in selected_features if k in encoder_map}
            main_df = create_application_df(cleaned_df, encoder_map=selected_encoders)
            logger.debug("Feature encoding completed")
        except Exception as e:
            logger.error(f"Feature encoding failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error during feature encoding")
        try:
            enriched_df = enrich_with_aggregates(main_df)
            logger.debug("Data enrichment with aggregates successful")
        except Exception as e:
            logger.error(f"Data enrichment failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error enriching data with aggregates")
        try:
            enriched_df = prepare_data(enriched_df)
            final_df = enriched_df[model_features].to_pandas()
            final_df = replace_infinite_with_nan(final_df)
            logger.debug("Final preprocessing completed")
        except Exception as e:
            logger.error(f"Final preprocessing failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error during final preprocessing")
        try:
            prediction = model.predict(final_df)[0]
            prediction_proba = model.predict_proba(final_df)[:, 1][0]
            logger.info(f"Prediction: {prediction}, Probability: {prediction_proba:.4f}")
        except Exception as e:
            logger.error(f"Model prediction failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Model prediction error")

        return {
            "prediction": int(prediction),
            "probability": float(round(prediction_proba, 4))
        }

    except HTTPException as http_err:
        raise http_err  
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unexpected server error")
