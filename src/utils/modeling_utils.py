import polars as pl
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split


from sklearn.metrics import (
    roc_auc_score,
    f1_score, 
    precision_score,
    recall_score,
    classification_report,
)




from src.utils.eda_utils import (
    reduce_memory_usage_pl,
)

def get_feature_importance(pipeline):
    """
    Extracts feature importance or coefficients from a fitted pipeline.

    Args:
    - pipeline: The scikit-learn pipeline object

    Returns:
    - importance_df: DataFrame containing feature importance or coefficients
    """
    model = pipeline.named_steps["model"]
    importance_df = pd.DataFrame()

    # LightGBM
    if hasattr(model, "feature_importances_"):
        if hasattr(model, "feature_name_"):
            feature_names = model.feature_name_
        else:
            feature_names = model.get_booster().feature_names

        importance = model.feature_importances_
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        })

    # XGBoost (sklearn API)
    elif hasattr(model, "get_booster"):
        booster = model.get_booster()
        importance = booster.get_score(importance_type="weight")
        importance_df = pd.DataFrame({
            "feature": list(importance.keys()),
            "importance": list(importance.values())
        })

    # Linear models
    elif hasattr(model, "coef_"):
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        coef = model.coef_[0]
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coef
        })
        importance_df["abs_coefficient"] = importance_df["coefficient"].abs()
        importance_df = importance_df.sort_values(
            by="abs_coefficient", ascending=False
        ).drop("abs_coefficient", axis=1)

    else:
        return "Model does not support feature importance or coefficients."

    return importance_df.sort_values("importance", ascending=False)

def evaluate_model(model, model_name, X_train, y_train, X_test, y_test, sample_weight=None):
    """Evaluate a binary classifier with optional sample weights."""

    # Fit model with optional sample weights
    
    if sample_weight is not None:
        model.fit(X_train, y_train, **{"model__sample_weight": sample_weight})
    else:
        model.fit(X_train, y_train)



    # Predict labels and probabilities
    y_pred = model.predict(X_test)
    y_pred_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    # Metrics
    results = {
        "Model": model_name,
        "ROC AUC": roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
        "F1 Macro": f1_score(y_test, y_pred, average='macro'),
        "F1 Weighted": f1_score(y_test, y_pred, average='weighted'),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
    }

    print(f"\n📊 Classification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred, digits=3))

    return pd.DataFrame([results])



def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'category']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def prediction_metrics(pipeline, X_test, y_test):
    """
    Predicts and evaluates a trained ensemble model on test data.

    Args:
        ensemble: Trained ensemble model with predict and predict_proba methods.
        X_test (array-like): Test features.
        y_test (array-like): True labels for test data.
    """
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    metrics = {
        "ROC AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        "F1 Macro": f1_score(y_test, y_pred, average='macro'),
        "F1 Weighted": f1_score(y_test, y_pred, average='weighted'),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC:", metrics["ROC AUC"])

    return pd.DataFrame([metrics])


def print_error_rate(y_pred, y_test):
    """
    Calculates and prints total samples, number of errors, and error rate.

    Args:
        y_pred (array-like): Predictions made by the model.
        y_test (array-like): True labels.
    """

    total_samples = len(y_test)
    misclassified_idx = np.where(y_pred != y_test)[0]
    errors = len(misclassified_idx)
    error_rate = errors / total_samples

    print(f"Total Samples      : {total_samples}")
    print(f"Misclassified      : {errors}")
    print(f"Error Rate         : {error_rate:.2%}")
