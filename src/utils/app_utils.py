import pandas as pd
import polars as pl
import numpy as np
import re


def clean_applicans_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Cleans and transforms a Polars DataFrame with application data.
    Applies fixes to birth/employment days, flags anomalies, and removes invalid entries.
    """
    df = df.with_columns(
        (pl.col("DAYS_BIRTH") * -1 / 365).round().alias("DAYS_BIRTH")
    )
    df = df.with_columns(
        pl.col("DAYS_EMPLOYED").abs().alias("DAYS_EMPLOYED")
    )
    df = df.with_columns(
        pl.when(pl.col("DAYS_EMPLOYED") == 365243)
          .then(None)
          .otherwise(pl.col("DAYS_EMPLOYED"))
          .alias("DAYS_EMPLOYED")
    )
    df = df.with_columns(
        pl.col("DAYS_EMPLOYED").is_null().alias("YEAR_EMPLOYED_ANOM")
    )
    df = df.filter(pl.col("CODE_GENDER") != "XNA")
    df = df.filter(pl.col("NAME_INCOME_TYPE") != "Maternity leave")
    df = df.filter(pl.col("NAME_FAMILY_STATUS") != "Unknown")

    return df



def one_hot_encoder(df, nan_as_category=True):
    """
    Performs one-hot encoding on categorical columns of a DataFrame.
    Args:
        df (pd.DataFrame): Input DataFrame.
        nan_as_category (bool): Whether to treat NaN as a separate category.
    Returns:
        Tuple[pd.DataFrame, List[str]]: Encoded DataFrame and list of new columns.
    """
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'category']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns



def create_new_features_application(df):
    """
    Adds new engineered features to the application DataFrame for modeling.
    Args:
        df (pl.DataFrame): Input Polars DataFrame.
    Returns:
        pl.DataFrame: DataFrame with new features added.
    """
    df = df.with_columns(
        pl.mean_horizontal([
            pl.col('EXT_SOURCE_1'),
            pl.col('EXT_SOURCE_2'),
            pl.col('EXT_SOURCE_3')
        ]).alias('EXT_SOURCE_MEAN')
    )
    df = df.with_columns([
        pl.concat_list([pl.col(c).is_null().cast(pl.Int32) for c in df.columns]).list.sum().alias('APP_MISSING_COUNT'),
        (pl.col('EXT_SOURCE_1') * pl.col('EXT_SOURCE_2') * pl.col('EXT_SOURCE_3')).alias('EXT_SOURCE_PRODUCT'),
        (pl.col('EXT_SOURCE_1') * pl.col('EXT_SOURCE_2')).alias('EXT_SOURCE_1_X_2'),
        (pl.col('EXT_SOURCE_1') * pl.col('EXT_SOURCE_3')).alias('EXT_SOURCE_1_X_3'),
        (pl.col('EXT_SOURCE_2') * pl.col('EXT_SOURCE_3')).alias('EXT_SOURCE_2_X_3'),
        (pl.col('EXT_SOURCE_1') * pl.col('DAYS_EMPLOYED')).alias('EXT_SOURCE_1_EMPLOYED'),
        (pl.col('EXT_SOURCE_2') * pl.col('DAYS_EMPLOYED')).alias('EXT_SOURCE_2_EMPLOYED'),
        (pl.col('EXT_SOURCE_3') * pl.col('DAYS_EMPLOYED')).alias('EXT_SOURCE_3_EMPLOYED'),
        (pl.col('EXT_SOURCE_1') / pl.col('DAYS_BIRTH')).alias('EXT_SOURCE_1_BIRTH_RATIO'),
        (pl.col('EXT_SOURCE_2') / pl.col('DAYS_BIRTH')).alias('EXT_SOURCE_2_BIRTH_RATIO'),
        (pl.col('EXT_SOURCE_3') / pl.col('DAYS_BIRTH')).alias('EXT_SOURCE_3_BIRTH_RATIO'),
        (pl.col('AMT_CREDIT') - pl.col('AMT_GOODS_PRICE')).alias('CREDIT_MINUS_GOODS'),
        (pl.col('AMT_CREDIT') / pl.col('AMT_GOODS_PRICE')).alias('CREDIT_DIV_GOODS'),
        (pl.col('AMT_CREDIT') / pl.col('AMT_ANNUITY')).alias('CREDIT_DIV_ANNUITY'),
        (pl.col('AMT_CREDIT') / pl.col('AMT_INCOME_TOTAL')).alias('CREDIT_DIV_INCOME'),
        (pl.col('AMT_INCOME_TOTAL') / 12.0 - pl.col('AMT_ANNUITY')).alias('INCOME_MONTHLY_MINUS_ANNUITY'),
        (pl.col('AMT_INCOME_TOTAL') / pl.col('AMT_ANNUITY')).alias('INCOME_DIV_ANNUITY'),
        (pl.col('AMT_INCOME_TOTAL') - pl.col('AMT_GOODS_PRICE')).alias('INCOME_MINUS_GOODS'),
        (pl.col('AMT_INCOME_TOTAL') / pl.col('CNT_FAM_MEMBERS')).alias('INCOME_DIV_FAM_SIZE'),
        pl.col('AMT_GOODS_PRICE').is_in([225000, 450000, 675000, 900000]).cast(pl.Int32).alias('GOODS_PRICE_POPULAR_TIER_1'),
        pl.col('AMT_GOODS_PRICE').is_in([1125000, 1350000, 1575000, 1800000, 2250000]).cast(pl.Int32).alias('GOODS_PRICE_POPULAR_TIER_2'),
        (pl.col('OWN_CAR_AGE') / pl.col('DAYS_BIRTH')).alias('CAR_AGE_BIRTH_RATIO'),
        (pl.col('OWN_CAR_AGE') / pl.col('DAYS_EMPLOYED')).alias('CAR_AGE_EMPLOYED_RATIO'),
        (pl.col('DAYS_LAST_PHONE_CHANGE') / pl.col('DAYS_BIRTH')).alias('PHONE_CHANGE_BIRTH_RATIO'),
        (pl.col('DAYS_LAST_PHONE_CHANGE') / pl.col('DAYS_EMPLOYED')).alias('PHONE_CHANGE_EMPLOYED_RATIO'),
        (pl.col('DAYS_EMPLOYED') - pl.col('DAYS_BIRTH')).alias('EMPLOYED_MINUS_BIRTH'),
        (pl.col('DAYS_EMPLOYED') / pl.col('DAYS_BIRTH')).alias('EMPLOYED_BIRTH_RATIO'),
    ])
    return df

def apply_saved_encoding(df, encoder_map, nan_as_category=True):
    """
    Applies saved encoding map to DataFrame columns, creating one-hot encoded columns.
    Args:
        df (pd.DataFrame): Input DataFrame.
        encoder_map (dict): Dictionary mapping columns to categories.
        nan_as_category (bool): Whether to treat NaN as a separate category.
    Returns:
        pd.DataFrame: DataFrame with encoded columns and numerical columns.
    """
    df = df.copy()
    encoded_columns = []
    for col, categories in encoder_map.items():
        col_values = df[col].astype(str).fillna('NaN') if nan_as_category else df[col].astype(str)
        for cat in categories:
            encoded_col = f"{col}_{cat}"
            encoded_series = (col_values == cat).astype(int)
            encoded_columns.append(pd.DataFrame({encoded_col: encoded_series}))
    encoded_df = pd.concat(encoded_columns, axis=1)
    numerical_df = df.drop(columns=encoder_map.keys()).select_dtypes(include=['number'])
    final_df = pd.concat([encoded_df, numerical_df], axis=1)
    return final_df

def create_application_df(df, encoder_map):
    """
    Creates a processed application DataFrame with new features and encoded columns.
    Args:
        df (pl.DataFrame): Input Polars DataFrame.
        encoder_map (dict): Dictionary mapping columns to categories.
    Returns:
        pl.DataFrame: Processed Polars DataFrame ready for modeling.
    """
    df = create_new_features_application(df)
    df_application_pd = df.to_pandas()
    df_application_pd["CODE_GENDER"] = df_application_pd["CODE_GENDER"].replace({'XNA': np.nan})
    df_application_pd["NAME_INCOME_TYPE"] = df_application_pd["NAME_INCOME_TYPE"].replace({'Maternity leave': np.nan})
    df_application_pd['NAME_FAMILY_STATUS'] = df_application_pd['NAME_FAMILY_STATUS'].replace({'Unknown': np.nan})
    df_application = apply_saved_encoding(df_application_pd, encoder_map, nan_as_category=True)
    df_application = pl.from_pandas(df_application)
    return df_application

def sanitize_columns(df):
    """
    Replace special characters in column names with underscores.
    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: DataFrame with sanitized column names.
    """
    df.columns = [
        re.sub(r'[^a-zA-Z0-9_]', '_', col)
        for col in df.columns
    ]
    return df

def replace_infinite_with_nan(df):
    """
    Replace infinite values in a DataFrame with NaN.
    Useful for cleaning data before training models.
    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: DataFrame with infinite values replaced by NaN.
    """
    return df.replace([np.inf, -np.inf], np.nan)

def prepare_data(df):
    """
    Prepares data by sanitizing column names.
    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: DataFrame with sanitized column names.
    """
    return sanitize_columns(df)