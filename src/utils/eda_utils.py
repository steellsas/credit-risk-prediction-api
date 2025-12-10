
import itertools
import pandas as pd
import polars as pl
import numpy as np
import phik


from collections import Counter



class PhikCorrelationChecker:
    def __init__(self, df, target_col, sample_frac=1.0, random_state=42):
        self.df = df.copy()
        self.target_col = target_col
        self.sample_frac = sample_frac
        self.random_state = random_state
        self.phik_matrix = None

    def sample_data(self):
        if self.sample_frac < 1.0:
            self.df = self.df.sample(frac=self.sample_frac, random_state=self.random_state)

    def compute_phik_matrix(self):
        if self.sample_frac < 1.0:
            self.sample_data()
        self.phik_matrix = self.df.phik_matrix()
        
    def filter_by_threshold(self, threshold):
        """Return features with correlation > threshold."""
        corr_df = self.get_correlations()
        print(corr_df.shape)
        return corr_df[corr_df['phik_correlation'].abs() >= threshold]    

    def get_correlations(self):
        if self.phik_matrix is None:
            self.compute_phik_matrix()
        correlations = []
        for col in self.phik_matrix.columns:
            if col != self.target_col:
                corr = self.phik_matrix.loc[self.target_col, col]
                correlations.append({'feature': col, 'phik_correlation': corr})
        return pd.DataFrame(correlations)
    
    def get_feature_feature_correlations(self):

        if self.phik_matrix is None:
            self.compute_phik_matrix()
     
        features = self.phik_matrix.columns
        correlations = [
            {
                'feature_1': f1,
                'feature_2': f2,
                'phik_correlation': self.phik_matrix.loc[f1, f2]
            }
            for f1, f2 in itertools.combinations(features, 2)
        ]
        return pd.DataFrame(correlations)


def get_missing_percentages(df: pl.DataFrame) -> pl.DataFrame:
    features = []
    missing_percents = []

    for col in df.columns:
        missing_pct = df[col].is_null().mean() * 100  # compute the float value
        features.append(col)
        missing_percents.append(float(missing_pct))  # ensure it's a float

    return pl.DataFrame({
        'feature': features,
        'missing_percentage': missing_percents
    })


def get_numerical_columns(df: pl.DataFrame):
    numeric_types = [
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64
    ]
    numeric_cols = [col for col in df.columns if df[col].dtype in numeric_types]
    return numeric_cols


def get_categorical_columns(df: pl.DataFrame):
    categorical_types = [pl.Utf8, pl.Categorical]
    categorical_cols = [col for col in df.columns if df[col].dtype in categorical_types]
    return categorical_cols



def find_outlier_rows_by_iqr(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Identify and return rows in a DataFrame that contain outliers based on the
    Interquartile Range (IQR) method.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to analyze for outliers.
    columns : list, optional
        A list of column names to check for outliers. If None (default), the function
        will analyze all numerical columns (int64 and float64).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only the rows from the original DataFrame that are
        identified as having outliers in any of the specified columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    outlier_mask = pd.Series(False, index=df.index)

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        col_outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_mask |= col_outlier_mask

    return df[outlier_mask]


def outliers_counts(df: pd.DataFrame) -> None:
    """
    Print the count of outliers for each numerical column in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    """
    total_outliers = 0
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            outlier_rows = find_outlier_rows_by_iqr(df, columns=[col])
            print(f"{col}: {outlier_rows.shape[0]} outliers")
            total_outliers += outlier_rows.shape[0]
    print(f"Total outliers: {total_outliers}")






def reduce_memory_usage_pl(
    df: pl.DataFrame, verbose: bool = True
) -> pl.DataFrame:
    """Reduces memory usage of a Polars DataFrame by optimizing data types.

    Attempts to downcast numeric columns to the smallest possible data type
    that can represent the data without loss of information. Also converts
    string columns to categorical type.

    Args:
        df: A Polars DataFrame to optimize.
        verbose: If True, prints memory usage before and after optimization.

    Returns:
        A Polars DataFrame with optimized memory usage.
    """
    if verbose:
        print(f"Size before reduction: {df.estimated_size('mb'):.2f} MB")
        print(f"Initial data types: {Counter(df.dtypes)}")

    numeric_int_types = [pl.Int8, pl.Int16, pl.Int32, pl.Int64]
    numeric_float_types = [pl.Float32, pl.Float64]

    for col in df.columns:
        col_type = df[col].dtype

        if col_type in numeric_int_types:
            c_min = df[col].min()
            c_max = df[col].max()

            if np.iinfo(np.int8).min <= c_min <= c_max <= np.iinfo(np.int8).max:
                new_type = pl.Int8
            elif np.iinfo(np.int16).min <= c_min <= c_max <= np.iinfo(np.int16).max:
                new_type = pl.Int16
            elif np.iinfo(np.int32).min <= c_min <= c_max <= np.iinfo(np.int32).max:
                new_type = pl.Int32
            else:
                new_type = pl.Int64

            if new_type != col_type:
                df = df.with_columns(pl.col(col).cast(new_type))

        elif col_type in numeric_float_types:
            c_min = df[col].min()
            c_max = df[col].max()
            if np.finfo(np.float32).min <= c_min <= c_max <= np.finfo(np.float32).max:
                df = df.with_columns(pl.col(col).cast(pl.Float32))

        elif col_type == pl.Utf8:
            df = df.with_columns(pl.col(col).cast(pl.Categorical))

    if verbose:
        print(f"Size after reduction: {df.estimated_size('mb'):.2f} MB")
        print(f"Final data types: {Counter(df.dtypes)}")

    return df

def convert_selected_numeric_to_categorical(df: pl.DataFrame, columns_to_convert: list) -> pl.DataFrame:
    new_cols = []
    for col in df.columns:
        if col in columns_to_convert:
            col_series = df[col].cast(pl.Utf8).cast(pl.Categorical)
        else:
            col_series = df[col]
        new_cols.append(pl.Series(name=col, values=col_series))

    return pl.DataFrame(new_cols)

def find_integer_categorical_columns(df: pd.DataFrame, max_unique: int = 3) -> list:
    """
    Identifies numeric columns with a limited number of unique values,
    which are likely to be categorical.

    Parameters:
    - df: Input DataFrame
    - max_unique: Maximum number of unique values to consider as categorical (default = 3)

    Returns:
    - List of column names that are numeric and have <= max_unique unique values
    """
    int_cat_cols = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if df[col].nunique(dropna=True) <= max_unique:
            int_cat_cols.append(col)

    return int_cat_cols



def get_categorical_columns(df: pl.DataFrame):
    categorical_types = [pl.Utf8, pl.Categorical]
    categorical_cols = [col for col in df.columns if df[col].dtype in categorical_types]
    return categorical_cols
 

 
def get_numerical_columns(df: pl.DataFrame):
    numeric_types = [
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64
    ]
    numeric_cols = [col for col in df.columns if df[col].dtype in numeric_types]
    return numeric_cols
