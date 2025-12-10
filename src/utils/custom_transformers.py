""" Custom transformers for data preprocessing in machine learning pipelines.
This module contains custom transformers that can be used to sanitize column names,
replace infinite values with NaN, and select specific columns from a DataFrame.
These transformers are designed to be compatible with scikit-learn's pipeline and can be
easily integrated into machine learning workflows."""

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSanitizer(BaseEstimator, TransformerMixin):
    """Custom transformer to sanitize column names by replacing non-alphanumeric characters with underscores.
    This is useful for ensuring that column names are valid Python identifiers and do not cause issues in further processing."""    

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in X.columns]
        return X

# 🔧 Custom transformer for replacing inf with NaN
class InfReplacer(BaseEstimator, TransformerMixin):
    """ Custom transformer to replace infinite values in a DataFrame with NaN.
    This is useful for cleaning data before training models, as many algorithms cannot handle infinite values."""

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        return X.replace([np.inf, -np.inf], np.nan)
    

class ColumnSelector(BaseEstimator, TransformerMixin):
    """Custom transformer to select specific columns from a DataFrame.
    This is useful for feature selection in a machine learning pipeline."""

    def __init__(self, selected_features):
        self.selected_features = selected_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.selected_features]  