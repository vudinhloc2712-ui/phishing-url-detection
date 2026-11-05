"""ContinuousScaler — StandardScaler chỉ cho cột liên tục."""
import pandas as pd
from sklearn.preprocessing import StandardScaler


class ContinuousScaler:
    """Scale cột liên tục (mean 0, std 1); cột nhị phân 0/1 giữ nguyên."""

    def __init__(self):
        self.scaler_ = StandardScaler()
        self.cols_ = []

    def fit(self, X: pd.DataFrame, continuous_cols):
        self.cols_ = list(continuous_cols)
        self.scaler_.fit(X[self.cols_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.cols_] = self.scaler_.transform(X[self.cols_])
        return X
