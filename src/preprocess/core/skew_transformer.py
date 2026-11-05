"""SkewLog1p — log1p cho các cột liên tục lệch phải mạnh."""
import numpy as np
import pandas as pd

from .. import config as C


class SkewLog1p:
    """Chọn (trên train) cột liên tục có min >= 0 và skew > threshold rồi áp log1p.
    Cũng ghi nhớ danh sách cột nhị phân / liên tục để bước scale dùng lại."""

    def __init__(self, threshold: float = C.SKEW_THRESHOLD):
        self.threshold = threshold
        self.binary_cols_ = []
        self.continuous_cols_ = []
        self.skewed_cols_ = []

    def fit(self, X: pd.DataFrame):
        self.binary_cols_ = [c for c in X.columns if set(X[c].unique()).issubset({0, 1})]
        self.continuous_cols_ = [c for c in X.columns if c not in self.binary_cols_]
        self.skewed_cols_ = [c for c in self.continuous_cols_
                             if X[c].min() >= 0 and X[c].skew() > self.threshold]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.skewed_cols_:
            X[c] = np.log1p(X[c].clip(lower=0))   # clip: phòng giá trị âm ngoài tập train
        return X
