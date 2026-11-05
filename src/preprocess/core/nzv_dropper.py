"""NearZeroVarianceDropper — bỏ cột gần như hằng số."""
import pandas as pd

from .. import config as C


class NearZeroVarianceDropper:
    """Bỏ cột có 1 giá trị chiếm > threshold số dòng (đo trên tập train)."""

    def __init__(self, threshold: float = C.NZV_THRESHOLD):
        self.threshold = threshold
        self.dropped_ = []

    def fit(self, X: pd.DataFrame):
        self.dropped_ = [c for c in X.columns
                         if X[c].value_counts(normalize=True).iloc[0] > self.threshold]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.dropped_, errors="ignore")