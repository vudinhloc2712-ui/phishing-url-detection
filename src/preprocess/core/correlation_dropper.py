"""HighCorrelationDropper — bỏ cột đa cộng tuyến."""
import numpy as np
import pandas as pd

from .. import config as C


class HighCorrelationDropper:
    """Với mỗi cặp |corr| > threshold: giữ cột tương quan với nhãn mạnh hơn, bỏ cột còn lại."""

    def __init__(self, threshold: float = C.CORR_THRESHOLD):
        self.threshold = threshold
        self.dropped_ = []
        self.pairs_ = []          # (a, b, corr, cột bị loại) để báo cáo

    def fit(self, X: pd.DataFrame, y: pd.Series):
        corr_abs = X.corr().abs()
        target_corr = X.corrwith(y).abs()
        upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
        pairs = upper.stack().sort_values(ascending=False)

        dropped, log = set(), []
        for (a, b), val in pairs[pairs > self.threshold].items():
            if a in dropped or b in dropped:
                continue
            drop = a if target_corr[a] < target_corr[b] else b
            dropped.add(drop)
            log.append((a, b, float(val), drop))
        self.dropped_, self.pairs_ = list(dropped), log
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.dropped_, errors="ignore")
