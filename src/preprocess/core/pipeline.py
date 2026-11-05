"""
[DÙNG CHUNG cho train và demo]
PreprocessPipeline — gom toàn bộ các bước tiền xử lý (theo đúng thứ tự trong notebook EDA):

    [tuỳ chọn] WHOIS placeholder -> bỏ near-zero-variance -> bỏ đa cộng tuyến
    -> log1p cột lệch -> StandardScaler cột liên tục

Train : xem preprocess/train/run_preprocess.py  (fit trên train, lưu pipeline)
Demo  : xem preprocess/demo/demo_preprocessor.py (nạp pipeline, biến đổi 1 URL)
"""
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

from .. import config as C
from .nzv_dropper import NearZeroVarianceDropper
from .correlation_dropper import HighCorrelationDropper
from .skew_transformer import SkewLog1p
from .feature_scaler import ContinuousScaler


@dataclass
class PreprocessPipeline:
    handle_whois: bool = False        # False = đúng như notebook hiện tại (65 đặc trưng)
    nzv_threshold: float = C.NZV_THRESHOLD
    corr_threshold: float = C.CORR_THRESHOLD
    skew_threshold: float = C.SKEW_THRESHOLD


    nzv: NearZeroVarianceDropper = field(default=None, init=False)
    corr: HighCorrelationDropper = field(default=None, init=False)
    log: SkewLog1p = field(default=None, init=False)
    scaler: ContinuousScaler = field(default=None, init=False)
    input_columns_: list = field(default_factory=list, init=False)
    feature_names_: list = field(default_factory=list, init=False)

    # ------------------------------------------------------------------ fit
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Học mọi thống kê CHỈ từ tập train."""
        self.input_columns_ = list(X_train.columns)
        X = X_train.copy()


        self.nzv = NearZeroVarianceDropper(self.nzv_threshold).fit(X)
        X = self.nzv.transform(X)

        self.corr = HighCorrelationDropper(self.corr_threshold).fit(X, y_train)
        X = self.corr.transform(X)

        self.log = SkewLog1p(self.skew_threshold).fit(X)
        X = self.log.transform(X)

        self.scaler = ContinuousScaler().fit(X, self.log.continuous_cols_)
        self.feature_names_ = list(self.scaler.transform(X).columns)
        return self

    # ------------------------------------------------------------ transform
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Áp dụng pipeline đã fit. X cần có đủ các cột đặc trưng gốc (url/status nếu có sẽ bị bỏ)."""
        X = X[self.input_columns_].copy()
        X = self.nzv.transform(X)
        X = self.corr.transform(X)
        X = self.log.transform(X)
        X = self.scaler.transform(X)
        return X[self.feature_names_]

    def fit_transform(self, X_train, y_train):
        return self.fit(X_train, y_train).transform(X_train)

    # ------------------------------------------------------------- tóm tắt
    def summary(self) -> str:
        lines = [
            f"Đầu vào        : {len(self.input_columns_)} đặc trưng",
            f"Bỏ NZV         : {len(self.nzv.dropped_)} -> {self.nzv.dropped_}",
            f"Bỏ đa cộng tuyến: {len(self.corr.dropped_)} -> {self.corr.dropped_}",
            f"Đặc trưng cuối : {len(self.feature_names_)} "
            f"(nhị phân {len(self.log.binary_cols_)}, liên tục {len(self.log.continuous_cols_)})",
            f"Đã log1p       : {len(self.log.skewed_cols_)} cột",
        ]

        return "\n".join(lines)

    # ---------------------------------------------------------- lưu / nạp
    def save(self, path: Path = None) -> Path:
        path = Path(path) if path else C.PROCESSED_DIR / C.PIPELINE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path = None) -> "PreprocessPipeline":
        path = Path(path) if path else C.PROCESSED_DIR / C.PIPELINE_FILE
        return joblib.load(path)
