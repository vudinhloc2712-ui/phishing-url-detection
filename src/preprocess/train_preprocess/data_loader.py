"""DataLoader — đọc dữ liệu thô và tách (X, y)."""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .. import config as C


@dataclass
class DataLoader:
    raw_path: Path = C.RAW_PATH

    def load_raw(self) -> pd.DataFrame:
        """Đọc dataset_phishing.csv (url + 87 đặc trưng + status)."""
        return pd.read_csv(self.raw_path)

    def split_xy(self, df: pd.DataFrame):
        """Bỏ `url`, mã hoá `status` -> `label` (phishing=1). Trả về (X, y)."""
        y = (df[C.STATUS_COL] == C.POSITIVE_CLASS).astype(int).rename(C.LABEL_COL)
        X = df.drop(columns=[C.URL_COL, C.STATUS_COL]).copy()
        return X, y

    def load(self):
        return self.split_xy(self.load_raw())
