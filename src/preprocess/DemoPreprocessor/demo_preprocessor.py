"""
[PHẦN DEMO] DemoPreprocessor — làm sạch 87 đặc trưng của 1 URL để đưa vào model.

KHÔNG đọc dataset, KHÔNG chia train/test, KHÔNG fit lại gì cả: chỉ nạp pipeline đã fit ở bước train
(data/processed/preprocess_pipeline.joblib) rồi biến đổi.

    from preprocess.demo import DemoPreprocessor

    prep = DemoPreprocessor()                       # nạp pipeline 1 lần khi mở app
    X = prep.transform(feats_87)                    # dict (hoặc DataFrame) 87 đặc trưng -> 65 cột sạch
    proba = model.predict_proba(X)[0, 1]
"""
from pathlib import Path

import pandas as pd

from .. import config as C
from ..core.pipeline import PreprocessPipeline


class DemoPreprocessor:
    def __init__(self, pipeline_path=None):
        self.pipeline = PreprocessPipeline.load(Path(pipeline_path) if pipeline_path
                                                else C.PROCESSED_DIR / C.PIPELINE_FILE)

    @property
    def required_columns(self):
        """87 tên đặc trưng gốc mà hàm trích xuất URL phải trả về."""
        return list(self.pipeline.input_columns_)

    @property
    def output_columns(self):
        """Các cột (đúng thứ tự) mà model mong đợi."""
        return list(self.pipeline.feature_names_)

    def validate(self, features: pd.DataFrame):
        missing = [c for c in self.required_columns if c not in features.columns]
        if missing:
            raise ValueError(f"Thiếu {len(missing)} đặc trưng: {missing}")
        bad = [c for c in self.required_columns if features[c].isna().any()]
        if bad:
            raise ValueError(f"Đặc trưng bị rỗng (NaN): {bad}")

    def transform(self, features) -> pd.DataFrame:
        """features: dict {tên: giá trị} của 1 URL, hoặc DataFrame nhiều URL. Trả về DataFrame sạch."""
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        self.validate(features)
        return self.pipeline.transform(features)
