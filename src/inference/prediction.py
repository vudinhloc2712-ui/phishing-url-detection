"""
Dự đoán 1 URL là phishing hay legitimate (dùng cho demo / app).

Luồng:  URL -> url_features (87 đặc trưng, tự điền giá trị dự phòng nếu không tra được)
            -> DemoPreprocessor (87 -> 65 cột sạch)
            -> model đã train (artifacts/*.joblib) -> xác suất phishing

Cách dùng trong code:
    from src.inference.prediction import PhishingPredictor
    predictor = PhishingPredictor(model="xgboost")      # nạp 1 lần khi mở app
    result = predictor.predict("http://abc.tk/login")
    # {'url':..., 'label': 'phishing', 'proba_phishing': 0.93, 'reliable': True, 'warnings': [...], ...}

Chạy thử từ thư mục gốc project:
    python -m src.inference.prediction "http://abc.tk/login" --model knn
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]      # .../project  (file nằm ở src/inference/)
SRC_DIR = PROJECT_DIR / "src"

PIPELINE_PATH = PROJECT_DIR / "data" / "processed" / "preprocess_pipeline.joblib"
MODEL_PATHS = {
    "knn": PROJECT_DIR / "artifacts" / "knn_save" / "knn_model.joblib",
    "xgboost": PROJECT_DIR / "artifacts" / "xgboost_save" / "phishing_xgboost.joblib",
}

# Các cột đến từ dịch vụ ngoài và rất quan trọng với model: nếu phải điền dự phòng thì kết quả kém tin cậy
KEY_EXTERNAL_COLS = {"google_index", "page_rank", "web_traffic", "domain_age"}


def _import_preprocess():
    """Import DemoPreprocessor + extract_features_filled, chịu được cả 2 cách chạy:
    từ thư mục gốc (src.preprocess...) hoặc từ thư mục src (preprocess...)."""
    try:
        from src.preprocess.DemoPreprocessor.demo_preprocessor import DemoPreprocessor
        from src.preprocess.DemoPreprocessor.url_features import extract_features_filled
    except ModuleNotFoundError as e:
        if not (e.name or "").startswith("src"):
            raise
        sys.path.insert(0, str(SRC_DIR))
        from preprocess.DemoPreprocessor.demo_preprocessor import DemoPreprocessor
        from preprocess.DemoPreprocessor.url_features import extract_features_filled
    return DemoPreprocessor, extract_features_filled


class PhishingPredictor:
    def __init__(self, model: str = "xgboost", threshold: float = 0.5, level: str = "full",
                 timeout: float = 8.0, model_path=None, pipeline_path=PIPELINE_PATH):
        """
        model     : 'knn' hoặc 'xgboost' (chọn file trong MODEL_PATHS), hoặc truyền model_path riêng
        threshold : xác suất phishing >= threshold thì kết luận phishing
        level     : 'full' (87 cột, cần mạng) | 'html' | 'url' -- demo dùng model 65 cột nên cần 'full'
        timeout   : thời gian chờ (giây) cho mỗi lần tải trang / tra cứu
        """
        DemoPreprocessor, self._extract = _import_preprocess()

        path = Path(model_path) if model_path else MODEL_PATHS.get(model)
        if path is None:
            raise ValueError(f"model phải thuộc {list(MODEL_PATHS)}, nhận '{model}'")
        if not path.exists():
            raise FileNotFoundError(f"Không thấy file model: {path}")
        if not Path(pipeline_path).exists():
            raise FileNotFoundError(f"Không thấy {pipeline_path} - chạy run_preprocess trước")

        try:
            self.preprocessor = DemoPreprocessor(pipeline_path)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"{e}. File preprocess_pipeline.joblib được tạo bằng một cách import khác "
                f"(vd. 'preprocess.core...' thay vì 'src.preprocess.core...'). "
                f"Hãy chạy lại run_preprocess bằng đúng cách bạn chạy demo.") from e

        self.model = joblib.load(path)
        self.model_name = model if not model_path else path.stem
        self.threshold = threshold
        self.level = level
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    def predict_features(self, features: dict, imputed=(), url: str = "") -> dict:
        """Dự đoán từ dict 87 đặc trưng (đã đủ, không NaN)."""
        X = self.preprocessor.transform(features)
        proba = float(self.model.predict_proba(X)[0, 1])

        imputed = list(imputed)
        key_missing = sorted(KEY_EXTERNAL_COLS & set(imputed))
        warnings = []
        if key_missing:
            warnings.append(f"Không tra cứu được {key_missing} -> dùng giá trị dự phòng, "
                            f"kết quả kém tin cậy (kiểm tra mạng / API key).")
        page_cols = [c for c in imputed if c not in KEY_EXTERNAL_COLS]
        if page_cols:
            warnings.append(f"Không tải được nội dung trang ({len(page_cols)} cột) -> coi như trang rỗng.")

        return {
            "url": url,
            "label": "phishing" if proba >= self.threshold else "legitimate",
            "proba_phishing": round(proba, 4),
            "threshold": self.threshold,
            "model": self.model_name,
            "reliable": not key_missing,
            "n_imputed": len(imputed),
            "imputed": imputed,
            "warnings": warnings,
        }

    def predict(self, url: str) -> dict:
        """Dự đoán từ 1 URL thô."""
        url = (url or "").strip()
        if not url:
            raise ValueError("URL rỗng")
        feats, imputed = self._extract(url, level=self.level, timeout=self.timeout)
        return self.predict_features(feats, imputed, url=url)

    def predict_many(self, urls) -> pd.DataFrame:
        """Dự đoán nhiều URL, trả về DataFrame (bỏ cột imputed/warnings cho gọn)."""
        rows = [self.predict(u) for u in urls]
        return pd.DataFrame(rows).drop(columns=["imputed", "warnings"])


# ---------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Dự đoán URL phishing")
    ap.add_argument("url", help="URL cần kiểm tra")
    ap.add_argument("--model", default="xgboost", choices=list(MODEL_PATHS))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--timeout", type=float, default=8.0)
    a = ap.parse_args(argv)

    r = PhishingPredictor(a.model, a.threshold, timeout=a.timeout).predict(a.url)
    print(f"URL        : {r['url']}")
    print(f"Kết luận   : {r['label'].upper()}  (xác suất phishing = {r['proba_phishing']:.2%}, model = {r['model']})")
    print(f"Độ tin cậy : {'ổn' if r['reliable'] else 'THẤP'}  (điền dự phòng {r['n_imputed']}/87 cột)")
    for w in r["warnings"]:
        print("  !", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())