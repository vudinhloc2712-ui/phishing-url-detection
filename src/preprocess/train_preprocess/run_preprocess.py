"""
[PHẦN TRAIN] Đọc dataset gốc -> chia train/test -> fit pipeline trên train -> lưu kết quả.
Chạy 1 lần (đứng ở thư mục gốc project):
    python -m preprocess.train.run_preprocess
    python -m preprocess.train.run_preprocess --whois     # thêm bước xử lý WHOIS placeholder

Đầu ra trong data/processed/:
    train_processed.csv, test_processed.csv   -> để train model
    preprocess_pipeline.joblib                -> để phần DEMO nạp lại
    preprocessing_artifacts.joblib            -> dict thông tin (giống notebook)
"""
import argparse
import sys
from pathlib import Path

import joblib

if __package__ in (None, ""):                 # cho phép: python preprocess/train/run_preprocess.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from preprocess import config as C
    from preprocess.core.pipeline import PreprocessPipeline
    from preprocess.train_preprocess.data_loader import DataLoader
    from preprocess.train_preprocess.data_splitter import DataSplitter
else:
    from .. import config as C
    from ..core.pipeline import PreprocessPipeline
    from .data_loader import DataLoader
    from .data_splitter import DataSplitter


def run(raw_path=C.RAW_PATH, out_dir=C.PROCESSED_DIR,  verbose=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y = DataLoader(raw_path).load()
    X_train, X_test, y_train, y_test = DataSplitter().split(X, y)
    if verbose:
        print(f"Train: {X_train.shape} | Test: {X_test.shape}")
        print(f"Tỉ lệ phishing - train: {y_train.mean():.3f} | test: {y_test.mean():.3f}")

    pipe = PreprocessPipeline()
    X_train_p = pipe.fit_transform(X_train, y_train)     # chỉ học thống kê từ train
    X_test_p = pipe.transform(X_test)
    if verbose:
        print(pipe.summary())

    X_train_p.assign(label=y_train.values).to_csv(out_dir / C.TRAIN_FILE, index=False)
    X_test_p.assign(label=y_test.values).to_csv(out_dir / C.TEST_FILE, index=False)
    pipe.save(out_dir / C.PIPELINE_FILE)
    joblib.dump({
        "scaler": pipe.scaler.scaler_,
        "continuous_cols": pipe.log.continuous_cols_,
        "skewed_cols": pipe.log.skewed_cols_,
        "dropped_nzv": pipe.nzv.dropped_,
        "dropped_corr": pipe.corr.dropped_,
    }, out_dir / C.ARTIFACTS_FILE)

    if verbose:
        print(f"\nĐã lưu vào {out_dir}:")
        for f in (C.TRAIN_FILE, C.TEST_FILE, C.PIPELINE_FILE, C.ARTIFACTS_FILE):
            print(" -", f)
    return X_train_p, X_test_p, y_train, y_test, pipe


def main():
    ap = argparse.ArgumentParser(description="Tiền xử lý (PHẦN TRAIN) - Phishing URL Dataset")
    ap.add_argument("--raw", default=str(C.RAW_PATH), help="đường dẫn dataset_phishing.csv")
    ap.add_argument("--out", default=str(C.PROCESSED_DIR), help="thư mục lưu kết quả")

    a = ap.parse_args()
    run(Path(a.raw), Path(a.out))


if __name__ == "__main__":
    main()
