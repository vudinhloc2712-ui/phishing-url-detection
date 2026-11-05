"""Hằng số & đường dẫn dùng chung cho toàn bộ bước tiền xử lý."""
from pathlib import Path

# Thư mục gốc project = thư mục chứa folder `preprocess/`
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_PATH = PROJECT_DIR / "data" / "raw" / "dataset_phishing.csv"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

# Cột
URL_COL = "url"
STATUS_COL = "status"
LABEL_COL = "label"
POSITIVE_CLASS = "phishing"          # phishing -> 1, legitimate -> 0

# Chia dữ liệu
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Ngưỡng (giống notebook EDA)
NZV_THRESHOLD = 0.99                 # 1 giá trị chiếm > 99% -> bỏ cột
CORR_THRESHOLD = 0.95                # |corr| giữa 2 cột > 0.95 -> bỏ 1 cột
SKEW_THRESHOLD = 1.5                 # skew > 1.5 -> log1p



# Tên file đầu ra
TRAIN_FILE = "train_processed.csv"
TEST_FILE = "test_processed.csv"
PIPELINE_FILE = "preprocess_pipeline.joblib"      # object pipeline đã fit (dùng cho demo)
ARTIFACTS_FILE = "preprocessing_artifacts.joblib"  # dict giống notebook
