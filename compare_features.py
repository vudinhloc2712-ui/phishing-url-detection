"""
So sánh 33 cột (nội dung trang + dịch vụ ngoài) do url_features.py tính ra với giá trị trong dataset gốc,
trên cùng một số URL. Chạy từ thư mục gốc project (cần mạng):

    python compare_features.py            # 30 URL legitimate + 30 URL phishing
    python compare_features.py --n 50     # nhiều URL hơn (chậm hơn)

Cách đọc kết quả (mỗi dòng = 1 cột):
    miss%      : % URL mà cột này phải điền dự phòng (tải/tra cứu thất bại)  -> KHÔNG tính vào độ khớp
    agree%     : % URL (không bị điền) mà giá trị tính ra gần bằng giá trị dataset (lệch <= 10%)
    ds_mean    : trung bình trong dataset      | my_mean : trung bình do code tính ra
Cột agree% thấp VÀ my_mean khác xa ds_mean = nhiều khả năng cách tính / quy ước giá trị bị sai.
Lưu ý: trang web đã thay đổi theo thời gian nên agree% không bao giờ đạt 100%; hãy tìm cột thấp BẤT THƯỜNG.
"""
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.preprocess.DemoPreprocessor.url_features import (
    EXTERNAL_FEATURES, HTML_FEATURES, extract_features_filled)

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=30, help="số URL mỗi lớp")
ap.add_argument("--timeout", type=float, default=8.0)
ap.add_argument("--csv", default="data/raw/dataset_phishing.csv")
a = ap.parse_args()

df = pd.read_csv(a.csv)
sample = pd.concat([df[df.status == "legitimate"].sample(a.n, random_state=7),
                    df[df.status == "phishing"].sample(a.n, random_state=7)])

cols = HTML_FEATURES + EXTERNAL_FEATURES
mine, imputed = [], []
for i, u in enumerate(sample["url"], 1):
    f, imp = extract_features_filled(u, level="full", timeout=a.timeout)
    mine.append({c: f[c] for c in cols})
    imputed.append(set(imp))
    print(f"  {i}/{len(sample)}", end="\r")
mine = pd.DataFrame(mine, index=sample.index)

rows = []
for c in cols:
    ok = np.array([c not in imp for imp in imputed])
    ds, my = sample[c].astype(float).values, mine[c].astype(float).values
    if ok.sum() == 0:
        rows.append((c, 100.0, np.nan, ds.mean(), np.nan, "không có giá trị thật nào"))
        continue
    d, m = ds[ok], my[ok]
    agree = (np.abs(d - m) <= 0.1 * np.maximum(np.maximum(np.abs(d), np.abs(m)), 1)).mean() * 100
    rows.append((c, 100 * (~ok).mean(), agree, d.mean(), m.mean(), ""))

rep = pd.DataFrame(rows, columns=["feature", "miss%", "agree%", "ds_mean", "my_mean", "note"])
rep = rep.sort_values("agree%", na_position="first").reset_index(drop=True)
with pd.option_context("display.width", 140, "display.max_rows", None):
    print("\n", rep.round(2).to_string(index=False))
print("\nTrung bình agree% (cột có dữ liệu thật):", round(rep["agree%"].mean(), 1))