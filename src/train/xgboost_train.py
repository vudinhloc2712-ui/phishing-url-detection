
from xml.parsers.expat import model

import pandas as pd
import joblib
import os


from pathlib import Path
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"/"xgboost_save"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Đọc dữ liệu
def load_data():
    train_path = DATA_DIR / "train_processed.csv"
    test_path = DATA_DIR / "test_processed.csv"
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    return df_train, df_test


# 2. Chia dữ liệu Train, Validation và Test
def split_data(df_train, df_test):

    X_train = df_train.drop(columns=["label"])
    y_train = df_train["label"]

    df_val, df_test = train_test_split(
        df_test,
        test_size=0.5,
        random_state=42
    )

    X_val = df_val.drop(columns=["label"])
    y_val = df_val["label"]

    X_test = df_test.drop(columns=["label"])
    y_test = df_test["label"]

    return X_train, y_train, X_val, y_val, X_test, y_test

# Khởi tạo mô hình XGBoost
def create_model():

    model = XGBClassifier(
        n_estimators=2191,
        max_depth=5,
        learning_rate=0.01,
        random_state=123,
        eval_metric="logloss",
        early_stopping_rounds=20,
        n_jobs=-1
    )

    return model

# 3. Huấn luyện mô hình
def train_model(model, X_train, y_train, X_val, y_val):

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_train, y_train),
            (X_val, y_val)
        ],
        verbose=False
    )

    return model


# 4. Đánh giá mô hình
def evaluate_model(model, X_test, y_test):

    # Dự đoán class
    y_pred = model.predict(X_test)

    # Xác suất dự đoán class 1
    y_prob = model.predict_proba(X_test)[:, 1]

    # Các metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }

    print("\n===== Evaluation Metrics =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\n===== Classification Report =====")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    return metrics, y_test, y_pred, y_prob
def save_metrics(metrics):

    metrics_path = ARTIFACTS_DIR / "phishing_xgboost_metrics.csv"

    metrics_df = pd.DataFrame([metrics])

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    print("Saved metrics to:", metrics_path)

# 6. Lưu predictions
def save_predictions(y_test, y_pred, y_prob):

    predictions_path = (
        ARTIFACTS_DIR / "phishing_xgboost_predictions.csv"
    )

    predictions_df = pd.DataFrame({
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
    })

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    print(
        "Saved predictions to:",
        predictions_path
    )

# . Lưu mô hình
def save_model(model):

    

    joblib.dump(model, ARTIFACTS_DIR / "phishing_xgboost.joblib")

    print("Saved model to:", ARTIFACTS_DIR / "phishing_xgboost.joblib")





# . Chương trình chính
def main():

    # Đọc dữ liệu
    df_train, df_test = load_data()

    # Chia dữ liệu
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        df_train, df_test
    )

    # Khởi tạo mô hình
    model = create_model()

    # Training
    model = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val
    )

    metrics, y_test, y_pred, y_prob = evaluate_model(
        model,
        X_test,
        y_test
    )

# Lưu metrics
    save_metrics(metrics)

# Lưu y_test, y_pred, y_prob
    save_predictions(
        y_test,
        y_pred,
        y_prob
    )

# Lưu mô hình
    save_model(model)


if __name__ == "__main__":
    main()
