from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    precision_score,
    classification_report,
)

# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"/"knn_save"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)



# ============================================================
# 2. Load data
# ============================================================

def load_data():

    train_path = DATA_DIR / "train_processed.csv"
    test_path = DATA_DIR / "test_processed.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train data: {train_df.shape}")
    print(f"Test data : {test_df.shape}")

    return train_df, test_df


# ============================================================
# 3. Train KNN
# ============================================================

def run():
    

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------
  
    train_df, test_df = load_data()

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------
    target = "label"

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # ========================================================
    # 4. KNN
    # ========================================================

    knn = KNeighborsClassifier()

    # --------------------------------------------------------
    # GridSearchCV
    # --------------------------------------------------------
    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11, 15, 21, 25],
        "weights": ["uniform", "distance"],
        "metric": [
            "euclidean",
            "manhattan",
            "minkowski",
        ],
    }

    grid = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        refit=True,
    )

    print("\nTraining KNN with GridSearchCV...")

    grid.fit(
        X_train,
        y_train,
    )

    # ========================================================
    # 5. Best model
    # ========================================================

    best_knn = grid.best_estimator_

    print("\nBest parameters:")
    print(grid.best_params_)

    print(f"\nBest CV F1: {grid.best_score_:.4f}")

    # ========================================================
    # 6. Predict
    # ========================================================

    y_pred = best_knn.predict(X_test)

    # Probability cho ROC-AUC
    y_prob = best_knn.predict_proba(X_test)[:, 1]

    # ========================================================
    # 7. Calculate metrics
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )
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
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob,
    )

    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "best_cv_f1": grid.best_score_,
    }

    print("\nTest Metrics:")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
        )
    )

    # ========================================================
    # 8. Save model
    # ========================================================

    model_path = ARTIFACTS_DIR / "knn_model.joblib"

    joblib.dump(
        best_knn,
        model_path,
    )

    print(f"\nModel saved: {model_path}")

 

    # ========================================================
    # 10. Save metrics
    # ========================================================

    metrics_df = pd.DataFrame([
        {
            "model": "knn",
            "accuracy": accuracy,
            "f1": f1,
            "roc_auc": roc_auc,
            "best_cv_f1": grid.best_score_,
            "best_n_neighbors": grid.best_params_[
                "n_neighbors"
            ],
            "best_weights": grid.best_params_[
                "weights"
            ],
            "best_metric": grid.best_params_[
                "metric"
            ],
        }
    ])

    metrics_path = ARTIFACTS_DIR / "knn_metrics.csv"

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    print(f"Metrics saved: {metrics_path}")

    # ========================================================
    # 11. Save predictions
    # ========================================================

    predictions_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred,
        "y_prob": y_prob,
    })

    predictions_path = (
        ARTIFACTS_DIR / "knn_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    print(
        f"Predictions saved: "
        f"{predictions_path}"
    )


# ============================================================
# 12. Entry point
# ============================================================

if __name__ == "__main__":
    run()