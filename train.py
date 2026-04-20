# =============================================================================
# Lab 3: CI/CD cho ML - Training Script
# Mục tiêu: Train model và lưu để pipeline CI/CD sử dụng
# =============================================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


# Version của model - tăng mỗi khi có thay đổi lớn
MODEL_VERSION = "1.0.0"
RANDOM_SEED = 42


def generate_training_data():
    """Tạo dữ liệu huấn luyện (fraud detection)."""
    rng = np.random.default_rng(RANDOM_SEED)
    n = 3000

    X = rng.standard_normal((n, 8))  # 8 features
    # Label: fraud nếu tổng 3 features đầu > 1.5
    y = ((X[:, 0] + X[:, 1] + X[:, 2]) > 1.5).astype(int)

    feature_names = [f"feature_{i}" for i in range(8)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/train_data.csv", index=False)
    return df


def train_model(data_path="data/train_data.csv"):
    """
    Huấn luyện model và lưu kết quả.
    Returns: dict chứa metrics để CI/CD kiểm tra quality gate.
    """
    df = pd.read_csv(data_path)
    target = "target"
    features = [c for c in df.columns if c != target]

    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Chuẩn hóa
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)

    # Đánh giá
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":    round(accuracy_score(y_test, y_pred), 4),
        "f1_score":    round(f1_score(y_test, y_pred), 4),
        "roc_auc":     round(roc_auc_score(y_test, y_prob), 4),
        "model_version": MODEL_VERSION,
        "feature_names": features,
    }

    # Lưu artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("===== TRAINING COMPLETE =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return metrics


if __name__ == "__main__":
    generate_training_data()
    metrics = train_model()
    print("\nModel saved to models/model.pkl")
