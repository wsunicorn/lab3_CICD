# =============================================================================
# Lab 3: CI/CD cho ML - Test Suite
# Mục tiêu: Kiểm tra model chất lượng trước khi deploy
# CI/CD pipeline sẽ chạy file này và fail nếu test không pass
# =============================================================================

import os
import json
import pytest
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ============================================================
# Các ngưỡng chất lượng (quality gates)
# Nếu model không đạt ngưỡng này, pipeline sẽ fail
# ============================================================
MIN_ACCURACY = 0.85   # Accuracy tối thiểu
MIN_F1_SCORE = 0.70   # F1 Score tối thiểu
MIN_ROC_AUC  = 0.85   # ROC AUC tối thiểu


# ============================================================
# Fixtures: Setup dữ liệu test dùng chung
# ============================================================
@pytest.fixture(scope="module")
def sample_input():
    """Tạo dữ liệu đầu vào mẫu để test."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 8))
    return X


@pytest.fixture(scope="module")
def trained_model():
    """Load model đã train. Yêu cầu chạy train.py trước."""
    model_path = "models/model.pkl"
    assert os.path.exists(model_path), (
        f"Model file khong ton tai: {model_path}. "
        "Chay 'python train.py' truoc."
    )
    return joblib.load(model_path)


@pytest.fixture(scope="module")
def scaler():
    """Load scaler đã train."""
    scaler_path = "models/scaler.pkl"
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    return None


@pytest.fixture(scope="module")
def metrics():
    """Load metrics đã save."""
    metrics_path = "metrics/metrics.json"
    assert os.path.exists(metrics_path), "metrics.json khong ton tai"
    with open(metrics_path) as f:
        return json.load(f)


# ============================================================
# TEST 1: Kiểm tra model artifacts tồn tại
# ============================================================
class TestModelArtifacts:
    def test_model_file_exists(self):
        """Model pkl phải tồn tại sau khi train."""
        assert os.path.exists("models/model.pkl"), "models/model.pkl khong ton tai"

    def test_scaler_file_exists(self):
        """Scaler pkl phải tồn tại."""
        assert os.path.exists("models/scaler.pkl"), "models/scaler.pkl khong ton tai"

    def test_metrics_file_exists(self):
        """Metrics json phải tồn tại."""
        assert os.path.exists("metrics/metrics.json"), "metrics/metrics.json khong ton tai"

    def test_metrics_has_required_keys(self, metrics):
        """Metrics phải có đủ các key cần thiết."""
        required_keys = ["accuracy", "f1_score", "roc_auc", "model_version"]
        for key in required_keys:
            assert key in metrics, f"Thieu key trong metrics: {key}"


# ============================================================
# TEST 2: Kiểm tra chất lượng model (Quality Gate)
# ============================================================
class TestModelQuality:
    def test_accuracy_meets_threshold(self, metrics):
        """Accuracy phải >= ngưỡng tối thiểu."""
        acc = metrics["accuracy"]
        assert acc >= MIN_ACCURACY, (
            f"Accuracy {acc:.4f} thap hon nguong {MIN_ACCURACY}. "
            "Model chua du chat luong de deploy."
        )

    def test_f1_score_meets_threshold(self, metrics):
        """F1 Score phải >= ngưỡng tối thiểu."""
        f1 = metrics["f1_score"]
        assert f1 >= MIN_F1_SCORE, (
            f"F1 Score {f1:.4f} thap hon nguong {MIN_F1_SCORE}."
        )

    def test_roc_auc_meets_threshold(self, metrics):
        """ROC AUC phải >= ngưỡng tối thiểu."""
        auc = metrics["roc_auc"]
        assert auc >= MIN_ROC_AUC, (
            f"ROC AUC {auc:.4f} thap hon nguong {MIN_ROC_AUC}."
        )


# ============================================================
# TEST 3: Kiểm tra model có thể predict được
# ============================================================
class TestModelInference:
    def test_model_can_predict(self, trained_model, sample_input, scaler):
        """Model phải trả về kết quả predict cho input mới."""
        X = sample_input.copy()
        if scaler:
            X = scaler.transform(X)
        predictions = trained_model.predict(X)
        assert len(predictions) == len(sample_input)

    def test_predictions_are_binary(self, trained_model, sample_input, scaler):
        """Với bài toán classification 2 class, output phải là 0 hoặc 1."""
        X = sample_input.copy()
        if scaler:
            X = scaler.transform(X)
        predictions = trained_model.predict(X)
        assert set(predictions).issubset({0, 1}), "Predictions phai la 0 hoac 1"

    def test_predict_proba_sums_to_one(self, trained_model, sample_input, scaler):
        """Xác suất của tất cả class phải cộng lại = 1."""
        X = sample_input.copy()
        if scaler:
            X = scaler.transform(X)
        proba = trained_model.predict_proba(X)
        # Mỗi hàng phải tổng = 1 (với tolerance nhỏ)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_model_handles_single_sample(self, trained_model, scaler):
        """Model phải xử lý được single sample (1 row)."""
        X_single = np.zeros((1, 8))  # 1 sample, 8 features
        if scaler:
            X_single = scaler.transform(X_single)
        pred = trained_model.predict(X_single)
        assert len(pred) == 1, "Du doan tren 1 sample phai tra ve 1 ket qua"

    def test_model_output_shape(self, trained_model, sample_input, scaler):
        """Output shape phải match input shape."""
        n_samples = len(sample_input)
        X = sample_input.copy()
        if scaler:
            X = scaler.transform(X)
        predictions = trained_model.predict(X)
        assert predictions.shape == (n_samples,), (
            f"Output shape {predictions.shape} khong match input {n_samples}"
        )

    def test_model_is_not_all_same_class(self, trained_model, sample_input, scaler):
        """Model không được predict tất cả cùng 1 class (underfitting check)."""
        X = sample_input.copy()
        if scaler:
            X = scaler.transform(X)
        predictions = trained_model.predict(X)
        unique_classes = set(predictions)
        assert len(unique_classes) > 1, (
            "Model predict tat ca cung 1 class - co the bi underfitting"
        )


# ============================================================
# TEST 4: Kiểm tra version model (quan trọng cho rollback)
# ============================================================
class TestModelVersion:
    def test_model_version_exists(self, metrics):
        """Model phải có version string."""
        assert "model_version" in metrics
        assert isinstance(metrics["model_version"], str)
        assert len(metrics["model_version"]) > 0

    def test_model_version_format(self, metrics):
        """Version phải theo format semantic versioning X.Y.Z."""
        version = metrics["model_version"]
        parts = version.split(".")
        assert len(parts) == 3, f"Version '{version}' phai co dang X.Y.Z"
        for part in parts:
            assert part.isdigit(), f"Moi phan version phai la so: '{part}'"
