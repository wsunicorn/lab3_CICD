# =============================================================================
# Lab 3: CI/CD cho ML - Deploy Script
# Mục tiêu: Deploy model lên production, hỗ trợ rollback
# Trong thực tế: copy model lên S3, update Kubernetes deployment,...
# =============================================================================

import os
import sys
import json
import shutil
import datetime


# Thư mục chứa các phiên bản model đã deploy
DEPLOYMENT_DIR = "deployments"
CURRENT_LINK = os.path.join(DEPLOYMENT_DIR, "current")


def deploy_model(model_version=None):
    """
    Deploy model mới nhất lên production.
    Chiến lược: Copy model vào versioned folder, update 'current' symlink/file.
    """
    # Đọc metrics và version
    with open("metrics/metrics.json") as f:
        metrics = json.load(f)

    version = model_version or metrics.get("model_version", "unknown")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    deploy_id = f"v{version}_{timestamp}"

    print(f"Deploying model version {version} (deploy_id: {deploy_id})")

    # Tạo thư mục deployment
    os.makedirs(DEPLOYMENT_DIR, exist_ok=True)
    version_dir = os.path.join(DEPLOYMENT_DIR, deploy_id)
    os.makedirs(version_dir, exist_ok=True)

    # Copy model artifacts
    shutil.copy("models/model.pkl",   os.path.join(version_dir, "model.pkl"))
    shutil.copy("models/scaler.pkl",  os.path.join(version_dir, "scaler.pkl"))
    shutil.copy("metrics/metrics.json", os.path.join(version_dir, "metrics.json"))

    # Lưu deployment metadata
    deploy_meta = {
        "deploy_id":   deploy_id,
        "version":     version,
        "timestamp":   timestamp,
        "metrics":     metrics,
        "status":      "active",
    }
    with open(os.path.join(version_dir, "deployment.json"), "w") as f:
        json.dump(deploy_meta, f, indent=2)

    # Cập nhật pointer 'current' -> phiên bản mới nhất
    current_file = os.path.join(DEPLOYMENT_DIR, "current.json")
    with open(current_file, "w") as f:
        json.dump({"current_deploy": deploy_id, "version": version}, f, indent=2)

    print("Deploy thanh cong!")
    print(f"Model artifacts tai: {version_dir}/")
    print(f"Current deployment: {deploy_id}")

    return deploy_id


def rollback(deploy_id=None):
    """
    Rollback về deployment trước đó.
    Nếu không chỉ định deploy_id, rollback về bản ngay trước current.
    """
    # Lấy danh sách tất cả deployments, sort theo thời gian
    if not os.path.exists(DEPLOYMENT_DIR):
        print("Chua co deployment nao.")
        return

    deployments = sorted([
        d for d in os.listdir(DEPLOYMENT_DIR)
        if os.path.isdir(os.path.join(DEPLOYMENT_DIR, d))
    ])

    if len(deployments) < 2:
        print("Can it nhat 2 deployments de rollback.")
        return

    # Đọc current deployment
    current_file = os.path.join(DEPLOYMENT_DIR, "current.json")
    with open(current_file) as f:
        current_info = json.load(f)
    current_deploy = current_info["current_deploy"]

    # Tìm deployment trước đó
    if deploy_id:
        target_deploy = deploy_id
    else:
        # Lấy deployment ngay trước current
        current_idx = deployments.index(current_deploy)
        if current_idx == 0:
            print("Dang o version cu nhat, khong the rollback.")
            return
        target_deploy = deployments[current_idx - 1]

    print(f"Rollback tu {current_deploy} -> {target_deploy}")

    # Copy artifacts từ target về models/
    target_dir = os.path.join(DEPLOYMENT_DIR, target_deploy)
    shutil.copy(os.path.join(target_dir, "model.pkl"),   "models/model.pkl")
    shutil.copy(os.path.join(target_dir, "scaler.pkl"),  "models/scaler.pkl")
    shutil.copy(os.path.join(target_dir, "metrics.json"), "metrics/metrics.json")

    # Cập nhật current pointer
    with open(current_file, "w") as f:
        json.dump({
            "current_deploy": target_deploy,
            "version": target_deploy.split("_")[0],
            "rolled_back_from": current_deploy,
        }, f, indent=2)

    print(f"Rollback thanh cong! Current deployment: {target_deploy}")


def list_deployments():
    """Liệt kê tất cả deployments và trạng thái."""
    if not os.path.exists(DEPLOYMENT_DIR):
        print("Chua co deployment nao.")
        return

    # Đọc current
    current_file = os.path.join(DEPLOYMENT_DIR, "current.json")
    current_deploy = None
    if os.path.exists(current_file):
        with open(current_file) as f:
            current_deploy = json.load(f).get("current_deploy")

    print("\n===== DEPLOYMENT HISTORY =====")
    print(f"{'Deploy ID':<35} {'Version':<10} {'Status'}")
    print("-" * 60)

    for d in sorted(os.listdir(DEPLOYMENT_DIR)):
        dpath = os.path.join(DEPLOYMENT_DIR, d)
        if not os.path.isdir(dpath):
            continue
        meta_file = os.path.join(dpath, "deployment.json")
        if os.path.exists(meta_file):
            with open(meta_file) as f:
                meta = json.load(f)
            status = "CURRENT" if d == current_deploy else "previous"
            print(f"{d:<35} {meta['version']:<10} {status}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "deploy"

    if command == "deploy":
        deploy_model()
    elif command == "rollback":
        target = sys.argv[2] if len(sys.argv) > 2 else None
        rollback(deploy_id=target)
    elif command == "list":
        list_deployments()
    else:
        print(f"Unknown command: {command}")
        print("Commands: deploy | rollback [deploy_id] | list")
        sys.exit(1)
