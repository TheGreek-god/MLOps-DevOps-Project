#!/usr/bin/env python3
"""
Deployment script for F1Net K8s cluster.
Loads Docker images into Minikube and applies all K8s manifests.
"""
import subprocess
import sys
import os

IMAGES = [
    "f1net-ingest:latest",
    "f1net-predict:latest",
    "f1net-finetune:latest",
    "f1net-frontend:latest",
]

K8S_MANIFESTS = [
    "k8s/namespace.yml",
    "k8s/secret.yml",
    "k8s/monitoring/prometheus.yml",
    "k8s/monitoring/grafana.yml",
    "k8s/monitoring/mlflow.yml",
    "k8s/ingest/deploy.yml",
    "k8s/predict/deploy.yml",
    "k8s/finetune/deploy.yml",
    "k8s/frontend/deploy.yml",
]


def run(cmd, check=True):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        print(f"    {result.stdout.strip()[:200]}")
    if result.returncode != 0 and result.stderr.strip():
        print(f"    stderr: {result.stderr.strip()[:200]}")
    if check and result.returncode != 0:
        print(f"    FAILED (exit {result.returncode})")
        return False
    return True


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    print("=" * 60)
    print("F1Net K8s Deployment")
    print("=" * 60)

    print("\n1. Loading Docker images into Minikube...")
    for img in IMAGES:
        if not run(f"minikube image load {img}", check=False):
            print(f"    WARNING: Failed to load {img}")

    print("\n2. Applying K8s manifests...")
    for manifest in K8S_MANIFESTS:
        if os.path.exists(manifest):
            if not run(f"kubectl apply -f {manifest}"):
                print(f"    WARNING: Failed to apply {manifest}")
        else:
            print(f"    SKIP: {manifest} not found")

    print("\n3. Waiting for deployments...")
    run("kubectl rollout status deployment/ingest-deploy -n f1net --timeout=120s", check=False)
    run("kubectl rollout status deployment/predict-deploy -n f1net --timeout=120s", check=False)
    run("kubectl rollout status deployment/finetune-deploy -n f1net --timeout=120s", check=False)
    run("kubectl rollout status deployment/frontend-deploy -n f1net --timeout=120s", check=False)

    print("\n4. Service status:")
    run("kubectl get all -n f1net", check=False)

    print("\n" + "=" * 60)
    print("Deployment complete!")
    print("=" * 60)
    print("  Frontend:  http://localhost:30083")
    print("  Ingest:    http://localhost:30080")
    print("  Predict:   http://localhost:30081")
    print("  Finetune:  http://localhost:30082")
    print("  Grafana:   http://localhost:30030")
    print("  MLflow:    http://localhost:30050")
    print("  Prometheus: http://localhost:30090")


if __name__ == "__main__":
    main()
