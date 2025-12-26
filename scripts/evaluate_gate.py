from mlflow.tracking import MlflowClient
import os

MODEL_NAME = "creditcard_model"
MIN_F1 = float(os.getenv("MIN_F1", 0.80))
MAX_DROP = float(os.getenv("MAX_DROP", 0.02))

client = MlflowClient()

# Get latest STAGING
staging = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
if not staging:
    raise RuntimeError("No STAGING model found")

staging_run = client.get_run(staging[0].run_id)
staging_f1 = staging_run.data.metrics.get("f1")

if staging_f1 is None:
    raise RuntimeError("STAGING model missing f1 metric")

# Hard threshold
if staging_f1 < MIN_F1:
    raise SystemExit(f"❌ Gate failed: f1={staging_f1:.4f} < {MIN_F1}")

# Compare with Production (if exists)
prod = client.get_latest_versions(MODEL_NAME, stages=["Production"])
if prod:
    prod_run = client.get_run(prod[0].run_id)
    prod_f1 = prod_run.data.metrics.get("f1")
    if prod_f1 and staging_f1 < prod_f1 - MAX_DROP:
        raise SystemExit(
            f"❌ Gate failed: staging f1 dropped too much "
            f"(staging={staging_f1:.4f}, prod={prod_f1:.4f})"
        )

print(f"✅ Gate passed (staging f1={staging_f1:.4f})")
