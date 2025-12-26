from mlflow.tracking import MlflowClient

MODEL_NAME = "creditcard_model"

client = MlflowClient()
staging = client.get_latest_versions(MODEL_NAME, stages=["Staging"])

if not staging:
    raise RuntimeError("No STAGING model to promote")

version = staging[0].version

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=version,
    stage="Production",
    archive_existing_versions=True
)

print(f"🚀 Promoted {MODEL_NAME} v{version} to PRODUCTION")
