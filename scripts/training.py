import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import os

# Config
MODEL_NAME = "creditcard_model"
EXPERIMENT_NAME = "creditcard_training"

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"

# Load data
df = pd.read_csv("data/creditcard.csv")
X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
f1 = f1_score(y_test, preds)

# Log MLflow
with mlflow.start_run() as run:
    mlflow.log_metric("f1", f1)
    mlflow.log_params({
        "n_estimators": 100,
        "class_weight": "balanced"
    })

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name=MODEL_NAME,
        signature=infer_signature(X_train, model.predict(X_train))
    )

    run_id = run.info.run_id

# Move model to STAGING
client = MlflowClient()
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest = sorted(
    [v for v in versions if v.run_id == run_id],
    key=lambda x: int(x.version),
    reverse=True
)[0]

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=latest.version,
    stage="Staging",
    archive_existing_versions=False
)

print(f"Registered {MODEL_NAME} v{latest.version} to STAGING (f1={f1:.4f})")
