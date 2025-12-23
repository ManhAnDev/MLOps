import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from mlflow.models.signature import infer_signature
import os

# MLFlow config
mlflow.set_tracking_uri("http://mlflow:5000")
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://minio:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'

# Load dataset
df = pd.read_csv("data/creditcard.csv")
X = df.drop(columns=["Class"])
y = df["Class"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
clf.fit(X_train, y_train)

# Predict & evaluate
preds = clf.predict(X_test)
print(classification_report(y_test, preds))

# Log to MLFlow
with mlflow.start_run(run_name="creditcard_rf"):
    mlflow.sklearn.log_model(
        clf,
        artifact_path="model",
        registered_model_name="creditcard_model",
        signature=infer_signature(X_train, clf.predict(X_train))
    )
    mlflow.log_params({"n_estimators": 100, "class_weight": "balanced"})

# Promote to Production
client = mlflow.tracking.MlflowClient()
versions = client.get_latest_versions("creditcard_model", stages=["None"])
if versions:
    client.transition_model_version_stage("creditcard_model", versions[0].version, stage="Production")
