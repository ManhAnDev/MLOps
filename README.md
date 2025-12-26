# ML Monitoring & Drift Detection – Credit Card Dataset

End-to-end MLOps demo project using **FastAPI + MLflow + Evidently + Prometheus + Grafana**.

This repository demonstrates how to:
- Train and serve an ML model
- Generate production traffic
- Detect data drift
- Monitor the system using industry-standard tools

---

## Architecture

Simulator → API → Evidently → Prometheus → Grafana

---

## Requirements

- Docker & Docker Compose
- Python >= 3.9

---

## Start Infrastructure

```bash
docker-compose up -d postgres minio minio-init mlflow prometheus grafana
```
---

## Train Model

extract creditcard.zip in data folder

```bash
pip install -r scripts/requirements.txt
python scripts/training.py
```

---


### ️Train and Register Model

**⚠️ IMPORTANT**: The API requires a trained model in MLFlow Registry before it can start!

```bash
# Install Python dependencies
pip install -r scripts/requirements.txt

# Train and register model
python scripts/training.py
```

**Expected Output:**
```
 MLFlow Tracking URI: http://localhost:5000
 Model promoted to Production!
   Model: creditcard_model 
   Version: 1
   Stage: Production
```

**What this script does:**
1. Trains a RandomForest classifier on credit card dataset
2. Logs model, parameters, and metrics to MLFlow
3. Registers model in MLFlow Model Registry
4. Promotes model to **Production** stage
5. Stores artifacts in MinIO bucket

### Start Model Serving API

Now that the model is registered, start the API:

```bash
docker-compose up -d api
```

**Verify API is running:**

```bash
# Check health
curl http://localhost:8000/health

# Expected output:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "model_name": "creditcard_model ",
#   "model_version": "1",
#   "uptime_seconds": 12.34
# }
```

### Start Evidently Drift Service (Optional but Recommended)

Evidently provides drift reports and additional Prometheus metrics:

```bash
docker-compose up -d evidently
```

**Verify Evidently is running:**

```bash
curl http://localhost:8001/health
```

You should see a JSON response with `status: "healthy"`.


## Upload Reference Data

```bash
cd simulations
pip install -r requirements.txt

python - <<'PY'
import requests
from data_generator import CreditCardDataGenerator

gen = CreditCardDataGenerator("config.yaml")
ref = gen.generate_batch(500, "normal")

payload = {
    "data": ref,
    "feature_names": list(ref[0].keys())
}

print(requests.post("http://localhost:8001/reference", json=payload).status_code)
PY
```

---

## Run Normal Traffic (No Drift)

```bash
python run_simulation.py -n 300 -s normal

curl -X POST http://localhost:8001/analyze \
  -H "Content-Type: application/json" \
  -d '{"window_size":300}'
```

Expected:
```
drift_detected = false
```

---

## Trigger Drift

```bash
curl -X DELETE http://localhost:8001/production-data

python run_simulation.py -n 300 -s severe_drift

curl -X POST http://localhost:8001/analyze \
  -H "Content-Type: application/json" \
  -d '{"window_size":300}'
```

Expected:
```
drift_detected = true
```

---

## Monitoring

- Grafana: http://localhost:3000 (admin / admin)
- Prometheus: http://localhost:9090
- MLflow: http://localhost:5000

---

## Reset System

```bash
docker-compose down
docker-compose up -d
```

---

## Summary

- Normal traffic → no drift
- Drift traffic → drift detected
- Full MLOps monitoring pipeline
