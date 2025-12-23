from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Dict, Optional
import mlflow
import mlflow.pyfunc
import time
import logging
from datetime import datetime
import os

import numpy as np
import pandas as pd

# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# ============================================
# CONFIGURATION
# ============================================

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "credit_card_fraud_model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")  # Production, Staging, None/""

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================
# PROMETHEUS METRICS
# ============================================

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total predictions made",
    ["model_name", "model_version"],
)

PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Model prediction latency",
    ["model_name"],
    buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
)

CURRENT_MODEL_VERSION = Gauge(
    "model_version_info",
    "Current model version",
    ["model_name", "version"],
)

MODEL_LOAD_TIME = Gauge(
    "model_load_time_seconds",
    "Time taken to load model",
    ["model_name"],
)

# Credit-card fraud usually outputs probability in [0, 1]
PREDICTION_VALUE = Histogram(
    "model_prediction_value",
    "Distribution of prediction values",
    ["model_name"],
    buckets=[0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
)

PREDICTION_ERRORS = Counter(
    "model_prediction_errors_total",
    "Total prediction errors",
    ["model_name", "error_type"],
)

FEATURE_VALUE = Histogram(
    "model_feature_value",
    "Distribution of feature values",
    ["feature_name"],
    buckets=np.linspace(-5, 5, 41).tolist(),
)

# ============================================
# PYDANTIC MODELS
# ============================================

class PredictionRequest(BaseModel):
    """Request model for prediction (Credit Card Fraud)"""
    features: Dict[str, float] = Field(..., description="Feature dict (column -> value)")

    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "Time": 0.0,
                    "V1": -1.3598071336738,
                    "V2": -0.0727811733098497,
                    "V3": 2.53634673796914,
                    "V4": 1.37815522427443,
                    "V5": -0.338320769942518,
                    "V6": 0.462387777762292,
                    "V7": 0.239598554061257,
                    "V8": 0.0986979012610507,
                    "V9": 0.363786969611213,
                    "V10": 0.0907941719789316,
                    "V11": -0.551599533260813,
                    "V12": -0.617800855762348,
                    "V13": -0.991389847235408,
                    "V14": -0.311169353699879,
                    "V15": 1.46817697209427,
                    "V16": -0.470400525259478,
                    "V17": 0.207971241929242,
                    "V18": 0.025791652984807,
                    "V19": 0.403992960255733,
                    "V20": 0.251412098239705,
                    "V21": -0.018306777944153,
                    "V22": 0.277837575558899,
                    "V23": -0.110473910188767,
                    "V24": 0.0669280749146731,
                    "V25": 0.128539358273528,
                    "V26": -0.189114843888824,
                    "V27": 0.133558376740387,
                    "V28": -0.0210530534538215,
                    "Amount": 149.62,
                }
            }
        }


class PredictionResponse(BaseModel):
    prediction: float
    model_name: str
    model_version: str
    timestamp: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_version: str
    uptime_seconds: float


# ============================================
# MODEL MANAGER
# ============================================

class ModelManager:
    """Manage ML model from MLFlow Registry"""

    def __init__(self):
        self.model = None
        self.model_name = MODEL_NAME
        self.model_version = None
        self.model_uri = None
        self.load_time = None

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"MLFlow tracking URI: {MLFLOW_TRACKING_URI}")

    def load_model(self) -> bool:
        try:
            start_time = time.time()

            stage = MODEL_STAGE if MODEL_STAGE not in ("", "None", None) else None

            if stage:
                self.model_uri = f"models:/{self.model_name}/{stage}"
                logger.info(f"Loading model: {self.model_name} (Stage: {stage})")
            else:
                self.model_uri = f"models:/{self.model_name}/latest"
                logger.info(f"Loading model: {self.model_name} (Latest)")

            self.model = mlflow.pyfunc.load_model(self.model_uri)

            client = mlflow.tracking.MlflowClient()
            if stage:
                versions = client.get_latest_versions(self.model_name, stages=[stage])
            else:
                versions = client.get_latest_versions(self.model_name)

            self.model_version = versions[0].version if versions else "unknown"

            self.load_time = time.time() - start_time

            CURRENT_MODEL_VERSION.labels(
                model_name=self.model_name,
                version=str(self.model_version),
            ).set(int(self.model_version) if str(self.model_version).isdigit() else 0)

            MODEL_LOAD_TIME.labels(model_name=self.model_name).set(self.load_time)

            logger.info("✓ Model loaded successfully")
            logger.info(f"  Model: {self.model_name}")
            logger.info(f"  Version: {self.model_version}")
            logger.info(f"  Load time: {self.load_time:.2f}s")
            return True

        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            PREDICTION_ERRORS.labels(
                model_name=self.model_name,
                error_type="model_load_error",
            ).inc()
            return False

    def predict(self, features: Dict[str, float]) -> tuple[float, float]:
        """Make prediction (MLflow schema-safe: DataFrame with named columns)"""
        if self.model is None:
            raise ValueError("Model not loaded")

        # CRITICAL: keep column names for MLflow schema enforcement
        df = pd.DataFrame([features])

        start_time = time.time()
        pred = self.model.predict(df)
        latency = time.time() - start_time

        PREDICTION_COUNT.labels(
            model_name=self.model_name,
            model_version=str(self.model_version),
        ).inc()

        PREDICTION_LATENCY.labels(model_name=self.model_name).observe(latency)

        pred_value = float(pred[0])
        PREDICTION_VALUE.labels(model_name=self.model_name).observe(pred_value)

        return pred_value, latency


# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(
    title="ML Model API",
    description="Production ML model serving with monitoring",
    version="1.0.0",
)

model_manager = ModelManager()
app_start_time = time.time()

# ============================================
# MIDDLEWARE - REQUEST TRACKING
# ============================================

@app.middleware("http")
async def track_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=str(response.status_code),
    ).inc()

    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path,
    ).observe(latency)

    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} latency={latency:.3f}s"
    )
    return response

# ============================================
# STARTUP EVENT
# ============================================

@app.on_event("startup")
async def startup_event():
    logger.info("Starting API server...")
    ok = model_manager.load_model()
    if ok:
        logger.info("API server ready!")
    else:
        logger.error("Failed to load model on startup")

# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
async def root():
    return {
        "message": "ML Model API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "metrics": "/metrics",
            "model_info": "/model/info",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    uptime = time.time() - app_start_time
    return HealthResponse(
        status="healthy" if model_manager.model is not None else "unhealthy",
        model_loaded=model_manager.model is not None,
        model_name=model_manager.model_name,
        model_version=str(model_manager.model_version or "unknown"),
        uptime_seconds=uptime,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        if model_manager.model is None:
            PREDICTION_ERRORS.labels(
                model_name=model_manager.model_name,
                error_type="model_not_loaded",
            ).inc()
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Track feature distributions (dict-based)
        for fname, fvalue in request.features.items():
            FEATURE_VALUE.labels(feature_name=fname).observe(float(fvalue))

        start_time = time.time()
        prediction, _pred_latency = model_manager.predict(request.features)
        total_latency = time.time() - start_time

        return PredictionResponse(
            prediction=float(prediction),
            model_name=model_manager.model_name,
            model_version=str(model_manager.model_version),
            timestamp=datetime.now().isoformat(),
            latency_ms=total_latency * 1000.0,
        )

    except ValueError as e:
        PREDICTION_ERRORS.labels(
            model_name=model_manager.model_name,
            error_type="value_error",
        ).inc()
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        PREDICTION_ERRORS.labels(
            model_name=model_manager.model_name,
            error_type="unknown_error",
        ).inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.get("/model/info")
async def model_info():
    if model_manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_name": model_manager.model_name,
        "model_version": str(model_manager.model_version),
        "model_uri": model_manager.model_uri,
        "load_time_seconds": model_manager.load_time,
        "tracking_uri": MLFLOW_TRACKING_URI,
        "stage": MODEL_STAGE,
    }


@app.post("/model/reload")
async def reload_model():
    logger.info("Reloading model...")
    ok = model_manager.load_model()
    if ok:
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "model_version": str(model_manager.model_version),
        }
    raise HTTPException(status_code=500, detail="Model reload failed")


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
