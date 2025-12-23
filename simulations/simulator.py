import time
import yaml
import logging
import requests
import numpy as np
from datetime import datetime
from typing import Dict, Optional
from tqdm import tqdm

from data_generator import CreditCardDataGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionSimulator:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.api = self.config["api"]
        self.data_generator = CreditCardDataGenerator(config_path)

        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "latency": [],
            "predictions": [],
            "errors": [],
        }

    def check_api_health(self) -> bool:
        try:
            r = requests.get(self.api["health_url"], timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def send_prediction(self, features: Dict[str, float]) -> Optional[Dict]:
        # ✅ IMPORTANT: API expects features as DICT
        payload = {"features": features}

        start = time.time()
        try:
            r = requests.post(self.api["prediction_url"], json=payload, timeout=10)
            latency = time.time() - start

            self.stats["total"] += 1
            self.stats["latency"].append(latency)

            # if r.status_code == 200:
            #     data = r.json()
            #     pred = data.get("prediction")
            #     self.stats["success"] += 1
            #     self.stats["predictions"].append(pred)
            #     return {"success": True, "prediction": pred}

            if r.status_code == 200:
                data = r.json()
                pred = data.get("prediction")
                self.stats["success"] += 1
                self.stats["predictions"].append(pred)

                if "evidently_capture_url" in self.api and self.api["evidently_capture_url"]:
                    self._capture_to_evidently(features, pred)

                return {"success": True, "prediction": pred}


            self.stats["failed"] += 1
            self.stats["errors"].append(r.text)
            return {"success": False, "error": r.text}

        except Exception as e:
            self.stats["failed"] += 1
            self.stats["errors"].append(str(e))
            return {"success": False, "error": str(e)}

    def _capture_to_evidently(self, features: Dict[str, float], prediction: float) -> None:
        """Send one prediction record to Evidently."""
        try:
            payload = {
                "features": features,                       # dict 30 features
                "prediction": float(prediction),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "model_version": str(self.config.get("model", {}).get("version", "unknown")),
            }
            r = requests.post(self.api["evidently_capture_url"], json=payload, timeout=10)

            if r.status_code >= 300:
                # log để biết fail vì gì (422/500…)
                self.stats["errors"].append(f"capture_failed {r.status_code}: {r.text}")
                logger.warning("[CAPTURE] failed %s %s", r.status_code, r.text[:200])
            else:
                logger.info("[CAPTURE] ok")
        except Exception as e:
            self.stats["errors"].append(f"capture_exception: {e}")
            logger.warning("[CAPTURE] exception: %s", e)


    def run_simulation(self, n_requests: int = 100, scenario: str = "normal", rps: float = 2.0):
        if not self.check_api_health():
            logger.error("API not healthy")
            return

        delay = 1.0 / rps
        samples = self.data_generator.generate_batch(n_requests, scenario)

        for features in tqdm(samples, desc="Sending predictions"):
            self.send_prediction(features)
            time.sleep(delay)

        self.summary()

    def summary(self):
        logger.info("\n" + "=" * 60)
        logger.info("Simulation Complete")
        logger.info("=" * 60)
        logger.info(f"Total Requests:      {self.stats['total']}")
        logger.info(f"Successful:          {self.stats['success']}")
        logger.info(f"Failed:              {self.stats['failed']}")
        if self.stats["total"] > 0:
            logger.info(f"Success Rate:        {self.stats['success'] / self.stats['total'] * 100:.2f}%")

        if self.stats["errors"]:
            logger.warning("\nErrors encountered: %d", len(self.stats["errors"]))
            for i, err in enumerate(self.stats["errors"][:5], 1):
                logger.warning("  %d. %s", i, err)

        logger.info("=" * 60)
