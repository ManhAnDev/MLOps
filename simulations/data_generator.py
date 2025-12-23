import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import yaml


class CreditCardDataGenerator:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        ds = self.config["dataset"]
        self.dataset_path = ds["path"]
        self.target_col = ds["target_column"]
        self.feature_columns: List[str] = ds["feature_columns"]

        df = pd.read_csv(self.dataset_path)

        # baseline traffic: normal transactions only
        df = df[df[self.target_col] == 0]
        self.df = df[self.feature_columns].reset_index(drop=True)

        self.std = self.df.std(numeric_only=True).replace(0, 1e-6)

    def generate_normal_sample(self) -> Dict[str, float]:
        row = self.df.sample(1).iloc[0]
        return {c: float(row[c]) for c in self.feature_columns}

    def generate_drifted_sample(
        self,
        drift_multiplier: float,
        affected_features: Optional[List[str]],
        noise_level: float,
    ) -> Dict[str, float]:
        """
        Create drifted sample that Evidently can detect reliably on credit-card PCA features.
        Strategy:
        - For PCA-like features: scale value strongly and/or shift by std
        - For Amount: scale + additive shift (easy to detect)
        - Add noise proportional to std
        """
        sample = self.generate_normal_sample()

        # If user doesn't specify affected features, pick some PCA features (exclude Time, Amount by default)
        if not affected_features:
            candidates = [c for c in self.feature_columns if c not in ("Time", "Amount")]
            # pick 5 PCA features
            affected_features = list(np.random.choice(candidates, size=min(5, len(candidates)), replace=False))

        # Always include Amount drift to make dataset drift detectable
        if "Amount" in self.feature_columns and "Amount" not in affected_features:
            affected_features = affected_features + ["Amount"]

        for feat in affected_features:
            base = float(sample[feat])
            sigma = float(self.std.get(feat, 1.0))
            noise = float(np.random.normal(0, sigma * noise_level))

            if feat == "Amount":
                # Make Amount drift very visible
                # (scale + shift) then keep non-negative
                val = base * max(drift_multiplier, 2.0) + 1000.0 + noise
                val = max(0.0, val)

            elif feat == "Time":
                # Optional: shift time a bit (usually not necessary)
                val = base + 20000.0 + noise

            else:
                # PCA features: scaling base might be too weak if base ~ 0,
                # so also shift by a few sigmas.
                # drift_multiplier=2.0 -> shift by 2*sigma, 4.0 -> 4*sigma, etc.
                val = (base * drift_multiplier) + (drift_multiplier * sigma) + noise

            sample[feat] = float(val)

        return sample


    def generate_batch(self, n_samples: int = 100, scenario: str = "normal") -> List[Dict[str, float]]:
        scen = self.config.get("scenarios", {}).get(scenario, {})
        drift_multiplier = float(scen.get("drift_multiplier", 1.0))
        noise_level = float(scen.get("noise_level", 0.05))
        affected_n = int(scen.get("affected_features", 0))

        batch = []
        for _ in range(n_samples):
            if scenario == "normal":
                batch.append(self.generate_normal_sample())
            else:
                features = None
                if affected_n > 0:
                    candidates = [c for c in self.feature_columns if c != "Time"]
                    features = list(np.random.choice(candidates, size=affected_n, replace=False))
                batch.append(
                    self.generate_drifted_sample(
                        drift_multiplier=drift_multiplier,
                        affected_features=features,
                        noise_level=noise_level,
                    )
                )
        return batch
