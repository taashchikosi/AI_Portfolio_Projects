import joblib
import pandas as pd
from pathlib import Path

# Resolve base directory from this file location
BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "energy_savings_rf.joblib"
FEATURES_PATH = BASE_DIR / "models" / "features.csv"


class EnergySavingsPredictor:
    """
    Application-grade prediction wrapper.
    Loads trained model + feature contract and exposes a predict() method.
    """

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

        features_df = pd.read_csv(FEATURES_PATH)
        self.features = features_df["feature"].tolist()

    def predict(self, inputs: dict) -> float:
        # Validate feature completeness
        missing = set(self.features) - set(inputs.keys())
        if missing:
            raise ValueError(f"Missing input features: {missing}")

        # Order inputs exactly as the model expects
        X = pd.DataFrame(
            [[inputs[f] for f in self.features]],
            columns=self.features
        )

        return float(self.model.predict(X)[0])
