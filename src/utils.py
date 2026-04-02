import logging
import os
import joblib
import numpy as np
import random
from datetime import datetime


def setup_logger(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # also print to terminal
        ]
    )

    return logging.getLogger(__name__)


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)


def save_model(model, scaler, path="models/"):
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, os.path.join(path, "model.pkl"))
    joblib.dump(scaler, os.path.join(path, "scaler.pkl"))
    print(f"Model and scaler saved to {path}")


def load_model(model_path="models/model.pkl", scaler_path="models/scaler.pkl"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler
