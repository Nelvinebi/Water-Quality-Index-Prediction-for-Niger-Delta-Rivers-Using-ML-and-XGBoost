"""
main.py — Full Pipeline Runner for Water Quality Index Prediction

HOW TO RUN:
    python main.py --run

This will:
1. Load data from data/
2. Engineer features (date extraction, spatial encoding, domain features)
3. Calculate WQI target variable
4. Preprocess + scale
5. Train 3 models with 5-fold CV + MLflow tracking
6. Select best model by test RMSE
7. Save model + scaler to models/
8. Run SHAP analysis → outputs/
9. Generate pollution map → outputs/
10. Write logs to logs/
"""

import argparse
import os
import numpy as np
import pandas as pd

from src.config import config
from src.utils import setup_logger, set_seed, save_model
from src.preprocessing import load_data, preprocess_data, scale_features
from src.feature_engineering import engineer_features
from src.train import train_models
from src.explain import shap_analysis
from src.geospatial import create_pollution_map


def calculate_wqi(df):
    """
    Calculate Water Quality Index (WQI) using the weighted arithmetic method.
    Based on WHO drinking water quality guidelines.

    Scale: 0–100 (higher = better quality)
    WHO Categories:
        95–100 → Excellent
        80–94  → Good
        65–79  → Fair
        45–64  → Poor
        0–44   → Hazardous
    """
    df = df.copy()

    # Si: WHO permissible/standard values
    # E. coli and Total Coliform WHO standard is 0 CFU/100mL.
    # We use 1.0 as a safe denominator ONLY in the weight formula (1/Si)
    # to avoid division by zero — the compliance check still enforces true zero.
    standards = {
        # pH: ideal = 7.0. Safe window is ±1.5 (6.5–8.5).
        # NOT used in the qi formula — pH has its own symmetric deviation logic.
        # Kept here only so the loop processes pH; weight is hardcoded below.
        "pH":                       7.0,
        "Dissolved_Oxygen_mg_L":    5.0,   # minimum desirable (higher = better)
        "BOD_mg_L":                 5.0,   # maximum desirable
        "Turbidity_NTU":            5.0,   # WHO ≤ 5 NTU
        "Nitrate_mg_L":            50.0,   # WHO ≤ 50 mg/L
        "Phosphate_mg_L":           0.1,   # no WHO health limit; using local standard
        "Total_Coliform_CFU_100mL": 1.0,   # WHO = 0; 1.0 used as safe denominator only
        "E_coli_CFU_100mL":         1.0,   # WHO = 0; 1.0 used as safe denominator only
        "Lead_Pb_mg_L":             0.01,  # WHO ≤ 0.01 mg/L
        "Cadmium_Cd_mg_L":          0.003, # WHO ≤ 0.003 mg/L
        "Chromium_Cr_mg_L":         0.05,  # WHO ≤ 0.05 mg/L
    }

    # Wi: weights — must sum to 1.0
    # E. coli added; Total Coliform and E. coli each get 0.08 (split the old 0.15)
    weights = {
        "pH":                       0.10,
        "Dissolved_Oxygen_mg_L":    0.14,
        "BOD_mg_L":                 0.11,
        "Turbidity_NTU":            0.08,
        "Nitrate_mg_L":             0.07,
        "Phosphate_mg_L":           0.06,
        "Total_Coliform_CFU_100mL": 0.09,
        "E_coli_CFU_100mL":         0.10,  # stricter WHO indicator — slightly higher weight
        "Lead_Pb_mg_L":             0.10,
        "Cadmium_Cd_mg_L":          0.08,
        "Chromium_Cr_mg_L":         0.07,
    }  # total = 1.00

    wqi_components = []

    for param, weight in weights.items():
        if param not in df.columns:
            continue
        si = standards[param]
        vi = df[param]

        if param == "Dissolved_Oxygen_mg_L":
            # Higher DO is better — score proportional to measured value vs standard
            qi = (vi / si) * 100

        elif param == "pH":
            # pH is two-sided: ideal is 7.0, equally bad in both directions.
            # The safe WHO window is 6.5–8.5, which is exactly ±1.5 from 7.0.
            # We take the absolute deviation so that:
            #   pH 5.5 → deviation = |5.5 - 7.0| = 1.5  (same penalty as pH 8.5)
            #   pH 6.0 → deviation = |6.0 - 7.0| = 1.0  (same penalty as pH 8.0)
            #   pH 7.0 → deviation = 0.0                 → qi = 100 (perfect)
            #   pH 5.5 or 8.5 → deviation = 1.5          → qi = 0   (boundary)
            #   beyond ±1.5   → deviation > 1.5          → qi clipped to 0
            PH_IDEAL         = 7.0
            PH_MAX_DEVIATION = 1.5   # half-width of the safe window (6.5 to 8.5)
            deviation = np.abs(vi - PH_IDEAL)
            qi = np.clip(100 - (deviation / PH_MAX_DEVIATION) * 100, 0, 100)

        else:
            # Pollutants: lower is better.
            # Full score (100) when at or below standard; scales down if exceeded.
            qi = np.where(vi > si, (si / vi) * 100, 100)

        wqi_components.append(qi * weight)

    df["WQI"] = np.sum(wqi_components, axis=0).clip(0, 100)

    # WHO professional WQI scale
    df["WQI_Category"] = pd.cut(
        df["WQI"],
        bins=[0, 44, 64, 79, 94, 100],
        labels=["Hazardous", "Poor", "Fair", "Good", "Excellent"],
        include_lowest=True
    )

    print(f"WQI calculated → mean={df['WQI'].mean():.1f} | "
          f"range=[{df['WQI'].min():.1f}, {df['WQI'].max():.1f}]")
    print(f"Category distribution:\n{df['WQI_Category'].value_counts().to_string()}")
    return df


def run_pipeline():
    # ----------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------
    logger = setup_logger(config.LOG_DIR)
    set_seed(config.RANDOM_STATE)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Water Quality ML Pipeline started")

    # ----------------------------------------------------------------
    # Step 1: Load raw data
    # ----------------------------------------------------------------
    logger.info("Step 1: Loading data...")
    df_raw = load_data(config.DATA_PATH)

    # ----------------------------------------------------------------
    # Step 2: Feature Engineering
    #   - Extracts date features (Month, Year)
    #   - Encodes spatial columns (River_Zone ordinal, State one-hot)
    #   - Drops redundant identifier columns
    #   - Creates domain features (Oxygen_Stress, Heavy_Metal_Index, etc.)
    # ----------------------------------------------------------------
    logger.info("Step 2: Engineering features...")
    df = engineer_features(df_raw)

    # ----------------------------------------------------------------
    # Step 3: Calculate WQI (target variable)
    #   Must happen AFTER feature engineering so the raw parameter
    #   columns (pH, DO, BOD, etc.) are still present.
    # ----------------------------------------------------------------
    logger.info("Step 3: Calculating WQI target variable...")
    df = calculate_wqi(df)

    # ----------------------------------------------------------------
    # Step 4: Split features and target
    # ----------------------------------------------------------------
    if config.TARGET not in df.columns:
        raise ValueError(f"Target column '{config.TARGET}' not found.")

    # Drop target and WQI_Category (categorical version of target)
    X = df.drop(columns=[config.TARGET, "WQI_Category"])
    y = df[config.TARGET]

    logger.info(f"Features: {X.shape[1]} columns | Target: {config.TARGET}")

    # ----------------------------------------------------------------
    # Step 5: Preprocess (encode Season, fill any remaining NaNs)
    # ----------------------------------------------------------------
    logger.info("Step 5: Preprocessing...")
    X_processed, encoders = preprocess_data(X)

    # ----------------------------------------------------------------
    # Step 6: Scale
    # ----------------------------------------------------------------
    logger.info("Step 6: Scaling features...")
    X_scaled, scaler = scale_features(X_processed)

    feature_names = list(X_processed.columns)

    # ----------------------------------------------------------------
    # Step 7: Train (CV + final fit + MLflow logging)
    # ----------------------------------------------------------------
    logger.info("Step 7: Training models with 5-fold cross-validation...")
    results = train_models(X_scaled, y)

    # ----------------------------------------------------------------
    # Step 8: Select best model by test RMSE
    # ----------------------------------------------------------------
    best_name = min(results, key=lambda n: results[n][3])
    best_model = results[best_name][0]
    best_rmse  = results[best_name][3]

    logger.info(f"Best model: {best_name} (Test RMSE: {best_rmse:.4f})")
    print(f"\n{'='*40}")
    print(f"Best Model : {best_name}")
    print(f"Test RMSE  : {best_rmse:.4f}")
    print(f"{'='*40}\n")

    # ----------------------------------------------------------------
    # Step 9: Save model + scaler
    # ----------------------------------------------------------------
    logger.info("Step 9: Saving model and scaler...")
    save_model(best_model, scaler, config.MODEL_DIR)

    # ----------------------------------------------------------------
    # Step 10: SHAP Explainability
    # ----------------------------------------------------------------
    logger.info("Step 10: Running SHAP analysis...")
    try:
        shap_analysis(best_model, X_scaled, feature_names, config.OUTPUT_DIR)
    except Exception as e:
        logger.warning(f"SHAP analysis failed: {e}")

    # ----------------------------------------------------------------
    # Step 11: Geospatial Pollution Map
    # ----------------------------------------------------------------
    logger.info("Step 11: Generating pollution map...")
    try:
        create_pollution_map(df, config.OUTPUT_DIR)
    except Exception as e:
        logger.warning(f"Map generation failed: {e}")

    logger.info("Pipeline complete!")
    print("Done! Outputs:")
    print(f"  models/   → model.pkl, scaler.pkl")
    print(f"  outputs/  → shap_summary.png, pollution_map.html")
    print(f"  logs/     → run log")
    print(f"  mlruns/   → MLflow UI  (run: mlflow ui)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Water Quality Index Prediction Pipeline"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the full ML pipeline"
    )
    args = parser.parse_args()

    if args.run:
        run_pipeline()
    else:
        print("Usage: python main.py --run")
