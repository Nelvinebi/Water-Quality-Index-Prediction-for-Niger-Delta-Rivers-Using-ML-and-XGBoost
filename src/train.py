import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from src.config import config

# 5-fold CV used for all models
_CV_FOLDS = 5


def evaluate(y_true, preds):
    """Calculate RMSE, MAE, and R2 on held-out test set."""
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae  = mean_absolute_error(y_true, preds)
    r2   = r2_score(y_true, preds)
    return rmse, mae, r2


def cross_validate_model(model, X, y, cv=_CV_FOLDS):
    """
    Run K-Fold cross-validation and return mean ± std for RMSE and R2.
    Uses negative MSE scoring (sklearn convention) then converts to RMSE.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=config.RANDOM_STATE)

    neg_mse_scores = cross_val_score(
        model, X, y, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1
    )
    r2_scores = cross_val_score(
        model, X, y, cv=kf, scoring="r2", n_jobs=-1
    )

    cv_rmse_mean = np.sqrt(-neg_mse_scores.mean())
    cv_rmse_std  = np.sqrt(neg_mse_scores.std())   # std of MSE → approx RMSE std
    cv_r2_mean   = r2_scores.mean()
    cv_r2_std    = r2_scores.std()

    return cv_rmse_mean, cv_rmse_std, cv_r2_mean, cv_r2_std


def train_models(X, y):
    """
    Train 3 models, each with:
      - 5-fold cross-validation (robustness check)
      - Final fit on 80% train, evaluated on 20% held-out test
      - MLflow run logging (CV metrics + test metrics + params)

    Returns:
        dict of { model_name: (model_object, X_test, y_test, test_rmse) }
    """
    results = {}

    # Split data ONCE — shared across all models for fair comparison
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )

    mlflow.set_experiment("Water_Quality_Prediction")

    # ================================================================
    # MODEL 1: Linear Regression (Baseline)
    # ================================================================
    with mlflow.start_run(run_name="LinearRegression"):

        model = LinearRegression()

        # --- Cross-validation ---
        cv_rmse, cv_rmse_std, cv_r2, cv_r2_std = cross_validate_model(model, X_train, y_train)
        print(f"LinearRegression CV ({_CV_FOLDS}-fold) → "
              f"RMSE: {cv_rmse:.4f} ± {cv_rmse_std:.4f} | R2: {cv_r2:.4f} ± {cv_r2_std:.4f}")

        mlflow.log_param("cv_folds", _CV_FOLDS)
        mlflow.log_metric("CV_RMSE_mean", cv_rmse)
        mlflow.log_metric("CV_RMSE_std",  cv_rmse_std)
        mlflow.log_metric("CV_R2_mean",   cv_r2)
        mlflow.log_metric("CV_R2_std",    cv_r2_std)

        # --- Final fit + test evaluation ---
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse, mae, r2 = evaluate(y_test, preds)

        mlflow.log_metric("Test_RMSE", rmse)
        mlflow.log_metric("Test_MAE",  mae)
        mlflow.log_metric("Test_R2",   r2)
        mlflow.sklearn.log_model(model, "LinearRegression")

        print(f"LinearRegression Test  → RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}\n")
        results["LinearRegression"] = (model, X_test, y_test, rmse)

    # ================================================================
    # MODEL 2: Random Forest
    # ================================================================
    with mlflow.start_run(run_name="RandomForest"):

        model = RandomForestRegressor(
            n_estimators=200,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )

        # --- Cross-validation ---
        cv_rmse, cv_rmse_std, cv_r2, cv_r2_std = cross_validate_model(model, X_train, y_train)
        print(f"RandomForest CV ({_CV_FOLDS}-fold) → "
              f"RMSE: {cv_rmse:.4f} ± {cv_rmse_std:.4f} | R2: {cv_r2:.4f} ± {cv_r2_std:.4f}")

        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("cv_folds",     _CV_FOLDS)
        mlflow.log_metric("CV_RMSE_mean", cv_rmse)
        mlflow.log_metric("CV_RMSE_std",  cv_rmse_std)
        mlflow.log_metric("CV_R2_mean",   cv_r2)
        mlflow.log_metric("CV_R2_std",    cv_r2_std)

        # --- Final fit + test evaluation ---
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse, mae, r2 = evaluate(y_test, preds)

        mlflow.log_metric("Test_RMSE", rmse)
        mlflow.log_metric("Test_MAE",  mae)
        mlflow.log_metric("Test_R2",   r2)
        mlflow.sklearn.log_model(model, "RandomForest")

        print(f"RandomForest Test  → RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}\n")
        results["RandomForest"] = (model, X_test, y_test, rmse)

    # ================================================================
    # MODEL 3: XGBoost (Main Model)
    # ================================================================
    with mlflow.start_run(run_name="XGBoost"):

        model = XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=config.RANDOM_STATE,
            verbosity=0,
            n_jobs=-1
        )

        # --- Cross-validation ---
        cv_rmse, cv_rmse_std, cv_r2, cv_r2_std = cross_validate_model(model, X_train, y_train)
        print(f"XGBoost CV ({_CV_FOLDS}-fold) → "
              f"RMSE: {cv_rmse:.4f} ± {cv_rmse_std:.4f} | R2: {cv_r2:.4f} ± {cv_r2_std:.4f}")

        mlflow.log_param("n_estimators",  200)
        mlflow.log_param("max_depth",     5)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("cv_folds",      _CV_FOLDS)
        mlflow.log_metric("CV_RMSE_mean", cv_rmse)
        mlflow.log_metric("CV_RMSE_std",  cv_rmse_std)
        mlflow.log_metric("CV_R2_mean",   cv_r2)
        mlflow.log_metric("CV_R2_std",    cv_r2_std)

        # --- Final fit + test evaluation ---
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse, mae, r2 = evaluate(y_test, preds)

        mlflow.log_metric("Test_RMSE", rmse)
        mlflow.log_metric("Test_MAE",  mae)
        mlflow.log_metric("Test_R2",   r2)
        mlflow.sklearn.log_model(model, "XGBoost")

        print(f"XGBoost Test  → RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}\n")
        results["XGBoost"] = (model, X_test, y_test, rmse)

    return results
