import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(path):
    """Load the Excel dataset."""
    df = pd.read_excel(path)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def preprocess_data(df):
    """
    - Fill missing values with column median (numeric only)
    - Encode only remaining string categoricals (e.g. Season → 0/1)
    - Spatial columns and date columns are already handled in
      feature_engineering.py; this step only catches anything left over.

    Returns: cleaned DataFrame + dict of encoders
    """
    df = df.copy()

    # Fill missing numeric values with column median
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Season: explicit binary encoding (Dry=0, Wet=1)
    if "Season" in df.columns:
        df["Season"] = (
            df["Season"].astype(str).str.strip().str.lower()
            .map({"dry": 0, "wet": 1})
            .fillna(0).astype(int)
        )

    # Encode any remaining object columns with LabelEncoder
    # (should be none after feature_engineering.py, but kept as safety net)
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    encoders = {}

    if categorical_cols:
        print(f"  [preprocess] Label-encoding leftover columns: {categorical_cols}")
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        print("  [preprocess] No remaining string columns — all handled upstream.")

    return df, encoders


def scale_features(X):
    """
    Standardize features using StandardScaler.
    Returns: scaled X (numpy array) + fitted scaler
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
