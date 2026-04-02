import numpy as np
import pandas as pd


# Columns that are redundant identifiers — not useful as model features
_DROP_COLS = [
    "Station_Name",    # same info as Sample_Point_ID
    "River_Name",      # same info as Sample_Point_ID
    "LGA",             # too granular, collinear with State
    "Sample_Point_ID", # arbitrary string ID — no ordinal meaning
]

# River_Zone → ordered numeric (upstream cleanest on average)
_ZONE_MAP = {"Upstream": 0, "Midstream": 1, "Downstream": 2}

# State one-hot column names (4 Niger Delta states)
_STATE_COLS = ["State_Bayelsa", "State_Delta", "State_Imo", "State_Rivers"]


def engineer_features(df):
    """
    Prepares all non-WQI features for modelling:

    A. Date extraction  — Collection_Date → Month + Year integers
    B. Spatial encoding — River_Zone ordinal, State one-hot, Oil_Spill_History binary
    C. Drop redundant identifier columns
    D. Domain-specific engineered features (Heavy_Metal_Index, etc.)

    WQI is NOT calculated here — that happens in main.py after this step.
    """
    df = df.copy()

    # ----------------------------------------------------------------
    # A. DATE FEATURE EXTRACTION
    #    Raw date strings are meaningless to XGBoost; extract the
    #    useful signal (month captures seasonality, year captures trend).
    # ----------------------------------------------------------------
    if "Collection_Date" in df.columns:
        dates = pd.to_datetime(df["Collection_Date"], errors="coerce")
        df["Collection_Month"] = dates.dt.month.astype("Int64")
        df["Collection_Year"]  = dates.dt.year.astype("Int64")
        df.drop(columns=["Collection_Date"], inplace=True)
        print("  [date] Extracted Collection_Month and Collection_Year")

    # Season stays as-is (will be encoded by preprocessing.py as binary)

    # ----------------------------------------------------------------
    # B. SPATIAL ENCODING
    # ----------------------------------------------------------------

    # River_Zone: ordinal encoding (pollution generally worsens downstream)
    if "River_Zone" in df.columns:
        df["River_Zone_Encoded"] = (
            df["River_Zone"].map(_ZONE_MAP).fillna(1).astype(int)
        )
        df.drop(columns=["River_Zone"], inplace=True)
        print("  [spatial] River_Zone → River_Zone_Encoded (0/1/2)")

    # State: one-hot encoding (4 states, no ordinal relationship)
    if "State" in df.columns:
        state_dummies = pd.get_dummies(df["State"], prefix="State").reindex(
            columns=_STATE_COLS, fill_value=0
        )
        df = pd.concat([df.drop(columns=["State"]), state_dummies], axis=1)
        print(f"  [spatial] State → {_STATE_COLS}")

    # Oil_Spill_History: explicit Yes/No → 1/0 (avoids LabelEncoder ambiguity)
    if "Oil_Spill_History" in df.columns:
        df["Oil_Spill_History"] = (
            df["Oil_Spill_History"]
            .astype(str).str.strip().str.lower()
            .map({"yes": 1, "no": 0})
            .fillna(0).astype(int)
        )
        print("  [spatial] Oil_Spill_History → 1/0")

    # ----------------------------------------------------------------
    # C. DROP REDUNDANT IDENTIFIER COLUMNS
    #    These are string IDs with no predictive signal; label-encoding
    #    them would introduce false ordinal relationships.
    # ----------------------------------------------------------------
    cols_to_drop = [c for c in _DROP_COLS if c in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"  [drop] Removed identifier columns: {cols_to_drop}")

    # ----------------------------------------------------------------
    # D. DOMAIN-SPECIFIC ENGINEERED FEATURES
    # ----------------------------------------------------------------

    # 1. Heavy Metal Pollution Index (key Niger Delta concern)
    df["Heavy_Metal_Index"] = (
        df.get("Lead_Pb_mg_L", 0) +
        df.get("Cadmium_Cd_mg_L", 0) +
        df.get("Chromium_Cr_mg_L", 0)
    )

    # 2. Oxygen Stress — high BOD + low DO signals organic pollution
    df["Oxygen_Stress"] = df["BOD_mg_L"] / (df["Dissolved_Oxygen_mg_L"] + 1e-5)

    # 3. Microbial Risk Index
    df["Microbial_Risk"] = (
        df.get("Total_Coliform_CFU_100mL", 0) +
        df.get("E_coli_CFU_100mL", 0)
    )

    # 4. Human Impact Score
    df["Human_Impact"] = (
        df.get("Oil_Spill_Count", 0) +
        1 / (df.get("Proximity_To_Settlement_km", 1) + 1)
    )

    print(f"Feature engineering complete. Final shape: {df.shape}")
    return df
