# Niger Delta Water Quality Synthetic Dataset Generator
# Project: Water-Quality-Index-Prediction-for-Niger-Delta-Rivers-Using-ML-and-XGBoost
# Author: Ebingiye Nelvin Agbozu

import pandas as pd
import numpy as np

# -------------------------------
# Reproducibility
# -------------------------------
np.random.seed(42)

# -------------------------------
# Generate Synthetic Dataset
# -------------------------------
n_samples = 1000

data = {
    "pH": np.random.uniform(5.5, 8.5, n_samples),
    "Dissolved_Oxygen_mg_L": np.random.uniform(2, 9, n_samples),
    "BOD_mg_L": np.random.uniform(1, 10, n_samples),
    "Turbidity_NTU": np.random.uniform(5, 150, n_samples),
    "TDS_mg_L": np.random.uniform(50, 1200, n_samples),
    "Nitrate_mg_L": np.random.uniform(0.1, 20, n_samples),
    "Phosphate_mg_L": np.random.uniform(0.01, 5, n_samples),
    "Temperature_C": np.random.uniform(24, 34, n_samples),
}

df = pd.DataFrame(data)

# -------------------------------
# Save Dataset to Excel
# -------------------------------
output_file = "Niger_Delta_Water_Quality_Synthetic_Dataset.xlsx"
df.to_excel(output_file, index=False)

print(f"Dataset generated successfully: {output_file}")
print(df.head())
