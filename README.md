Water Quality Index Prediction for Niger Delta Rivers Using ML and XGBoost

This project applies machine learning and XGBoost to predict the Water Quality Index (WQI) of Niger Delta rivers using synthetic environmental data. It demonstrates how data-driven models can support water resource monitoring and pollution assessment.

📌 Project Objectives

Generate realistic synthetic river water-quality data

Compute Water Quality Index (WQI)

Train an XGBoost regression model for WQI prediction

Evaluate model performance using standard metrics

📂 Project Structure
├── generate_water_quality_dataset.py
├── Niger_Delta_Water_Quality_Synthetic_Dataset.xlsx
├── wqi_prediction.py
└── README.md

🧪 Parameters Used

pH

Dissolved Oxygen (DO)

Biological Oxygen Demand (BOD)

Turbidity

Total Dissolved Solids (TDS)

Nitrate

Phosphate

Temperature

These parameters reflect pollution patterns common in the Niger Delta region.

⚙️ Technologies Used

Python

NumPy

Pandas

Scikit-learn

XGBoost

Matplotlib

🚀 How to Run

Install dependencies:

pip install numpy pandas matplotlib scikit-learn xgboost


Generate the dataset:

python generate_water_quality_dataset.py


Train and evaluate the model:

python wqi_prediction.py

📊 Model Evaluation Metrics

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² Score

🌍 Environmental Relevance

The project supports sustainable water management, pollution control, and decision-making for river systems in oil-producing and industrial regions like the Niger Delta.

Author:
AGBOZU EBINGIYE NELVIN

📄 License

This project is for academic and research purposes.
