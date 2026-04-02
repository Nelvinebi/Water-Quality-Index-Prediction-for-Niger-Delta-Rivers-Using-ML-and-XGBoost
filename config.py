import os

class Config:
    DATA_PATH = "data/Niger_Delta_Water_Quality_Enriched.xlsx"
    MODEL_DIR = "models/"
    OUTPUT_DIR = "outputs/"
    LOG_DIR = "logs/"

    TARGET = "WQI"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

config = Config()
