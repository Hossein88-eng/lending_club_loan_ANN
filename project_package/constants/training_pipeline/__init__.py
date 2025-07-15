import os
import numpy as np


"""
Defining common constant variable for training pipeline
"""
TARGET_COLUMN = "loan_repaid"   # This is the TARGET column as "y" to predict
PIPELINE_NAME: str = "ANNProject"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "lending_club_loan.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

#SAVED_MODEL_DIR =os.path.join("saved_models")
#MODEL_FILE_NAME = "model.pkl"


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "PROJECTDATA"
DATA_INGESTION_DATABASE_NAME: str = "HOSSEINAI"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2

"""
Data Validation related constant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

## KNN imputer to replace nan values
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}

DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"
DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"







""" 
Model Trainer Hyperparameters
"""
MODEL_TRAINER_HYPERPARAMS: dict = {
    "units_first_layer": 78,
    "units_second_layer": 39,
    "units_third_layer": 19,
    "activation": "relu",
    "decision_activation": "sigmoid",
    "epochs": 25,
    "batch_size": 256,
    "validation_split": 0.2,
    "dropout_rate": 0.2,
    "learning_rate": 0.001,
    "loss": "binary_crossentropy",
    "optimizer": "adam",
    "metrics": ["accuracy"]
}