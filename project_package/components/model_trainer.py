
import os
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from urllib.parse import urlparse
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf    # Should be Python 3.8–3.10 for TensorFlow 2.17.0.
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from scikeras.wrappers import KerasClassifier
from functools import partial
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier


from project_package.exception.exception import ProjectException 
from project_package.logging.logger import logging
from project_package.constants.training_pipeline import TARGET_COLUMN
from project_package.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from project_package.entity.config_entity import ModelTrainerConfig
from project_package.utils.ml_utils import model
from project_package.utils.ml_utils.model.estimator import ML_DL_Ops_Model
from project_package.utils.main_utils.utils import save_object, load_object
from project_package.utils.main_utils.utils import load_numpy_array_data, evaluate_models
from project_package.utils.ml_utils.metric.classification_metric import get_classification_score




load_dotenv()

#os.environ["MLFLOW_TRACKING_URI"]="https://github.com/Hossein88-eng/lending_club_loan_ANN"
#os.environ["MLFLOW_TRACKING_USERNAME"]="Hossein88-eng"
#os.environ["MLFLOW_TRACKING_PASSWORD"]="7104284f1bb44ece21e0e2adb4e36a250ae3251f"




def build_model(input_shape):
    print("Model received input_shape =", input_shape)
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise ProjectException(e,sys)


    """
    def build_model(self, input_shape, layers=[64, 32], activation="relu", output_activation="sigmoid", learning_rate=0.001):
        try:
            print("Model received input_shape =", input_shape)
            model = Sequential()
            model.add(Dense(layers[0], input_shape=(input_shape,), activation=activation))
            for units in layers[1:]:
                model.add(Dense(units, activation=activation))
            model.add(Dense(1, activation=output_activation))
            model.compile(
                loss="binary_crossentropy",
                optimizer=Adam(learning_rate=learning_rate),
                metrics=["accuracy"]
            )
            return model
        except Exception as e:
            raise ProjectException(e, sys)
    """

    def track_mlflow(self, best_model, classificationmetric):
        mlflow.set_registry_uri("https://github.com/Hossein88-eng/lending_club_loan_ANN")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run():
            f1_score= classificationmetric.f1_score
            precision_score= classificationmetric.precision_score
            recall_score= classificationmetric.recall_score
            accuracy_score= classificationmetric.accuracy_score
            confusion_matrix= classificationmetric.confusion_matrix
            classification_report= classificationmetric.classification_report
            
            mlflow.log_param("f1_score", f1_score)
            mlflow.log_param("precision_score", precision_score)
            mlflow.log_param("recall_score", recall_score)
            mlflow.log_param("accuracy_score", accuracy_score)
            mlflow.log_param("confusion_matrix", confusion_matrix)
            mlflow.log_param("classification_report", classification_report)            
            mlflow.sklearn.log_model(best_model, "model")
            
            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                mlflow.sklearn.log_model(best_model, "model", registered_model_name=best_model)
            else:
                mlflow.sklearn.log_model(best_model, "model")

    def train_model(self, X_train, y_train, x_test, y_test):
        models = {
                # ML models
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "AdaBoost": AdaBoostClassifier(),
                #"KNeighbors Classifier": KNeighborsClassifier(),

                # DL models
                #"Neural Network": KerasClassifier(model=build_model, input_shape=(X_train.shape[1],))
                "Neural Network": KerasClassifier(
                model=build_model,
                model__input_shape=(X_train.shape[1],),
                loss="binary_crossentropy",
                optimizer="adam",
                metrics=["accuracy"],
                verbose=1
                )
            }
        # Define hyperparameters for each model
        params={
            "Decision Tree": {
                #'criterion':['gini', 'entropy', 'log_loss'],
                ## 'splitter':['best', 'random'],
                ## 'max_features':['sqrt', 'log2'],

                'criterion':['gini', 'entropy']
            },
            "Random Forest":{
                ## 'criterion':['gini', 'entropy', 'log_loss'],
                ## 'max_features':['sqrt', 'log2', None],
                #'n_estimators': [8, 16, 32, 128, 256]

                'n_estimators': [8, 256]
            },
            "Gradient Boosting":{
                ## 'loss':['log_loss', 'exponential'],
                #'learning_rate':[0.1, 0.01, 0.05, 0.001],
                #'subsample':[0.6, 0.7, 0.75, 0.85, 0.9],
                ## 'criterion':['squared_error', 'friedman_mse'],
                ## 'max_features':['auto', 'sqrt', 'log2'],
                #'n_estimators': [8, 16, 32, 64, 128, 256]

                'learning_rate':[0.1, 0.001],
                'subsample':[0.6, 0.9],
                'n_estimators': [8, 256]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                #'learning_rate':[0.1, 0.01, 0.001],
                #'n_estimators': [8, 16, 32, 64, 128, 256]

                'learning_rate':[0.1, 0.001],
                'n_estimators': [8, 256]
            },
            "Neural Network": {
                ##"model__learning_rate": [0.001, 0.01],
                ##"model__optimizer": ["adam"],
                ##"epochs": [10, 20],
                ##"batch_size": [32]

                "batch_size": [32, 64], 
                "epochs": [10],
                "optimizer": ["adam"],
                "loss": ["binary_crossentropy"],
                "metrics": [["accuracy"]]
            }
        }
        logging.info("Evaluating models...")

        model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=x_test, y_test=y_test,
                                          models=models, param=params)

        # To get best model score from dict
        best_model_score = max(sorted(model_report.values()))

        # To get best model name from dict
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]
        y_train_pred = best_model.predict(X_train)

        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

        # Track the experiements with mlflow
        self.track_mlflow(best_model, classification_train_metric)

        y_test_pred = best_model.predict(x_test)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

        self.track_mlflow(best_model, classification_test_metric)

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        ML_DL_Ops_model = ML_DL_Ops_Model(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=ML_DL_Ops_model)

        # model pusher
        save_object("final_model/model.pkl", best_model)

        # Model Trainer Artifact
        model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_train_metric,
                             test_metric_artifact=classification_test_metric
                             )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            # Normalizing the data
            logging.info("Normalizing the data started...")
            sclaler = MinMaxScaler()
            train_arr = sclaler.fit_transform(train_arr)
            test_arr = sclaler.transform(test_arr)
            logging.info("Normalization of data completed successfully.")

            column_names = load_object(self.model_trainer_config.transformed_column_file_path)
            TARGET_COLUMN_idx = column_names.index(TARGET_COLUMN)
            x_train = np.delete(train_arr, TARGET_COLUMN_idx, axis=1)
            y_train = train_arr[:, TARGET_COLUMN_idx]
            x_test = np.delete(test_arr, TARGET_COLUMN_idx, axis=1)
            y_test = test_arr[:, TARGET_COLUMN_idx]
            logging.info(f"X_train: {x_train.shape}, y_train: {y_train.shape}, X_test: {x_test.shape}, y_test: {y_test.shape}")

            # Training the model
            logging.info("Training the model stated...")
            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            logging.info("Training the model completed successfully.")
            return model_trainer_artifact
        except Exception as e:
            raise ProjectException(e,sys)