import yaml
import os,sys
import numpy as np
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from project_package.exception.exception import ProjectException
from project_package.logging.logger import logging



def read_yaml_file(file_path: str) -> dict:
    """
    Read a YAML file and return its content as a dictionary.
    Args:
        file_path (str): The path to the YAML file.
    Returns:
        dict: The content of the YAML file.
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise ProjectException(e, sys) from e
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Write content to a YAML file.
    Args:
        file_path (str): The path to the YAML file.
        content (object): The content to write to the YAML file.
        replace (bool): Whether to replace the file if it exists.
    """
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise ProjectException(e, sys)
    
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise ProjectException(e, sys) from e
    
def save_object(file_path: str, obj: object) -> None:
    """
    Save an object to a file using pickle.
    """
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise ProjectException(e, sys) from e

def load_object(file_path: str) -> object:
    """
    Load an object from a file using pickle.
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise ProjectException(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise ProjectException(e, sys) from e
    


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate multiple models using GridSearchCV and return their scores.
    Logs progress and captures model-specific exceptions.
    """
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Starting GridSearch for model: {model_name}")
            print(f"Trying model: {model_name}")

            try:
                param_grid = param.get(model_name, {})
                gs = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, error_score="raise")
                gs.fit(X_train, y_train)

                best_model = gs.best_estimator_
                logging.info(f"{model_name} - Best params: {gs.best_params_}")

                # Evaluate
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)

                train_score = r2_score(y_train, y_train_pred)
                test_score = r2_score(y_test, y_test_pred)

                logging.info(f"{model_name} - Train R2: {train_score:.4f}, Test R2: {test_score:.4f}")
                report[model_name] = test_score

                # Update model dict with trained version
                models[model_name] = best_model

            except Exception as model_err:
                logging.error(f"Model '{model_name}' failed: {model_err}")
                report[model_name] = float("-inf")

        return report

    except Exception as e:
        raise ProjectException(e, sys)