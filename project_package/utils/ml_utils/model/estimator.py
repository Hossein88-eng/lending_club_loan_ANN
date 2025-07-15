import os
import sys

from project_package.constants.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME
from project_package.exception.exception import ProjectException



class ML_DL_Ops_Model:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise ProjectException(e, sys)

    def predict(self, x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise ProjectException(e, sys)