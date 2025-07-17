import sys

from project_package.entity.artifact_entity import ClassificationMetricArtifact
from project_package.exception.exception import ProjectException
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, classification_report
from project_package.logging.logger import logging


def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        """
        Calculate classification metrics such as F1 score, precision, recall, and accuracy.
        Args:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.
        """
        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score = precision_score(y_true, y_pred)
        model_accuracy_score = accuracy_score(y_true, y_pred)
        model_confusion_matrix = confusion_matrix(y_true, y_pred)
        model_classification_report = classification_report(y_true, y_pred)
        logging.info(
            f"F1 Score: {model_f1_score}, "
            f"Precision Score: {model_precision_score}, "
            f"Recall Score: {model_recall_score}, "
            f"Accuracy Score: {model_accuracy_score}, "
            f"Confusion Matrix: {model_confusion_matrix}, \n"
            f"Classification Report: {model_classification_report}"
        )

        classification_metric =  ClassificationMetricArtifact(
                    f1_score=model_f1_score,
                    precision_score=model_precision_score, 
                    recall_score=model_recall_score,
                    accuracy_score=model_accuracy_score,
                    confusion_matrix=model_confusion_matrix,
                    classification_report=model_classification_report
                    )
        return classification_metric
    except Exception as e:
        raise ProjectException(e, sys)