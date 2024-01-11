import os
import pandas as pd
from sklearn.metrics import accuracy_score as acc_score, confusion_matrix as conf_matrix, classification_report as class_report
from Wine_prediction.utils.common import save_json
from urllib.parse import urlparse
import numpy as np
import joblib
from Wine_prediction.entity.config_entity import ModelEvaluationConfig
from pathlib import Path
import json


class ModelEvaluation:
    def __init__(self, model_evaluation_config):
        self.config = model_evaluation_config

    def eval_metrics(self, actual, pred):
        # Define or import functions like acc_score, conf_matrix, class_report here
        accuracy = acc_score(actual, pred)
        confusion = conf_matrix(actual, pred)
        classification = class_report(actual, pred)
        return accuracy, confusion, classification

    def save_results(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]
        
        predicted_qualities = model.predict(test_x)

        accuracy_score, confusion_matrix, classification_report = self.eval_metrics(test_y, predicted_qualities)
        
        # Convert NumPy arrays to serializable lists
        confusion_matrix = confusion_matrix.tolist()
        
        # Ensure classification_report is in JSON format before using json.loads()
        # Example assuming classification_report is a dictionary that needs to be converted to JSON
        classification_report = json.dumps(classification_report)

        # Saving metrics as local
        scores = {"accuracy_score": accuracy_score, "confusion_matrix": confusion_matrix, "classification_report": classification_report}
        save_json(path=Path(self.config.metric_file_name), data=scores)