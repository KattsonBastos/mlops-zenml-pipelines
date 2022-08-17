import mlflow

import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# zenml importing
from zenml.steps         import step, Output
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow


@enable_mlflow 
@step(enable_cache=False)
def evaluate_model(model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Output(
    accuracy = float, precision = float, recall = float, f1 = float
    ):
    
    y_preds = model.predict(X_test)

    # metricas
    accuracy = accuracy_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds, average = 'macro')
    recall = recall_score(y_test, y_preds, average = 'macro')
    f1 = f1_score(y_test, y_preds, average = 'macro')

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)

    return accuracy, precision, recall, f1