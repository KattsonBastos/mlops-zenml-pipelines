import mlflow
import numpy as np

from sklearn.ensemble import RandomForestClassifier

# zenml importing
from zenml.steps         import step, Output, BaseStepConfig
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
from sklearn.base import ClassifierMixin


class ModelConfig(BaseStepConfig):

    model_name: str = "model"

    model_params = {
        'max_depth': 4, 
        'random_state': 42
    }


@enable_mlflow
@step(enable_cache=False)
def train_rf(X_train: np.ndarray, y_train: np.ndarray, config: ModelConfig) -> Output(
    model = ClassifierMixin
    ):
    """Training a sklearn RF model."""

    params = config.model_params

    model = RandomForestClassifier(**config.model_params)
    
    model.fit(X_train, y_train)


    # mlflow logging
    mlflow.sklearn.log_model(model,config.model_name)

    for param in params.keys():
        mlflow.log_param(f'{param}', params[param])

        
    return model


