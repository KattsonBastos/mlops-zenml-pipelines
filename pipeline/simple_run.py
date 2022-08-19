import mlflow

import numpy            as np
import pandas           as pd

from sklearn.metrics                import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble               import RandomForestClassifier
from sklearn.model_selection        import train_test_split

# zenml importing
from zenml.steps                                        import step, Output, BaseStepConfig
from sklearn.base                                       import ClassifierMixin
from zenml.pipelines                                    import pipeline
from zenml.integrations.mlflow.mlflow_step_decorator    import enable_mlflow


def get_clf_metrics(
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ):

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average = 'macro')
    recall = recall_score(y_true, y_pred, average = 'macro')
    f1 = f1_score(y_true, y_pred, average = 'macro')

    return accuracy, precision, recall, f1


@step
def load_data() -> Output(
    data=pd.DataFrame
    ):

    """Load a dataset."""

    data = pd.read_csv('../data/cardio_train_sampled.csv')

    return data


@step
def data_preparation(data: pd.DataFrame) -> Output(
    X_train = np.ndarray, X_test = np.ndarray, y_train = np.ndarray, y_true = np.ndarray
):
    dataframe = data.copy()
    dataframe['age'] = round(dataframe['age'] / 365).astype(int)

    X = data.drop('cardio', axis = 1).values
    y = data['cardio'].values
    X_train, X_test, y_train, y_true = train_test_split(X, y, test_size = 0.2, random_state=42)

    return X_train, X_test, y_train, y_true



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

@enable_mlflow 
@step(enable_cache=False)
def evaluate_model(model: ClassifierMixin, X_test: np.ndarray, y_test: np.ndarray) -> Output(
    recall = float #accuracy = float, precision = float, recall = float, f1 = float
    ):
    """Model Evaluation and ML metrics register."""
    
    y_pred = model.predict(X_test)

    # metricas
    accuracy, precision, recall, f1 = get_clf_metrics(y_test, y_pred)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)

    return recall


@pipeline(enable_cache=False)
def training_rf_pipeline(
    get_data,
    data_preparation,
    train_model,
    evaluate_model
):

    data = get_data()
    X_train, X_test, y_train, y_test = data_preparation(data = data)
    model = train_model(X_train = X_train, y_train = y_train)
    recall_metric = evaluate_model(model=model, X_test = X_test, y_test = y_test)

    print(recall_metric)



def main():
    training = training_rf_pipeline(
        get_data = load_data(),
        data_preparation = data_preparation(),
        train_model = train_rf(),
        evaluate_model = evaluate_model()
    )
    
    training.run()


if __name__ == '__main__':
    main()