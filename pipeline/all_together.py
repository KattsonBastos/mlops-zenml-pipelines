import mlflow

import numpy            as np
import pandas           as pd

from sklearn.metrics                import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble               import RandomForestClassifier
from sklearn.preprocessing          import MinMaxScaler
from sklearn.model_selection        import train_test_split

# zenml importing
from zenml.steps                                        import step, Output, BaseStepConfig
from sklearn.base                                       import ClassifierMixin
from zenml.pipelines                                    import pipeline
from zenml.integrations.mlflow.mlflow_step_decorator    import enable_mlflow


@step
def load_data() -> Output(
    data=pd.DataFrame
    ):

    """Load a dataset."""

    data = pd.read_csv('../data/cardio_train_sampled.csv')

    return data



@step
def feature_engineering(data: pd.DataFrame) -> Output(
    dataframe=pd.DataFrame
):
    """Create new features."""

    dataframe = data.copy()

    dataframe['age'] = dataframe['age'].astype(np.int16)

    dataframe['ap_hi'] = dataframe['ap_hi'].astype(np.int16)

    dataframe['ap_lo'] = dataframe['ap_lo'].astype(np.int16)

    dataframe['cholesterol'] = dataframe['cholesterol'].astype(np.int8)

    dataframe['cardio'] = dataframe['cardio'].astype(np.int8)

    # age
    dataframe['age'] = round(dataframe['age'] / 365).astype(int)

    # weight
    dataframe = dataframe[dataframe['weight'] >= 33]

    # ap_hi
    dataframe = dataframe[(dataframe['ap_hi'] >=85) & (dataframe['ap_hi'] <= 240)]

    # ap_lo
    dataframe = dataframe[(dataframe['ap_lo'] >=65) & (dataframe['ap_lo'] <= 150)]


    return dataframe


@step
def split_train_test(data: pd.DataFrame) -> Output(
    X_train = np.ndarray, X_test = np.ndarray, y_train = np.ndarray, y_test = np.ndarray
    ):
    """Receive a pandas DataFrame and return the data splitted into train and test."""

    X = data.drop('cardio', axis = 1).values

    y = data['cardio'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

    return X_train, X_test, y_train, y_test


@step
def scale_training_data(X_train: np.ndarray, X_test: np.ndarray) -> Output(
    X_train_scaled = np.ndarray, X_test_scaled = np.ndarray
    ):
    """Scale X variables."""

    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


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

    return recall


@pipeline(enable_cache=False)
def training_rf_pipeline(
    get_data,
    feature_engineering,
    get_train_test_data,
    scale_data,
    train_model,
    evaluate_model
):

    data = get_data()

    data = feature_engineering(data = data)

    X_train, X_test, y_train, y_test = get_train_test_data(data = data)

    X_train, X_test = scale_data(X_train = X_train, X_test = X_test)

    model = train_model(X_train = X_train, y_train = y_train)

    recall_metric = evaluate_model(model=model, X_test = X_test, y_test = y_test)


def main():
    training = training_rf_pipeline(
        get_data = load_data(),
        feature_engineering = feature_engineering(),
        get_train_test_data = split_train_test(),
        scale_data = scale_training_data(),
        train_model = train_rf(),
        evaluate_model = evaluate_model()
    )
    
    training.run()


if __name__ == '__main__':
    main()