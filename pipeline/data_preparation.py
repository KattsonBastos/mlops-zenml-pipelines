
import pandas as pd
import numpy as np

from sklearn.preprocessing       import MinMaxScaler
from sklearn.model_selection     import train_test_split

# zenml importing
from zenml.steps         import step, Output


@step
def change_dtypes(data: pd.DataFrame) -> Output(
    dataframe=pd.DataFrame
    ):
    """Change some coluns data types."""

    dataframe =  data.copy()

    # optimize memory
    dataframe['age'] = dataframe['age'].astype(np.int16)
    dataframe['ap_hi'] = dataframe['ap_hi'].astype(np.int16)
    dataframe['ap_lo'] = dataframe['ap_lo'].astype(np.int16)
    dataframe['cholesterol'] = dataframe['cholesterol'].astype(np.int8)
    dataframe['cardio'] = dataframe['cardio'].astype(np.int8)

    return dataframe

@step
def feature_engineering(data: pd.DataFrame) -> Output(
    dataframe=pd.DataFrame
):
    """Create new features."""

    dataframe = data.copy()

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
