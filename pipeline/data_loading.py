
import pandas as pd

# zenml importing
from zenml.steps         import step, Output

@step
def load_data() -> Output(
    data=pd.DataFrame
    ):

    data = pd.read_csv('../data/cardio_train_sampled.csv')

    return data