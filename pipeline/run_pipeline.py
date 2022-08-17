from zenml.pipelines import pipeline

from evaluation          import evaluate_model
from data_loading        import load_data
from data_preparation    import change_dtypes, feature_engineering, split_train_test, scale_training_data
from training            import train_rf


@pipeline
def training_pipeline(
    get_data,
    change_type,
    create_features,
    get_train_test_data,
    scale_data,
    train_model,
    evaluate_model
):

    data = get_data()

    data = change_type(data = data)

    data = create_features(data = data)

    X_train, X_test, y_train, y_test = get_train_test_data(data = data)

    X_train, X_test = scale_data(X_train = X_train, X_test = X_test)

    model = train_model(X_train = X_train, y_train = y_train)

    results = evaluate_model(model=model, X_test = X_test, y_test = y_test)

    print(results)

def main():
    training = training_pipeline(
        get_data=load_data(),
        change_type=change_dtypes(),
        create_features=feature_engineering(),
        get_train_test_data=split_train_test(),
        scale_data=scale_training_data(),
        train_model=train_rf(),
        evaluate_model=evaluate_model()
    )
    
    training.run()


if __name__ == '__main__':
    main()