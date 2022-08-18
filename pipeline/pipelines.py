from zenml.pipelines import pipeline


@pipeline(enable_cache=False)
def training_rf_pipeline(
    get_data,
    feature_engineering,
    get_train_test_data,
    scale_data,
    train_model,
    evaluate_model,
    deployment_trigger,
    model_deployer
):

    data = get_data()

    data = feature_engineering(data = data)

    X_train, X_test, y_train, y_test = get_train_test_data(data = data)

    X_train, X_test = scale_data(X_train = X_train, X_test = X_test)

    model = train_model(X_train = X_train, y_train = y_train)

    recall_metric = evaluate_model(model=model, X_test = X_test, y_test = y_test)

    deployment_decision = deployment_trigger(recall_metric)  # new
    
    model_deployer(deployment_decision, model)