from zenml.repository import Repository

def test_deployed_model():
    repo = Repository()

    model_deployer = repo.active_stack.model_deployer

    services = model_deployer.find_model_server(
        pipeline_name="training_rf_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        running=True,
    )

    service = services[0]
    print(
        'model service status: ', 
        service.check_status()
    )

    p = repo.get_pipeline("training_rf_pipeline")
    
    last_run = p.runs[-1]
    X_test = last_run.steps[4].outputs["X_test_scaled"].read()
    y_test = last_run.steps[3].outputs["y_test"].read()
    y_pred = service.predict(X_test[0:1])

    print(
        f"Model prediction: {y_pred[0]}.\nTrue label: {y_test[0]}"
    )


if __name__ == '__main__':
    test_deployed_model()