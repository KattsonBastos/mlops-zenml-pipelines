from zenml.steps         import step, Output


@step
def deployment_trigger(recall: float) -> Output(
    to_deploy = bool
    ):
    """Deploying only if recall is higher than 70%."""

    to_deploy = recall > 0.7
    return to_deploy