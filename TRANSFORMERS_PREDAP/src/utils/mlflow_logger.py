try:
    import mlflow
except Exception:
    mlflow = None

class MLflowLogger:
    def __init__(self, active: bool = True):
        self.active = active and (mlflow is not None)

    def log_artifact(self, path: str, artifact_path: str = None):
        if not self.active:
            return
        if artifact_path:
            mlflow.log_artifact(path, artifact_path=artifact_path)
        else:
            mlflow.log_artifact(path)
    
    def log_metric(self, key: str, value, step: int = None):
        if not self.active:
            return
        if step is not None:
            mlflow.log_metric(key, value, step=step)
        else:
            mlflow.log_metric(key, value)

    def log_metrics(self, metrics: dict):
        if not self.active:
            return
        mlflow.log_metrics(metrics)

    def log_param(self, key: str, value):
        if not self.active:
            return
        mlflow.log_param(key, value)

    def log_params(self, params: dict):
        if not self.active:
            return
        mlflow.log_params(params)

    def start_run(self, **kwargs):
        if not self.active:
            return None
        return mlflow.start_run(**kwargs)

    def end_run(self):
        if not self.active:
            return
        mlflow.end_run()
