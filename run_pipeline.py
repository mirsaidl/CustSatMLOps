from pipelines.training_pipeline import training_pipeline
from zenml.client import Client
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

if __name__ == "__main__":
    # Run the pipeline
    print(
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "We can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here we'll also be able to compare the two runs.)"
    )
    
    """
    mlflow ui --backend-store-uri uri
    """
    training_pipeline(data_path="data/data.csv")