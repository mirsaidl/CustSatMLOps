import logging
import pandas as pd
from zenml import step
from src.evaluation_src import Accuracy, Precision, F1Score
from sklearn.base import ClassifierMixin
from typing import Tuple
from typing_extensions import Annotated

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: ClassifierMixin, # if you use DNN keras.src.models.sequential.Sequential,
    x_test: pd.DataFrame,
    y_test: pd.Series
                   ) -> Tuple[
                       Annotated[float, 'Accuracy'],
                       Annotated[float, 'F1 Score'],
                       Annotated[float, 'Precision']
                   ]:
    """Evaluate the model.
    
    Args:
        model: RegressorMixin
        x_test: pd.DataFrame
        y_test: pd.Series
        
    Returns:
        Tuple[float, float]
    """
    try:
        predictions = model.predict(x_test)
        predictions = [round(value) for value in predictions]
                
        accuracy_class = Accuracy() 
        accuracy = accuracy_class.calculate_scores(y_test, predictions)
        mlflow.log_metric('Accuracy', accuracy)
        
        
        precision_class = Precision()
        precision = precision_class.calculate_scores(y_test, predictions)
        mlflow.log_metric('Precision', precision)
        
        f1_class = F1Score()
        f1 = f1_class.calculate_scores(y_test, predictions)
        mlflow.log_metric('F1Score', f1)
        
        return accuracy, f1, precision
    
    except Exception as e:
        logging.error('Error in evaluating model')
        logging.error(str(e))
        raise