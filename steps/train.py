import logging
import pandas as pd
from zenml import step
import keras

import mlflow
from zenml.client import Client

from src.model_dep import RandomForest, XGboostCLassifier, DeepNeuralNetwork
from sklearn.base import ClassifierMixin
from .config import ModelNameConfig


experiment_tracker = Client().active_stack.experiment_tracker


# 3 Train model
@step(experiment_tracker=experiment_tracker.name)
def train_model(
   x_train: pd.DataFrame,
   y_train: pd.Series,
   config: ModelNameConfig
   ) -> ClassifierMixin: # or if you want to use DNN keras.src.models.sequential.Sequential
   """
    Trains a model on the given data
    
    Args:
         x_train: pd.DataFrame
         y_train: pd.Series
         config: ModelNameConfig
    Returns:
         RegressorMixin or Sequential
   """
   try:
      model = None
      if config.model_name == "RandomForest":
         mlflow.sklearn.autolog( )
         model = RandomForest()
         train_model = model.train(x_train, y_train)
         return train_model
      if config.model_name == "XGboost":
         mlflow.sklearn.autolog( )
         model = XGboostCLassifier()
         y_train_adjusted = y_train - 1
         train_model = model.train(x_train, y_train_adjusted)
         return train_model
      if config.model_name == "DeepNeuralNetwork":
         mlflow.keras.autolog()
         model = DeepNeuralNetwork()
         train_model = model.train(x_train, y_train)
         
         return train_model
      else:
         raise ValueError("Model not implemented")
   except Exception as e:
      logging.error('Error in training model')
      logging.error(str(e))
      raise
   
    
    