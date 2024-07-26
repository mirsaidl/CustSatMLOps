import logging
from abc import ABC, abstractmethod
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier


class Model(ABC):
    """
    Abstract class for all models
    """
    
    
    @abstractmethod
    def train(self, x_train, y_train):
        """
        Train the model
        """
        pass
        
class XGboostCLassifier(Model):
    """
    XGboost model
    """
    
    def train(self, x_train, y_train, **kwargs):
        """
        Train the model
        """
        try:
            logging.info('Training XGboost model...')
            xgboost = XGBClassifier(**kwargs)
            xgboost.fit(x_train, y_train)
            logging.info('Model trained successfully')
            return xgboost
        except Exception as e:
            logging.error('Error in training XGboost model')
            logging.error(str(e))
            raise
        
class RandomForest(Model):
    """
    Random Forest model
    """
    
    def train(self, x_train, y_train):
        """
        Train the model
        """
        try:
            logging.info('Training Random Forest model...')
            rf = RandomForestClassifier()
            rf.fit(x_train, y_train)
            logging.info('Model trained successfully')
            return rf
        except Exception as e:
            logging.error('Error in training Random Forest model')
            logging.error(str(e))
            raise
        
class DeepNeuralNetwork(Model):
    """
    Deep Neural Network model
    """
    
    def train(self, x_train, y_train, **kwargs):
        """
        Train the model
        """
        try:
            logging.info('Training Deep Neural Network model...')
            model = Sequential()
            model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(24, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(x_train, y_train, epochs=10, batch_size=10)
            logging.info('Model trained successfully')
            
            return model
        except Exception as e:
            logging.error('Error in training Deep Neural Network model')
            logging.error(str(e))
            raise
