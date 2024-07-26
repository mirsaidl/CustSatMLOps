import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score


class Evaluation(ABC):
    """
    Abstract class for all evaluations
    """
    
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Evaluate the model
        
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            score: float
        """
        pass
        
class Accuracy:
    """
    Accuracy
    """
    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Evaluate the model by Accuracy
        
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            score: float
        """
        try:
            logging.info('Calculating Accuracy...')
            accuracy = accuracy_score(y_true, y_pred)
            logging.info(f'Accuracy: {accuracy}')
            
            return accuracy
        except Exception as e:
            logging.error('Error in calculating Accuracy')
            logging.error(str(e))
            raise
        
class Precision:
    """
    Precision
    """
    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Evaluate the model by Precision
        
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            score: float
        """
        try:
            logging.info('Calculating Precision...')
            precision = precision_score(y_true, y_pred, average='macro')
            logging.info(f'Precision: {precision}')
            
            return precision
        except Exception as e:
            logging.error('Error in calculating Precision')
            logging.error(str(e))
            raise

class F1Score:
    """
    F1 Score
    """
    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Evaluate the model by F1 Score
        
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            score: float
        """
        try:
            logging.info('Calculating F1 Score...')
            f1 = f1_score(y_true, y_pred, average='macro')
            logging.info(f'F1 Score: {f1}')
            
            return f1
        except Exception as e:
            logging.error('Error in calculating F1 Score')
            logging.error(str(e))
            raise