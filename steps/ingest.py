import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    Class to ingest data from a given path
    """
    def __init__(self, path: str):
        self.path = path

    def get_data(self):
        logging.info(f'Reading data from {self.path}')
        
        return pd.read_csv(self.path)
       
# 1. Ingest data
@step
def ingest_data(path: str) -> pd.DataFrame:
    """ 
    Ingest data from a given path 
    
    Args:
       path: str: Path to the data file
    Returns:
       pd.DataFrame: Dataframe containing
    
    """
    try:
        ingest_data = IngestData(path)
        df = ingest_data.get_data()
        
        return df
    
    except Exception as e:
        logging.error(f'Failed to ingest data: {e}')
        return e
