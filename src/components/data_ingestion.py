import os 
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import pandas as pd 
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    
    def init_data_ingestion(self):
        logging.info('Initializing data ingestion')
        try:
            df = pd.read_excel('notebook/train.xlsx')
            logging.info('Reading data from training data file')
            
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            
            logging.info('Splitting the data into training and test datasets')
            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)
            
            logging.info('Data ingestion completed!')
            
            return {
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            }
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == '__main__':
    obj = DataIngestion()
    obj.init_data_ingestion()