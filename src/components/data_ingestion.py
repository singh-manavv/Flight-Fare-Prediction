import os 
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import pandas as pd 
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    raw_data_path: str =os.path.join('artifacts',"data.xlsx")
class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    
    def init_data_ingestion(self):
        logging.info('Initializing data ingestion')
        try:
            df = pd.read_excel('notebook/train.xlsx')
            logging.info('Reading data from training data file')
            
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
            df.to_excel(self.data_ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Data ingestion completed!')
            
            return (
                self.data_ingestion_config.raw_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == '__main__':
    obj = DataIngestion()
    raw_data = obj.init_data_ingestion()
    
    data_transformation = DataTransformation()
    df= data_transformation.init_data_transformation(raw_data)
    
    model_trainer = ModelTrainer()
    print(model_trainer.init_model_training(df))