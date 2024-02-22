import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', "raw_data.xlsx")
    train_data_path: str = os.path.join('artifacts',"train.xlsx")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    
    def init_data_ingestion(self):
        logging.info('Initializing data ingestion')
        try:
            data_path = self.data_ingestion_config.train_data_path
            df = pd.read_excel(data_path)
            logging.info(f'Reading data from training data file: {data_path}')
            
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
            df.to_excel(self.data_ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f'Data ingested and saved to {self.data_ingestion_config.raw_data_path}')
            
            return self.data_ingestion_config.raw_data_path
        except Exception as e:
            logging.error(f'Error during data ingestion: {e}', exc_info=True)
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    obj.init_data_ingestion()
    
    data_transformation = DataTransformation()
    df = data_transformation.init_data_transformation()
    
    model_trainer = ModelTrainer()
    print(model_trainer.init_model_training(df))