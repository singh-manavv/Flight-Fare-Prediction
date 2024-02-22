import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.logger import logging  # Ensure this matches your import
from src.exception import CustomException
@dataclass
class DataTransformationConfig:
    raw_data_path: str= os.path.join('artifacts', 'raw_data.xlsx')
    preprocessed_data_path: str= os.path.join('artifacts', 'preprocessed_data.xlsx')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        
    @staticmethod
    def preprocess_data(df):
        try:
            logging.info("Starting data preprocessing")
            df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], format="%d/%m/%Y")
            df['Month'] = df['Date_of_Journey'].dt.month
            df['Day'] = df['Date_of_Journey'].dt.day
            
            if 'Duration' in df.columns and df['Duration'].dtype == object:
                def convert_duration(duration):
                    parts = duration.split()
                    minutes = 0
                    for part in parts:
                        if 'h' in part:
                            minutes += int(part.replace('h', '')) * 60
                        elif 'm' in part:
                            minutes += int(part.replace('m', ''))
                    return minutes
                df['Duration'] = df['Duration'].apply(convert_duration)
            
            stops_mapping = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
            df['Total_Stops'] = df['Total_Stops'].map(stops_mapping).fillna(0)
            
            columns_to_drop = ['Route', 'Date_of_Journey', 'Additional_Info', 'Dep_Time', 'Arrival_Time']
            df.drop(columns=columns_to_drop, inplace=True)
            
            categorical_columns = ['Airline', 'Source', 'Destination']
            df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
            
            logging.info("Data preprocessing completed successfully")
            return df
        except Exception as e:
            logging.error("Error during data preprocessing: %s", e)
            raise CustomException(e, sys)
        
    def init_data_transformation(self):
        try:
            logging.info('Reading the training and test data from %s', self.transformation_config.raw_data_path)
            df = pd.read_excel(self.transformation_config.raw_data_path)
            df = self.preprocess_data(df)  
            logging.info("Data transformation completed successfully.")
            df.to_excel(self.transformation_config.preprocessed_data_path, index=False, header=True)
            return df
        except Exception as e:
            logging.error("Error in init_data_transformation: %s", e)
            raise CustomException(e, sys)