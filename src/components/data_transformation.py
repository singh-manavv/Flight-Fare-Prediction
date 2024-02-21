import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

def preprocess_data(df):
    # Other preprocessing steps...
    df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], format="%d/%m/%Y")
    df['Month'] = df['Date_of_Journey'].dt.month
    df['Day'] = df['Date_of_Journey'].dt.day
    
    # Assuming 'Duration' column exists or has been calculated
    # Convert duration to minutes if in a different format
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
    
    # Map total stops
    stops_mapping = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
    df['Total_Stops'] = df['Total_Stops'].map(stops_mapping).fillna(0)
    
    # Drop unnecessary columns
    columns_to_drop = ['Route', 'Date_of_Journey', 'Additional_Info', 'Dep_Time', 'Arrival_Time']
    df.drop(columns=columns_to_drop, inplace=True)
    
    # One-hot encode categorical columns
    categorical_columns = ['Airline', 'Source', 'Destination']
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    return df

@dataclass
class DataTransformation:
    def init_data_transformation(self, raw_data):
        try:
            df = pd.read_excel(raw_data)
            logging.info('Reading the training and test data')
            df = preprocess_data(df)  # Adjust based on your data
            print(df.columns)
            logging.info("Saved preprocessing object.")
            return df
        except Exception as e:
            raise CustomException(e, sys)