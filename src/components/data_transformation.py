import os
import sys 
import pandas as pd 
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from src.logger import logging
from src.exception import CustomException


@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
    
    def get_transformer_object(self):
        logging.info('Starting Data transformation')
        try:
            df = pd.read_csv(os.path.join('artifacts','train.csv'))
            print(df.shape)
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == '__main__':
    obj = DataTransformation()
    obj.get_transformer_object()
