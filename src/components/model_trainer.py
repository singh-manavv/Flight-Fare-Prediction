import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import numpy as np
from sklearn.model_selection import train_test_split
@dataclass 
class ModelTrainerConfig:
    model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_train_config = ModelTrainerConfig()
        
    def init_model_training(self,df):
        try:
            logging.info('Initializing model training')
            # print(df.columns)
            y = df["Price"]
            X = df.drop(columns=['Price'],axis=1)
            # print(X.columns)
            X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=42)
            # print(type(X_train),type(X_test))
            models = {
                'AdaBoost': AdaBoostRegressor(),
                'RandomForest': RandomForestRegressor(),
                'GradientBoost': GradientBoostingRegressor(),
                'LinearRegression': LinearRegression(),
                'DecisionTree': DecisionTreeRegressor(),
                'CatBoost': CatBoostRegressor(verbose=False),
                'KNeighbors': KNeighborsRegressor(),
                'XGBoost' : XGBRegressor()
            }
            # model = RandomForestRegressor()
            params = {
                "DecisionTree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "RandomForest":{
                    'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)],
                    'max_features': ['log2', 'sqrt'],
                    # 'min_samples_leaf': [1, 2, 5, 10],
                    # 'min_samples_split': [2, 5, 10, 15, 100],
                    'max_depth' : [int(x) for x in np.linspace(5, 30, num = 6)],
                },
                "GradientBoost":{
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
                },
                "LinearRegression":{},
                "KNeighbors":{},
                "XGBoost":{
                    'learning_rate':[int(x) for x in np.linspace(start = 0.1, stop = 0.001, num = 5)],
                    'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
                },
                "CatBoost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
                }
            }
            
            model_report: dict = evaluate_models(X_train = X_train,y_train = y_train,X_test = X_test,y_test = y_test,models=models,param=params)
            
            best_model_score = max(sorted(model_report.values()))
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException('No best model found')
            
            logging.info(f'Best model found on both training and test data {best_model_name}')
            save_object(
                file_path= self.model_train_config.model_file_path,
                obj= best_model
            )
            print(X_test.columns)
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            print(best_model)
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)