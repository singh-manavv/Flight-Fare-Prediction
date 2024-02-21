import os 
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from src.exception import CustomException
import pickle

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train,y_train,X_test,y_test,models,param):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            rs = RandomizedSearchCV(estimator = model,param_distributions = para,scoring='neg_mean_squared_error',cv=5,n_iter=10,n_jobs=-1,random_state=42,verbose=2)
            # gs = GridSearchCV(estimator = model,param_grid= para, cv=5,n_jobs=-1,verbose=2)
            rs.fit(X_train,y_train)
            print(rs.best_params_)
            model.set_params(**rs.best_params_)
            model.fit(X_train,y_train)
            print(X_train.columns)
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
    except Exception as e:
        raise CustomException(e,sys)