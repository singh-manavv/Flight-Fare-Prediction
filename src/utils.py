import os
import sys
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
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

def evaluate_model(model, params, X_train, y_train, X_test, y_test):
    rs = RandomizedSearchCV(estimator=model, param_distributions=params, scoring='neg_mean_squared_error', cv=5, n_iter=10, n_jobs=-1, random_state=42, verbose=2)
    rs.fit(X_train, y_train)
    model.set_params(**rs.best_params_)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    test_model_score = r2_score(y_test, y_test_pred)
    
    return test_model_score

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for model_name, model in models.items():
            model_params = params[model_name]
            test_model_score = evaluate_model(model, model_params, X_train, y_train, X_test, y_test)
            report[model_name] = test_model_score
            
        return report
    except Exception as e:
        raise CustomException(e, sys)