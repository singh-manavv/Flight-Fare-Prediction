import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from urllib.parse import urlparse
@dataclass
class ModelTrainerConfig:
    model_file_path: str = os.path.join("artifacts", "model.pkl")
    preprocessed_data_path: str = os.path.join("artifacts", "preprocessed_data.xlsx")

class ModelTrainer:
    def __init__(self):
        self.model_train_config = ModelTrainerConfig()
        
    def eval_metrics(self,actual, pred):
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)
            r2 = r2_score(actual, pred)
            return rmse, mae, r2
        
    def define_models_and_params(self):
        models = {
            "RandomForest": RandomForestRegressor(),
            "AdaBoost": AdaBoostRegressor(),
            "GradientBoost": GradientBoostingRegressor(),
            "LinearRegression": LinearRegression(),
            "DecisionTree": DecisionTreeRegressor(),
            "KNeighbors": KNeighborsRegressor(),
            "XGBoost": XGBRegressor(),
            "CatBoost": CatBoostRegressor(verbose=False),
        }
        params = {
            "DecisionTree": {
                "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                "splitter": ["best", "random"],
                "max_features": ["sqrt", "log2"],
            },
            "RandomForest": {
                "n_estimators": [int(x) for x in np.linspace(start=100, stop=1200, num=12)],
                "max_features": ["log2", "sqrt"],
                "max_depth": [int(x) for x in np.linspace(5, 30, num=6)],
            },
            "GradientBoost": {
                "loss": ["squared_error", "huber", "absolute_error", "quantile"],
                "learning_rate": [0.1, 0.01, 0.05, 0.001],
                "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                "n_estimators": [int(x) for x in np.linspace(start=100, stop=1200, num=12)],
            },
            "LinearRegression": {},
            "KNeighbors": {},
            "XGBoost": {
                "learning_rate": [int(x) for x in np.linspace(start=0.1, stop=0.001, num=5)],
                "n_estimators": [int(x) for x in np.linspace(start=100, stop=1200, num=12)],
            },
            "CatBoost": {
                "depth": [6, 8, 10],
                "learning_rate": [0.01, 0.05, 0.1],
                "iterations": [30, 50, 100],
            },
            "AdaBoost": {
                "learning_rate": [0.1, 0.01, 0.5, 0.001],
                "n_estimators": [int(x) for x in np.linspace(start=100, stop=1200, num=12)],
            },
        }
        return models, params
    
    def init_model_training(self, df):
        logging.info("Initializing model training")
        try:
            X, y = df.drop(columns=["Price"], axis=1), df["Price"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            models, params = self.define_models_and_params()
            model_performance_scores = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_name = max(model_performance_scores, key=model_performance_scores.get)
            best_score = model_performance_scores[best_model_name]

            if best_score < 0.6:
                logging.error("No best model found with acceptable performance")
                raise CustomException("No best model found with acceptable performance", sys)

            logging.info(f"Best model found: {best_model_name} with R2 score: {best_score}")

            best_model = models[best_model_name]
            save_object(self.model_train_config.model_file_path, best_model)
            logging.info("Finished model training and saved model to pickle file")
            
            model_names = list(params.keys())

            actual_model=""

            for model in model_names:
                if best_model_name == model:
                    actual_model = actual_model + model

            best_params = params[actual_model]
            
            mlflow.set_registry_uri("https://dagshub.com/singh-manavv/Flight-Fare-Prediction.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # mlflow

            with mlflow.start_run():

                predicted_qualities = best_model.predict(X_test)

                (rmse, mae, r2) = self.eval_metrics(y_test, predicted_qualities)

                mlflow.log_params(best_params)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)


                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
                else:
                    mlflow.sklearn.log_model(best_model, "model")
            return best_score
        except Exception as e:
            logging.error(f"Error in model training: {e}", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    model_trainer_config = ModelTrainerConfig()
    model_trainer = ModelTrainer(model_trainer_config)
    df = pd.read_excel(model_trainer_config.preprocessed_data_path)
    r2_score = model_trainer.init_model_training(df)
    print(f"Best model R2 score on test data: {r2_score}")
