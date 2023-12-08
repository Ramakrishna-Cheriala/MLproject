import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


from exception import CustomException
from logger import logging
from utils import save_object, evaluate_model


@dataclass
class ModelTranierConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTranierConfig()

    def intiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting traning and test input data")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_report: dict = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
            )

            best_model_score = max(sorted(model_report.values()))
            print("\nBest score: ", best_model_score)

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            print("\nModel_name: ", best_model_name)

            best_model = models[best_model_name]
            # print(best_model)

            if best_model_score < 0.6:
                raise CustomException("No best model")

            logging.info("Best model found on both traning and test data")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)
            model_score = r2_score(y_test, predicted)

            return model_score

        except Exception as e:
            CustomException(e, sys)
