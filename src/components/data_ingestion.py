import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from exception import CustomException
from logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from data_transformation import DataTransformation, DataTransformationConfig
from model_trainer import ModelTrainer, ModelTranierConfig


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def intiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            df = pd.read_csv(
                "C:\\Users\\ramak\\OneDrive\\Desktop\\Project\\notebook\\data\\stud.csv"
            )

            logging.info("read the dataset as dataframe")

            # print("Current Working Directory:", os.getcwd())
            # print(parent_dir, "\n", current_dir)
            artifacts_dir = os.path.join(os.getcwd(), "artifacts")
            # print("Artifacts Directory:", artifacts_dir)
            os.makedirs(artifacts_dir, exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train_test_split Initiated")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=24)

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )

            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            logging.info("Ingestion of data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.intiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data
    )

    model_trainer = ModelTrainer()
    print(
        "\nBest Model Score: ", model_trainer.intiate_model_trainer(train_arr, test_arr)
    )
