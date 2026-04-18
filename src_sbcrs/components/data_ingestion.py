import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src_sbcrs.exception import CustomException
from src_sbcrs.logger import logging
from src_sbcrs.components.data_transformation import DataTransformation
from src_sbcrs.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook\\data\\WA_Fn-UseC_-Telco-Customer-Churn.csv')
            logging.info('Dataset read successfully as dataframe')

            # ----- ALIGNMENT WITH NOTEBOOK -----
            logging.info("Dropping irrelevant columns: 'customerID' and 'gender'")
            df.drop(['customerID', 'gender'], axis=1, inplace=True)
            logging.info(f"Columns dropped. Remaining shape: {df.shape}")

            logging.info("Converting 'TotalCharges' to numeric (errors='coerce')")
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            logging.info("'TotalCharges' conversion completed.")
            # ---------------------------------

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw processed data (optional)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw processed data saved to {self.ingestion_config.raw_data_path}")

            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=53, stratify=df['Churn']
            )
            logging.info(f"Train set shape: {train_set.shape}, Test set shape: {test_set.shape}")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)