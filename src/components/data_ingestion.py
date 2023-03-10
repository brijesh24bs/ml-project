import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass #used to create variables in classes without __init__ method

@dataclass 
class DataIngestionConfig:  
    '''
    SAVING ALL THE DATA INGESTION TO THE FOLLOWING PATHS...
    ALL THE DATA INCOMING WILL BE SAVED TO ARTIFACTS FOLDER
    '''
    train_data_path : str = os.path.join('artifacts','train.csv')
    test_data_path : str = os.path.join('artifacts','test.csv')
    raw_data_path : str = os.path.join('artifacts','data.csv')

class DataIngestion:
    '''
    Use @dataclass only if you want to declare variables but if you want to define functions..got with 
    __init__ method
    '''

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Read the Dataset as Dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("train_test_split initiated")

            train_set , test_set = train_test_split(df, test_size=0.3, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion Completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)    

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()



