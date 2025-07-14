
import os
import sys
import json
import certifi
import pandas as pd
import numpy as np
import pymongo
from dotenv import load_dotenv

from project_package.exception.exception import ProjectException
from project_package.logging.logger import logging


load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)

ca=certifi.where()


class ProjectDataExtract():
    """
    Class to handle project data extraction and insertion into MongoDB.
    """
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise ProjectException(e, sys)
        
    def csv_to_json_convertor(self, file_path):
        """
        Convert CSV file to JSON format.
        """
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise ProjectException(e, sys)
        
    def insert_data_mongodb(self, records, database, collection):
        """
        Insert records into MongoDB collection.
        """
        try:
            self.database = database
            self.collection = collection
            self.records = records

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]

            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return (len(self.records))
        except Exception as e:
            raise ProjectException(e, sys)


if __name__=='__main__':
    FILE_PATH = "Input_Data_File\lending_club_loan.csv"
    DATABASE = "HOSSEINAI"
    COLLECTION = "PROJECTDATA"
    projectobj = ProjectDataExtract()
    records = projectobj.csv_to_json_convertor(file_path = FILE_PATH)
    print(records)
    no_of_records = projectobj.insert_data_mongodb(records, DATABASE, COLLECTION)
    print(no_of_records)