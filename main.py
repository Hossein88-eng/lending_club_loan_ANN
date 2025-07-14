from project_package.exception.exception import ProjectException
from project_package.logging.logger import logging
from project_package.components.data_ingestion import DataIngestion
from project_package.entity.config_entity import DataIngestionConfig
from project_package.entity.config_entity import TrainingPipelineConfig

import sys

if __name__=='__main__':
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig=DataIngestionConfig(trainingpipelineconfig)
        data_ingestion=DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion...")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        logging.info("Data Initiation Completed")
        print(dataingestionartifact)

    except Exception as e:
           raise ProjectException(e,sys)
