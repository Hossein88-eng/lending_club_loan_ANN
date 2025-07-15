import sys

from project_package.exception.exception import ProjectException
from project_package.logging.logger import logging
from project_package.components.data_ingestion import DataIngestion
from project_package.components.data_validation import DataValidation
from project_package.components.data_transformation import DataTransformation
from project_package.components.model_trainer import ModelTrainer
from project_package.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from project_package.entity.config_entity import TrainingPipelineConfig
from project_package.entity.config_entity import ModelTrainerConfig



if __name__=='__main__':
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig=DataIngestionConfig(trainingpipelineconfig)
        data_ingestion=DataIngestion(dataingestionconfig)
        logging.info("Data ingestion started...")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed")
        print(dataingestionartifact)
        
        data_validation_config = DataValidationConfig(trainingpipelineconfig)
        data_validation=DataValidation(dataingestionartifact, data_validation_config)
        logging.info("Data validation started...")
        data_validation_artifact=data_validation.initiate_data_validation()
        logging.info("Data validation completed")
        print(data_validation_artifact)

        data_transformation_config = DataTransformationConfig(trainingpipelineconfig)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        logging.info("Data transformation started...")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data transformation completed")
        print(data_transformation_artifact)

        logging.info("Model training started...")
        model_trainer_config = ModelTrainerConfig(trainingpipelineconfig)
        model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model training completed!")
        print(model_trainer_artifact)

    except Exception as e:
           raise ProjectException(e,sys)
