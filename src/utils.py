from src.logger import logging
from src.exception import CustomException
import pickle
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import os, sys

# To load the pickle file
def load_object(file_path):
    try:
        logging.info(f"Loading pickle file from path: {file_path}")
        with open(file_path, 'rb') as file_obj:
            obj = pickle.load(file_obj)
            logging.info("Pickle file loaded successfully")
            return obj
    except Exception as e:
        logging.exception("Error loading pickle file")
        raise CustomException(e, sys)
