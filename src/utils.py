from src.logger import logging
from src.exception import CustomException
import pickle
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import os, sys
import joblib
# file_path="E:/Project Loan Prediction/Loan-Approval-Prediction/notebook/pickle_file/GradientBoosting_best_model.pkl"

# To load the pickle file
def load_object(file_path):
    try:
        logging.info(f"Loading pickle file from path: {file_path}")
        model = joblib.load(file_path)
        logging.info("Pickle file loaded successfully")
        return model
    except ModuleNotFoundError as e:
        logging.error(f"ModuleNotFoundError: {e}")
        raise CustomException(f"Module not found while loading pickle: {e}", sys)
    except Exception as e:
        logging.exception("Error loading pickle file")
        raise CustomException(f"Error loading pickle file: {e}", sys)
