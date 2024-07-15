import os, sys
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
from dataclasses import dataclass
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# notebook\pickle_file\LogisticRegression_Smote.pkl
class predictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            logging.info("Prediction started")
            model_path = os.path.join("notebook/pickle_file", "LogisticRegression_Smote.pkl")

            logging.info(f"Loading model from path: {model_path}")
            model = load_object(model_path)

            logging.info("Model loaded successfully")
            pred = model.predict(features)
            logging.info(f"Prediction completed: {pred}")

            return pred

        except Exception as e:
            logging.exception("Error in predictionPipeline")
            raise CustomException(e, sys)


class customClass:
    def __init__(self, Gender: object, Married: object, Dependents: object, Education: object, Self_Employed: object, ApplicantIncome: int, CoapplicantIncome: float, LoanAmount: float, Loan_Amount_Term: float, Credit_History: float, Property_Area: object):
        self.Gender = Gender
        self.Married = Married
        self.Dependents = Dependents
        self.Education = Education
        self.Self_Employed = Self_Employed
        self.ApplicantIncome = ApplicantIncome
        self.CoapplicantIncome = CoapplicantIncome
        self.LoanAmount = LoanAmount
        self.Loan_Amount_Term = Loan_Amount_Term
        self.Credit_History = Credit_History
        self.Property_Area = Property_Area

    def get_data_frame(self):
        try:
            custom_input = {
                "Gender": [self.Gender],
                "Married": [self.Married],
                "Dependents": [self.Dependents],
                "Education": [self.Education],
                "Self_Employed": [self.Self_Employed],
                "ApplicantIncome": [self.ApplicantIncome],
                "CoapplicantIncome": [self.CoapplicantIncome],
                "LoanAmount": [self.LoanAmount],
                "Loan_Amount_Term": [self.Loan_Amount_Term],
                "Credit_History": [self.Credit_History],
                "Property_Area": [self.Property_Area]
            }

            data = pd.DataFrame(custom_input)
            logging.info(f"Data frame created: {data}")

            return data
        except Exception as e:
            logging.exception("Error in customClass.get_data_frame")
            raise CustomException(e, sys)
